"""Tests for RePaRe-Cycle.

Covers:
- Tarjan's SCC algorithm correctness
- CyclicLinearSEM sampling and intervention propagation
- random_directed_graph stability guarantee
- PartitionDgModelIvn: soft-only enforcement, basic fitting on cyclic chains
"""

import networkx as nx
import numpy as np
import pytest

from repare_cycle.graph import CyclicLinearSEM, random_directed_graph, tarjan_scc
from repare_cycle.repare_cycle import PartitionDgModelIvn, _get_totally_ordered_partition


# ---------------------------------------------------------------------------
# Tarjan's SCC
# ---------------------------------------------------------------------------


def test_tarjan_scc_dag():
    """A DAG has only singleton SCCs."""
    g = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    sccs = tarjan_scc(g)
    assert len(sccs) == 3
    assert all(len(scc) == 1 for scc in sccs)


def test_tarjan_scc_single_cycle():
    """A 3-cycle is one SCC."""
    g = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    sccs = tarjan_scc(g)
    assert len(sccs) == 1
    assert set(sccs[0]) == {0, 1, 2}


def test_tarjan_scc_mixed():
    """Graph with one 2-cycle (SCC of size 2) and two singletons."""
    g = nx.DiGraph([(0, 1), (1, 0), (1, 2), (2, 3)])
    sccs = tarjan_scc(g)
    assert len(sccs) == 3
    scc_sets = [frozenset(s) for s in sccs]
    assert frozenset({0, 1}) in scc_sets
    assert frozenset({2}) in scc_sets
    assert frozenset({3}) in scc_sets


def test_tarjan_scc_matches_networkx():
    """Tarjan output matches nx.strongly_connected_components on a random graph."""
    g = nx.erdos_renyi_graph(20, 0.3, seed=42, directed=True)
    tarjan_result = {frozenset(s) for s in tarjan_scc(g)}
    nx_result = {frozenset(s) for s in nx.strongly_connected_components(g)}
    assert tarjan_result == nx_result


# ---------------------------------------------------------------------------
# CyclicLinearSEM
# ---------------------------------------------------------------------------


def test_cyclic_sem_dag_matches_expected():
    """A single-edge DAG 0→1: X_1 = w * X_0 + N_1."""
    w = 0.6
    weights = np.array([[0.0, w], [0.0, 0.0]])
    rng = np.random.default_rng(0)
    model = CyclicLinearSEM(weights, means=(0.0, 0.0), variances=(1.0, 1.0), rng=rng)
    data = model.sample(100_000)
    # Covariance X_0, X_1 should be w * Var(X_0)
    cov = np.cov(data.T)
    assert abs(cov[0, 1] - w) < 0.05


def test_cyclic_sem_stability_error():
    """Spectral radius >= 1 raises ValueError (requires a cyclic matrix)."""
    # 2-cycle with weights 1.1: eigenvalues = ±1.1, spectral radius = 1.1 >= 1
    weights = np.array([[0.0, 1.1], [1.1, 0.0]])
    with pytest.raises(ValueError, match="Spectral radius"):
        CyclicLinearSEM(weights)


def test_cyclic_sem_soft_intervention_changes_distribution():
    """Soft intervention on node 0 shifts mean of X_0 and propagates to X_1."""
    weights = np.array([[0.0, 0.5], [0.0, 0.0]])
    model = CyclicLinearSEM(weights, means=(0.0, 0.0), variances=(1.0, 1.0), rng=np.random.default_rng(1))
    obs = model.sample(5000)
    ivn = model.sample(5000, shift_interventions={0: (5.0, 0.1)})
    # Both X_0 and X_1 should shift
    assert ivn[:, 0].mean() > obs[:, 0].mean() + 3
    assert ivn[:, 1].mean() > obs[:, 1].mean() + 1


def test_cyclic_sem_2cycle_intervention_propagates():
    """In a 2-cycle 0⇔1, intervening on 0 shifts both 0 and 1."""
    weights = np.array([[0.0, 0.4], [0.4, 0.0]])
    model = CyclicLinearSEM(weights, means=(0.0, 0.0), variances=(1.0, 1.0), rng=np.random.default_rng(2))
    obs = model.sample(10_000)
    ivn = model.sample(10_000, shift_interventions={0: (10.0, 0.1)})
    assert ivn[:, 0].mean() > obs[:, 0].mean() + 3
    assert ivn[:, 1].mean() > obs[:, 1].mean() + 1  # propagates through cycle


# ---------------------------------------------------------------------------
# random_directed_graph
# ---------------------------------------------------------------------------


def test_random_directed_graph_stability():
    """Generated graphs always have spectral radius < 1."""
    for seed in range(5):
        _, weights = random_directed_graph(10, 0.4, seed=seed)
        sr = np.max(np.abs(np.linalg.eigvals(weights)))
        assert sr < 1.0, f"seed={seed}: spectral radius={sr:.4f}"


def test_random_directed_graph_er_has_cycles():
    """ER directed graphs at density=0.5 on 10 nodes almost surely have cycles."""
    _, weights = random_directed_graph(10, 0.5, graph_type="er", seed=7)
    g = nx.DiGraph(weights.astype(bool))
    sccs = list(nx.strongly_connected_components(g))
    # Not all SCCs are singletons (cycles exist)
    assert any(len(scc) > 1 for scc in sccs)


# ---------------------------------------------------------------------------
# PartitionDgModelIvn: hard intervention rejection
# ---------------------------------------------------------------------------


def test_hard_intervention_rejected():
    """Hard interventions raise ValueError."""
    weights = np.array([[0.0, 0.5], [0.0, 0.0]])
    model_sem = CyclicLinearSEM(weights, rng=np.random.default_rng(0))
    obs = model_sem.sample(200)
    ivn = model_sem.sample(200, shift_interventions={0: (2.0, 1.0)})

    data_dict = {
        "obs": (obs, set(), "obs"),
        "env1": (ivn, {0}, "hard"),  # <- should be rejected
    }
    model = PartitionDgModelIvn()
    with pytest.raises(ValueError, match="Hard interventions"):
        model.fit(data_dict)


# ---------------------------------------------------------------------------
# PartitionDgModelIvn: fitting on a simple chain (DAG as special case)
# ---------------------------------------------------------------------------


def test_fit_chain_gaussian():
    """Chain 0→1→2 with soft interventions: should recover three singleton blocks."""
    n_samples = 500
    rng = np.random.default_rng(10)

    def sample(shift_node=None, shift=5.0):
        x0 = rng.normal(size=n_samples)
        if shift_node == 0:
            x0 += shift
        x1 = 0.7 * x0 + rng.normal(scale=0.4, size=n_samples)
        if shift_node == 1:
            x1 = rng.normal(loc=shift, size=n_samples)
        x2 = 0.6 * x1 + rng.normal(scale=0.4, size=n_samples)
        if shift_node == 2:
            x2 += shift
        return np.column_stack([x0, x1, x2])

    data_dict = {
        "obs": (sample(), set(), "obs"),
        "ivn0": (sample(shift_node=0), {0}, "soft"),
        "ivn1": (sample(shift_node=1), {1}, "soft"),
        "ivn2": (sample(shift_node=2), {2}, "soft"),
    }

    model = PartitionDgModelIvn()
    model.fit(data_dict, assume="gaussian")
    assert set(model.dag.nodes) == {(0,), (1,), (2,)}


# ---------------------------------------------------------------------------
# PartitionDgModelIvn: SCC grouping
# ---------------------------------------------------------------------------


def test_scc_nodes_grouped_together():
    """Nodes in a 2-cycle should land in the same partition block."""
    # 2-cycle: 0⇔1, downstream node 2
    weights = np.array([
        [0.0, 0.4, 0.0],
        [0.4, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ])
    model_sem = CyclicLinearSEM(weights, means=(0.0, 0.0), variances=(0.8, 1.2), rng=np.random.default_rng(3))

    n_samples = 2000
    obs = model_sem.sample(n_samples)
    # Soft intervention on node 2 (downstream, doesn't affect the cycle)
    ivn2 = model_sem.sample(n_samples, shift_interventions={2: (5.0, 1.0)})

    data_dict = {
        "obs": (obs, set(), "obs"),
        "ivn2": (ivn2, {2}, "soft"),
    }
    model = PartitionDgModelIvn()
    model.fit(data_dict, assume="gaussian")

    # Nodes 0 and 1 are in the same SCC → should be in the same partition block
    partition_dict = {}
    for part in model.dag.nodes:
        for node in part:
            partition_dict[node] = part
    assert partition_dict[0] == partition_dict[1], (
        f"Nodes 0 and 1 should be in the same block. Got: {partition_dict}"
    )


# ---------------------------------------------------------------------------
# _get_totally_ordered_partition
# ---------------------------------------------------------------------------


def test_totally_ordered_partition_basic():
    """Three variables: first two changed under ivn1, third only under ivn2."""
    masks = {
        "ivn1": np.array([True, True, False]),
        "ivn2": np.array([False, False, True]),
    }
    partition = _get_totally_ordered_partition(masks)
    # Should split into [{0,1}, {2}] or [{2}, {0,1}]
    sets = [frozenset(p) for p in partition]
    assert frozenset({0, 1}) in sets
    assert frozenset({2}) in sets


# ---------------------------------------------------------------------------
# Multi-SCC graph generation
# ---------------------------------------------------------------------------


def test_multi_scc_graph_structure():
    """random_multi_scc_graph creates the requested number of non-trivial SCCs."""
    from repare_cycle.graph import random_multi_scc_graph

    graph, weights = random_multi_scc_graph(
        n_sccs=3, scc_sizes=3, n_singletons=2, inter_density=0.3, seed=42,
    )
    n = weights.shape[0]
    assert n == 3 * 3 + 2  # 11 nodes

    sccs = list(nx.strongly_connected_components(graph))
    nontrivial = [s for s in sccs if len(s) > 1]
    singletons = [s for s in sccs if len(s) == 1]
    assert len(nontrivial) == 3
    assert len(singletons) == 2
    # Each non-trivial SCC should have exactly 3 nodes
    for scc in nontrivial:
        assert len(scc) == 3


def test_multi_scc_oica_recovery():
    """OICA correctly identifies multiple distinct SCCs from observational data."""
    from repare_cycle.graph import random_multi_scc_graph, CyclicLinearSEM
    from repare_cycle.oica import oica_scc_partition

    graph, weights = random_multi_scc_graph(
        n_sccs=3, scc_sizes=3, n_singletons=0, inter_density=0.3, seed=42,
    )
    true_sccs = {frozenset(s) for s in nx.strongly_connected_components(graph)}

    sem = CyclicLinearSEM(weights, rng=np.random.default_rng(42), noise_dist="laplace")
    obs = sem.sample(10000)

    est_sccs = set(oica_scc_partition(obs, threshold=0.1, random_state=42))
    assert est_sccs == true_sccs


# ---------------------------------------------------------------------------
# random_cyclic_graph (unified generator for main experiment)
# ---------------------------------------------------------------------------


def test_random_cyclic_graph_dag():
    """num_cycles=0 produces a DAG (no non-trivial SCCs)."""
    from repare_cycle.graph import random_cyclic_graph

    graph, weights = random_cyclic_graph(d=10, num_cycles=0, density=0.5, seed=0)
    assert weights.shape == (10, 10)
    sccs = list(nx.strongly_connected_components(graph))
    # All SCCs should be singletons
    for scc in sccs:
        assert len(scc) == 1


def test_random_cyclic_graph_exact_cycles():
    """random_cyclic_graph produces exactly num_cycles non-trivial SCCs."""
    from repare_cycle.graph import random_cyclic_graph

    for kappa in [1, 2, 3]:
        graph, weights = random_cyclic_graph(d=10, num_cycles=kappa, density=0.5, seed=42)
        assert weights.shape == (10, 10)
        sccs = list(nx.strongly_connected_components(graph))
        nontrivial = [s for s in sccs if len(s) > 1]
        assert len(nontrivial) == kappa, (
            f"Expected {kappa} non-trivial SCCs, got {len(nontrivial)}"
        )


def test_random_cyclic_graph_oica_recovery():
    """OICA recovers the SCC partition from random_cyclic_graph with enough data."""
    from repare_cycle.graph import random_cyclic_graph, CyclicLinearSEM
    from repare_cycle.oica import oica_scc_partition

    graph, weights = random_cyclic_graph(d=10, num_cycles=2, density=0.5, seed=7)
    true_sccs = {frozenset(s) for s in nx.strongly_connected_components(graph)}

    sem = CyclicLinearSEM(weights, rng=np.random.default_rng(7), noise_dist="laplace")
    obs = sem.sample(10000)

    est_sccs = set(oica_scc_partition(obs, threshold=0.1, random_state=7))
    assert est_sccs == true_sccs


# ---------------------------------------------------------------------------
# PartitionDgModelOICA: end-to-end + partition-equality with raw OICA
# ---------------------------------------------------------------------------


def test_oica_repare_partition_matches_raw_oica():
    """`PartitionDgModelOICA.fit` must yield the same SCC partition as
    `oica_scc_partition` on the same data + threshold + random_state.

    Both consume support(Â) → Tarjan, so the final `dag.nodes` (from the
    RePaRe loop) must equal the raw OICA partition. If this drifts, the
    RePaRe loop is mutating the partition — which would break the
    apples-to-apples comparison between `oica_scc_repare` and
    `oica_no_repare` in the synth experiments.
    """
    from repare_cycle.graph import random_cyclic_graph, CyclicLinearSEM
    from repare_cycle.oica import oica_scc_partition
    from repare_cycle.repare_cycle import PartitionDgModelOICA

    graph, weights = random_cyclic_graph(d=10, num_cycles=2, density=0.5, seed=7)
    sem = CyclicLinearSEM(weights, rng=np.random.default_rng(7), noise_dist="laplace")
    obs = sem.sample(10_000)

    raw_sccs = set(oica_scc_partition(obs, threshold=0.1, random_state=7))

    model = PartitionDgModelOICA()
    model.fit(obs, beta=0.01, assume="hsic", threshold=0.1, random_state=7)
    repare_blocks = {frozenset(part) for part in model.dag.nodes}

    assert repare_blocks == raw_sccs, (
        f"RePaRe loop changed the partition.\n"
        f"  raw OICA: {sorted(map(sorted, raw_sccs))}\n"
        f"  RePaRe:   {sorted(map(sorted, repare_blocks))}"
    )
