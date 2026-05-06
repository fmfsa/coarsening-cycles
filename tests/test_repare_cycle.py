"""Tests for RePaRe-Cycle.

Covers:
- Tarjan's SCC algorithm correctness
- CyclicLinearSEM sampling and intervention propagation
- random_directed_graph stability guarantee
- random_cyclic_graph / random_multi_scc_graph structure
- LiNG-D wrapper: SCC and condensation recovery
"""

import networkx as nx
import numpy as np
import pytest

from repare_cycle.graph import CyclicLinearSEM, random_directed_graph, tarjan_scc


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
    cov = np.cov(data.T)
    assert abs(cov[0, 1] - w) < 0.05


def test_cyclic_sem_stability_error():
    """Spectral radius >= 1 raises ValueError (requires a cyclic matrix)."""
    weights = np.array([[0.0, 1.1], [1.1, 0.0]])
    with pytest.raises(ValueError, match="Spectral radius"):
        CyclicLinearSEM(weights)


def test_cyclic_sem_soft_intervention_changes_distribution():
    """Soft intervention on node 0 shifts mean of X_0 and propagates to X_1."""
    weights = np.array([[0.0, 0.5], [0.0, 0.0]])
    model = CyclicLinearSEM(weights, means=(0.0, 0.0), variances=(1.0, 1.0), rng=np.random.default_rng(1))
    obs = model.sample(5000)
    ivn = model.sample(5000, shift_interventions={0: (5.0, 0.1)})
    assert ivn[:, 0].mean() > obs[:, 0].mean() + 3
    assert ivn[:, 1].mean() > obs[:, 1].mean() + 1


def test_cyclic_sem_2cycle_intervention_propagates():
    """In a 2-cycle 0⇔1, intervening on 0 shifts both 0 and 1."""
    weights = np.array([[0.0, 0.4], [0.4, 0.0]])
    model = CyclicLinearSEM(weights, means=(0.0, 0.0), variances=(1.0, 1.0), rng=np.random.default_rng(2))
    obs = model.sample(10_000)
    ivn = model.sample(10_000, shift_interventions={0: (10.0, 0.1)})
    assert ivn[:, 0].mean() > obs[:, 0].mean() + 3
    assert ivn[:, 1].mean() > obs[:, 1].mean() + 1


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
    assert any(len(scc) > 1 for scc in sccs)


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
    assert n == 3 * 3 + 2

    sccs = list(nx.strongly_connected_components(graph))
    nontrivial = [s for s in sccs if len(s) > 1]
    singletons = [s for s in sccs if len(s) == 1]
    assert len(nontrivial) == 3
    assert len(singletons) == 2
    for scc in nontrivial:
        assert len(scc) == 3


# ---------------------------------------------------------------------------
# random_cyclic_graph (unified generator for main experiment)
# ---------------------------------------------------------------------------


def test_random_cyclic_graph_dag():
    """num_cycles=0 produces a DAG (no non-trivial SCCs)."""
    from repare_cycle.graph import random_cyclic_graph

    graph, weights = random_cyclic_graph(d=10, num_cycles=0, density=0.5, seed=0)
    assert weights.shape == (10, 10)
    sccs = list(nx.strongly_connected_components(graph))
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


# ---------------------------------------------------------------------------
# LiNG-D: SCC recovery on a known-stable cyclic SEM
# ---------------------------------------------------------------------------


def test_lingd_recovers_scc_partition():
    """LiNG-D + Tarjan recovers the SCC partition from cyclic data."""
    from repare_cycle.graph import random_cyclic_graph
    from repare_cycle.lingd import run_lingd

    graph, weights = random_cyclic_graph(d=10, num_cycles=2, density=0.5, seed=7)
    true_sccs = {frozenset(s) for s in nx.strongly_connected_components(graph)}

    sem = CyclicLinearSEM(weights, rng=np.random.default_rng(7), noise_dist="laplace")
    obs = sem.sample(10000)

    result = run_lingd(obs, threshold_b=0.1, threshold_w=0.1)
    B = result["B_chosen"]

    g = nx.DiGraph()
    g.add_nodes_from(range(B.shape[0]))
    rows, cols = np.where(np.abs(B) > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i != j:
            g.add_edge(int(j), int(i))
    est_sccs = {frozenset(s) for s in nx.strongly_connected_components(g)}

    assert est_sccs == true_sccs


# ---------------------------------------------------------------------------
# LiNG-D: Lacerda Example 1 — SCC + condensation recovery
# ---------------------------------------------------------------------------


def test_lingd_recovers_lacerda_example_1():
    """LiNG-D + Tarjan recovers the SCC partition and condensation of Lacerda Example 1.

    At n=5000, seed=0 the algorithm reliably picks a stable representative
    whose SCC partition matches the ground truth {x1}, {x2,x3,x4}, {x5} and
    whose condensation edges ({x1}→{x2,x3,x4}, {x2,x3,x4}→{x5}) are exact.
    """
    from repare_cycle.examples import get
    from repare_cycle.lingd import run_lingd

    ex = get("lacerda_example_1")
    sem = CyclicLinearSEM(ex.B.T, rng=np.random.default_rng(0), noise_dist="laplace")
    obs = sem.sample(5_000)

    result = run_lingd(obs, threshold_b=0.1, threshold_w=0.1)
    assert not result["failed"], "FastICA failed on Example 1 data"

    B = result["B_chosen"]

    g_hat = nx.DiGraph()
    g_hat.add_nodes_from(range(ex.d))
    for i, j in zip(*np.nonzero(B)):
        if i != j:
            g_hat.add_edge(int(j), int(i))

    # SCC partition
    est_sccs = {frozenset(s) for s in nx.strongly_connected_components(g_hat)}
    true_sccs = {frozenset(s) for s in ex.true_sccs}
    assert est_sccs == true_sccs, (
        f"SCC mismatch.\n  expected: {true_sccs}\n  got:      {est_sccs}"
    )

    # Condensation edges
    cond_hat = nx.condensation(g_hat)
    members_hat = {
        k: frozenset(int(v) for v in cond_hat.nodes[k]["members"])
        for k in cond_hat.nodes
    }
    hat_cond_e = {(members_hat[u], members_hat[v]) for u, v in cond_hat.edges}
    true_cond_e = set(ex.true_condensation.edges)
    assert hat_cond_e == true_cond_e, (
        f"Condensation edge mismatch.\n  expected: {true_cond_e}\n  got: {hat_cond_e}"
    )


# ---------------------------------------------------------------------------
# LiNG-D: hungarian_any pick_strategy yields the same condensation
# ---------------------------------------------------------------------------


def _condensation_signature(B: np.ndarray) -> tuple[frozenset, frozenset]:
    """Return (SCC partition, condensation edge set) for thresholded B."""
    g = nx.DiGraph()
    g.add_nodes_from(range(B.shape[0]))
    for i, j in zip(*np.nonzero(B)):
        if i != j:
            g.add_edge(int(j), int(i))
    sccs = frozenset(frozenset(s) for s in nx.strongly_connected_components(g))
    cond = nx.condensation(g)
    members = {
        k: frozenset(int(v) for v in cond.nodes[k]["members"]) for k in cond.nodes
    }
    edges = frozenset((members[u], members[v]) for u, v in cond.edges)
    return sccs, edges


def test_hungarian_any_matches_first_stable_condensation():
    """``pick_strategy='hungarian_any'`` recovers the same condensation as the
    default ``first_stable`` pick on the Lacerda Example 1.

    By Theorem 1 of the paper, every admissible representative of the
    distributional equivalence class shares the same condensation; this test
    verifies that the two pick strategies are interchangeable for the
    condensation-recovery target.
    """
    from repare_cycle.examples import get
    from repare_cycle.lingd import run_lingd

    ex = get("lacerda_example_1")
    sem = CyclicLinearSEM(ex.B.T, rng=np.random.default_rng(0), noise_dist="laplace")
    obs = sem.sample(5_000)

    res_fs = run_lingd(obs, threshold_b=0.1, threshold_w=0.1, pick_strategy="first_stable")
    res_hu = run_lingd(obs, threshold_b=0.1, threshold_w=0.1, pick_strategy="hungarian_any")

    assert not res_fs["failed"] and not res_hu["failed"]
    assert res_hu["n_perms"] == 1, "Hungarian path should evaluate exactly one permutation"

    sccs_fs, edges_fs = _condensation_signature(res_fs["B_chosen"])
    sccs_hu, edges_hu = _condensation_signature(res_hu["B_chosen"])

    assert sccs_fs == sccs_hu, (
        f"SCC partitions differ.\n  first_stable: {sccs_fs}\n  hungarian:    {sccs_hu}"
    )
    assert edges_fs == edges_hu, (
        f"Condensation edges differ.\n  first_stable: {edges_fs}\n  hungarian:    {edges_hu}"
    )
