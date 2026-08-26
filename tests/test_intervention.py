"""Tests for whole-SCC hard interventions and block-level effect estimation.

The d=4 closed-form case pins down the matrix convention (row convention,
``A[i, j]``: i → j) — a silent transpose error would flip which variables
respond to the intervention.
"""

import networkx as nx
import numpy as np
import pytest

from repare_cycle.graph import CyclicLinearSEM, random_cyclic_graph
from repare_cycle.intervention import (
    ancestral_subgraph_nodes,
    hard_intervention_means,
    observational_means,
    partitions_equal,
    propagate_block_intervention,
    simulate_hard_intervention,
)


def _chain_2cycle_weights():
    """d=4: x0 → (x1 ⇄ x2) → x3. SCCs: {0}, {1,2}, {3}."""
    A = np.zeros((4, 4))
    A[0, 1] = 0.8   # x0 → x1
    A[1, 2] = 0.5   # x1 → x2
    A[2, 1] = 0.4   # x2 → x1 (2-cycle)
    A[2, 3] = 0.7   # x2 → x3
    return A


def test_hard_intervention_closed_form_d4():
    """Clamp the 2-cycle {1, 2}; x3's mean must be μ_ε3 + 0.7·c2 and x0
    must be untouched. Wrong convention (transposed A) would instead move
    x0 and leave x3 at its noise mean."""
    A = _chain_2cycle_weights()
    mu_eps = np.array([1.0, -0.5, 2.0, 0.3])
    c = np.array([10.0, 20.0])  # clamp x1=10, x2=20

    mu_do = hard_intervention_means(A, mu_eps, target={1, 2}, c=c)

    # R = {0, 3}; A[R, R] = 0 so μ_R = μ_ε,R + c A[S, R].
    assert mu_do[0] == pytest.approx(1.0)          # x0 has no parents in S
    assert mu_do[3] == pytest.approx(0.3 + 0.7 * 20.0)  # x2 → x3 edge only
    assert mu_do[1] == 10.0 and mu_do[2] == 20.0

    # Observational means for contrast (Δμ is the compared quantity).
    mu_obs = observational_means(A, mu_eps)
    expected_obs = mu_eps @ np.linalg.inv(np.eye(4) - A)
    np.testing.assert_allclose(mu_obs, expected_obs)


def test_analytic_matches_monte_carlo():
    """Analytic post-intervention means vs sampling noise and solving the
    reduced R-system explicitly, on a random cyclic SEM (hard regime)."""
    graph, weights = random_cyclic_graph(
        d=8, num_cycles=2, density=0.5, seed=1, weight_range=(0.5, 0.95)
    )
    sem = CyclicLinearSEM(
        weights, rng=np.random.default_rng(1), noise_dist="laplace"
    )
    sccs = sorted(nx.strongly_connected_components(graph), key=len)
    target = sorted(sccs[-1])  # largest SCC
    c = np.full(len(target), 3.0)

    mu_analytic = hard_intervention_means(
        weights, sem.noise_means, target=target, c=c
    )
    X = simulate_hard_intervention(sem, target=target, c=c, n_samples=400_000)
    mu_mc = X.mean(axis=0)

    scale = np.abs(mu_analytic).max()
    np.testing.assert_allclose(mu_mc, mu_analytic, atol=0.01 * max(scale, 1.0))


@pytest.mark.parametrize("target_rho", [None, 1.5])  # stable / unstable
def test_block_regression_recovers_intervention_effect(target_rho):
    """Oracle-condensation block regression + propagation converges to the
    analytic Δμ at large n — including in the UNSTABLE regime, where the
    variable-level first-stable representative is provably wrong."""
    graph, weights = random_cyclic_graph(
        d=8, num_cycles=2, density=0.5, seed=2,
        weight_range=(0.5, 0.95), target_rho=target_rho,
    )
    sem = CyclicLinearSEM(
        weights, rng=np.random.default_rng(2), noise_dist="laplace",
        allow_unstable=target_rho is not None,
    )
    obs = sem.sample(200_000)

    partition = [frozenset(s) for s in nx.strongly_connected_components(graph)]
    var_to_block = {v: b for b, blk in enumerate(partition) for v in blk}
    edges = {
        (var_to_block[i], var_to_block[j])
        for i, j in zip(*np.nonzero(weights))
        if var_to_block[i] != var_to_block[j]
    }

    # Target: largest non-trivial block with at least one descendant block.
    g_blocks = nx.DiGraph()
    g_blocks.add_nodes_from(range(len(partition)))
    g_blocks.add_edges_from(edges)
    candidates = [
        b for b in range(len(partition))
        if len(partition[b]) > 1 and nx.descendants(g_blocks, b)
    ]
    assert candidates, "test SEM must have an intervenable block"
    target_block = max(candidates, key=lambda b: len(partition[b]))
    tvars = np.array(sorted(partition[target_block]))
    c = obs[:, tvars].mean(axis=0) + obs[:, tvars].std(axis=0)

    mu_do_true = hard_intervention_means(
        weights, sem.noise_means, target=tvars, c=c
    )
    delta_true = mu_do_true - observational_means(weights, sem.noise_means)

    mu_do_est = propagate_block_intervention(
        obs, partition, edges, target_block, c
    )
    delta_est = mu_do_est - obs.mean(axis=0)

    downstream_vars = np.array(sorted(
        v
        for b in nx.descendants(g_blocks, target_block)
        for v in partition[b]
    ))
    assert downstream_vars.size
    err = np.abs(delta_est[downstream_vars] - delta_true[downstream_vars])
    tol = 0.05 * max(np.abs(delta_true[downstream_vars]).max(), 1.0)
    assert err.max() < tol


def test_ancestral_subgraph_and_partition_equality():
    g = nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2), (2, 4), (5, 1)])
    # Paths 0→…→2: {0,1,2,3}; node 5 (ancestor of 2 but not descendant of 0)
    # and node 4 (descendant of 0 but not ancestor of 2) are excluded.
    assert ancestral_subgraph_nodes(g, 0, 2) == {0, 1, 2, 3}
    assert partitions_equal([{0, 1}, {2}], [frozenset({2}), frozenset({1, 0})])
    assert not partitions_equal([{0, 1}, {2}], [{0}, {1, 2}])
