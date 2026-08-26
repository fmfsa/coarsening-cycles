"""Tests for the internal TVB bridge (repare_cycle.tvb_bridge).

Pure tests (no TVB needed):
  - repository ↔ TVB orientation round-trip
  - exact equilibrium solution and input reconstruction
  - SCC preservation across orientation conversion
  - descendant-reachability scores on perfect and deliberately wrong
    predictions
  - connectome threshold statistics on a hand-built graph

Optional TVB tests (skipped automatically when the `.[tvb]` extra is not
installed):
  - a three-node cycle converges to the exact equilibrium
  - a two-SCC system's soft-intervention response matches the exact solution
  - the default-connectome preflight returns valid statistics
"""

import networkx as nx
import numpy as np
import pytest

from repare_cycle.tvb_bridge import (
    cdag_direct_scores,
    condensation_from_adjacency,
    connectome_threshold_stats,
    descendant_reachability_scores,
    exact_equilibrium,
    from_tvb_orientation,
    predicted_descendant_variables,
    reconstruct_inputs,
    to_tvb_orientation,
)

# ---------------------------------------------------------------------------
# Orientation conversion
# ---------------------------------------------------------------------------


def test_orientation_round_trip():
    """repo → TVB → repo is the identity, and single entries transpose."""
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(6, 6))

    weights_tvb = to_tvb_orientation(weights)
    assert np.array_equal(from_tvb_orientation(weights_tvb), weights)
    # repo edge i→j lands at TVB [target, source] = [j, i]
    for i, j in [(0, 1), (2, 5), (4, 0)]:
        assert weights_tvb[j, i] == weights[i, j]


def test_scc_partition_invariant_under_transpose():
    """Transposing the weight matrix preserves the SCC partition exactly."""
    from repare_cycle.graph import random_cyclic_graph

    for seed in range(3):
        _, weights = random_cyclic_graph(
            d=12, num_cycles=3, density=0.3, seed=seed, target_rho=0.9
        )
        sccs = nx.strongly_connected_components
        parts = {frozenset(s) for s in sccs(nx.DiGraph(weights.astype(bool)))}
        parts_t = {frozenset(s) for s in sccs(nx.DiGraph(weights.T.astype(bool)))}
        assert parts == parts_t, (
            f"SCC partition changed under transpose (seed {seed}).\n"
            f"  repo: {parts}\n  tvb:  {parts_t}"
        )


# ---------------------------------------------------------------------------
# Exact equilibrium and input reconstruction
# ---------------------------------------------------------------------------


def test_exact_equilibrium_matches_sem_solution():
    """exact_equilibrium reproduces CyclicLinearSEM's (I-W)^{-1} sampling map."""
    from repare_cycle.graph import CyclicLinearSEM, random_cyclic_graph

    _, weights = random_cyclic_graph(d=8, num_cycles=2, density=0.4, seed=1,
                                     target_rho=0.9)
    sem = CyclicLinearSEM(weights, rng=np.random.default_rng(1))
    rng = np.random.default_rng(2)
    inputs = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=(5, 8))

    expected = inputs @ sem._solution_matrix
    got = exact_equilibrium(weights, inputs)
    assert np.allclose(got, expected, atol=1e-10)

    # (d,) input returns a (d,) equilibrium
    single = exact_equilibrium(weights, inputs[0])
    assert single.shape == (8,)
    assert np.allclose(single, expected[0], atol=1e-10)


def test_input_reconstruction_round_trip():
    """reconstruct_inputs is the exact inverse of exact_equilibrium."""
    from repare_cycle.graph import random_cyclic_graph

    _, weights = random_cyclic_graph(d=10, num_cycles=2, density=0.3, seed=3,
                                     target_rho=0.9)
    rng = np.random.default_rng(4)
    inputs = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=(20, 10))

    recovered = reconstruct_inputs(weights, exact_equilibrium(weights, inputs))
    assert np.allclose(recovered, inputs, atol=1e-10)


# ---------------------------------------------------------------------------
# Descendant-reachability scores
# ---------------------------------------------------------------------------

# Shared 5-node ground truth: two 2-cycles A={0,1}, B={2,3} and singleton {4},
# with condensation A → B → {4}.
_DESC_TRUE = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
])
_DESC_LABELS = np.array([0, 0, 1, 1, 2])


def test_descendant_scores_perfect_prediction():
    """A perfect prediction scores 1.0 across the board."""
    scores = descendant_reachability_scores(_DESC_TRUE, _DESC_TRUE, _DESC_LABELS)
    assert scores["desc_precision"] == 1.0
    assert scores["desc_recall"] == 1.0
    assert scores["desc_fscore"] == 1.0
    # Cross-SCC reachable pairs: A×B (4) + A×{4} (2) + B×{4} (2) = 8.
    assert scores["n_desc_true"] == scores["n_desc_pred"] == 8
    assert scores["n_desc_pairs"] == 16  # ordered cross-SCC pairs: 20 - 4 intra


def test_descendant_scores_penalize_wrong_prediction():
    """Deliberately wrong predictions lose precision and/or recall."""
    # (a) Empty prediction: no predicted pairs → precision defaults to 1.0
    # (conventions of evaluate_synth.py), recall and F1 collapse to 0.
    empty = np.zeros_like(_DESC_TRUE)
    scores = descendant_reachability_scores(_DESC_TRUE, empty, _DESC_LABELS)
    assert scores["desc_precision"] == 1.0
    assert scores["desc_recall"] == 0.0
    assert scores["desc_fscore"] == 0.0

    # (b) Merging A and B (extra back-edge 2→0) predicts mutual reachability
    # between two true SCCs; the false direction B→A costs precision while
    # recall stays perfect.
    merged = _DESC_TRUE.copy()
    merged[2, 0] = 1
    scores = descendant_reachability_scores(_DESC_TRUE, merged, _DESC_LABELS)
    assert scores["desc_recall"] == 1.0
    assert scores["desc_precision"] < 1.0

    # (c) Reversing the condensation edge A→B (predict 2→1 instead of 1→2)
    # flips the predicted order of A and B: both precision and recall drop.
    reversed_edge = _DESC_TRUE.copy()
    reversed_edge[1, 2] = 0
    reversed_edge[2, 1] = 1
    scores = descendant_reachability_scores(
        _DESC_TRUE, reversed_edge, _DESC_LABELS
    )
    assert scores["desc_precision"] < 1.0
    assert scores["desc_recall"] < 1.0


# ---------------------------------------------------------------------------
# Direct C-DAG comparison and intervention-descendant prediction
# ---------------------------------------------------------------------------


def test_cdag_direct_scores_perfect_and_collapsed():
    """Exact C-DAG match scores 1.0; a collapsed one-node estimate scores 0."""
    true_cdag = condensation_from_adjacency(_DESC_TRUE)
    # Condensation of _DESC_TRUE: {0,1} → {2,3} → {4}, two edges.
    assert set(true_cdag.nodes) == {
        frozenset({0, 1}), frozenset({2, 3}), frozenset({4})
    }
    assert true_cdag.number_of_edges() == 2

    perfect = cdag_direct_scores(true_cdag, true_cdag)
    assert perfect["cdag_fscore"] == 1.0
    assert perfect["partition_exact"] == 1
    assert perfect["n_cdag_edges_true"] == perfect["n_cdag_edges_est"] == 2

    # All variables merged into one estimated cluster: no C-DAG edges at
    # all, so recall and F1 are 0 — unlike the projected cluster-DAG F1,
    # which stays nonzero via the surviving variable edges.
    collapsed = nx.DiGraph()
    collapsed.add_node(frozenset(range(5)))
    scores = cdag_direct_scores(true_cdag, collapsed)
    assert scores["cdag_precision"] == 1.0  # empty-prediction convention
    assert scores["cdag_recall"] == 0.0
    assert scores["cdag_fscore"] == 0.0
    assert scores["partition_exact"] == 0


def test_predicted_descendant_variables():
    """C-DAG descendant prediction covers stimulated + downstream clusters."""
    true_cdag = condensation_from_adjacency(_DESC_TRUE)
    # Stimulating the source SCC {0,1} reaches everything downstream.
    assert predicted_descendant_variables(true_cdag, {0, 1}) == frozenset(range(5))
    # Stimulating the sink singleton {4} reaches only itself.
    assert predicted_descendant_variables(true_cdag, {4}) == frozenset({4})
    # A collapsed one-cluster estimate predicts every variable responds.
    collapsed = nx.DiGraph()
    collapsed.add_node(frozenset(range(5)))
    assert predicted_descendant_variables(collapsed, {4}) == frozenset(range(5))


# ---------------------------------------------------------------------------
# Connectome threshold statistics
# ---------------------------------------------------------------------------


def test_connectome_threshold_stats_pure():
    """Hand-built 6-node graph: exact stats at q=0, monotone retention."""
    weights = np.zeros((6, 6))
    weights[0, 1] = 1.0
    weights[1, 0] = 0.9   # 2-cycle {0,1}
    weights[1, 2] = 0.5
    weights[2, 3] = 0.4
    weights[3, 2] = 0.35  # 2-cycle {2,3}
    weights[3, 4] = 0.2
    weights[4, 5] = 0.1

    rows = connectome_threshold_stats(weights)
    assert len(rows) == 7

    full = rows[0]  # q=0 retains every nonzero edge
    assert full["retained_edges"] == 7
    assert full["retained_fraction"] == 1.0
    assert full["reciprocity"] == pytest.approx(4 / 7)
    assert full["num_sccs"] == 4
    assert full["num_nontrivial_sccs"] == 2
    assert full["largest_scc_fraction"] == pytest.approx(2 / 6)
    # Condensation: {0,1} → {2,3} → {4} → {5}
    assert full["n_condensation_edges"] == 3

    for row in rows:
        assert 0.0 <= row["retained_fraction"] <= 1.0
        assert 0.0 <= row["reciprocity"] <= 1.0
        assert 0.0 <= row["symmetry_error"]
        assert 1 <= row["num_sccs"] <= 6
    retained = [row["retained_edges"] for row in rows]
    assert retained == sorted(retained, reverse=True)


# ---------------------------------------------------------------------------
# Reduced Wong-Wang math (pure)
# ---------------------------------------------------------------------------

# Small 4-node test network: 2-cycle {0,1} -> 2 -> 3.
_RWW_W = np.zeros((4, 4))
_RWW_W[0, 1], _RWW_W[1, 0] = 0.5, 0.4
_RWW_W[1, 2] = 0.6
_RWW_W[2, 3] = 0.5
_RWW_G = 0.5
_RWW_IO = np.array([0.32, 0.33, 0.34, 0.31])


def test_rww_jacobian_matches_finite_differences():
    """Analytic Reduced Wong-Wang Jacobian agrees with numeric derivatives."""
    from repare_cycle.tvb_bridge import rww_dfun, rww_jacobian

    rng = np.random.default_rng(11)
    state = rng.uniform(0.1, 0.6, size=4)
    jac = rww_jacobian(state, _RWW_IO, _RWW_W, _RWW_G)
    eps = 1e-7
    for j in range(4):
        bumped = state.copy()
        bumped[j] += eps
        fd = (rww_dfun(bumped, _RWW_IO, _RWW_W, _RWW_G)
              - rww_dfun(state, _RWW_IO, _RWW_W, _RWW_G)) / eps
        assert np.allclose(jac[:, j], fd, atol=1e-5), f"column {j} mismatch"


def test_rww_fixed_point_is_stable_with_matching_support():
    """Newton finds a stable interior fixed point whose cross-region
    Jacobian support equals the graph support."""
    from repare_cycle.tvb_bridge import rww_fixed_point

    fp = rww_fixed_point(_RWW_W, _RWW_IO, _RWW_G)
    assert fp.converged and fp.residual < 1e-12
    assert np.all((fp.state > 0.0) & (fp.state < 1.0))
    assert fp.max_real_eig < 0.0  # locally stable

    # Off-diagonal Jacobian support == W_tvb support (i<-j iff W[j,i] != 0).
    offdiag = fp.jacobian.copy()
    np.fill_diagonal(offdiag, 0.0)
    assert np.array_equal(offdiag != 0.0, _RWW_W.T != 0.0)


def test_rww_equilibria_unique_stable_on_small_net():
    """The enumerator certifies the small test regime as single-stable and
    rww_stable_fixed_point returns that equilibrium."""
    from repare_cycle.tvb_bridge import rww_equilibria, rww_stable_fixed_point

    eq = rww_equilibria(_RWW_W, _RWW_IO, _RWW_G)
    assert eq.n_equilibria == 1
    assert eq.unique_stable
    fp = rww_stable_fixed_point(_RWW_W, _RWW_IO, _RWW_G)
    assert np.allclose(fp.state, eq.states[0], atol=1e-9)
    assert fp.margin > 0.0


def test_rww_linearization_matches_fixed_point_sensitivity():
    """The linearized mixing M predicts how the fixed point moves under a
    small input bump, and b_col's support equals the graph's."""
    from repare_cycle.tvb_bridge import (
        rww_fixed_point, rww_linearization, rww_stable_fixed_point,
    )

    fp = rww_stable_fixed_point(_RWW_W, _RWW_IO, _RWW_G)
    lin = rww_linearization(fp.state, _RWW_IO, _RWW_W, _RWW_G)
    assert np.array_equal(lin.b_col != 0.0, _RWW_W.T != 0.0)

    eps = 1e-6
    bumped_io = _RWW_IO.copy()
    bumped_io[0] += eps
    bumped = rww_fixed_point(_RWW_W, bumped_io, _RWW_G, start=fp.state)
    assert bumped.converged
    sensitivity_fd = (bumped.state - fp.state) / eps
    assert np.allclose(sensitivity_fd, lin.mixing[:, 0], atol=1e-4)


def test_rww_dfun_broadcasts_over_episodes():
    """(n, d) episode rows evaluate identically to per-row 1-D calls."""
    from repare_cycle.tvb_bridge import rww_dfun

    rng = np.random.default_rng(12)
    states = rng.uniform(0.05, 0.8, size=(6, 4))
    inputs = _RWW_IO + rng.laplace(0.0, 0.02, size=(6, 4))
    batched = rww_dfun(states, inputs, _RWW_W, _RWW_G)
    for k in range(6):
        single = rww_dfun(states[k], inputs[k], _RWW_W, _RWW_G)
        assert np.allclose(batched[k], single, atol=1e-14)


# ---------------------------------------------------------------------------
# Optional TVB integration tests (skip when the extra is not installed)
# ---------------------------------------------------------------------------


def test_tvb_rww_dfun_matches_bridge():
    """TVB's ReducedWongWang dfun (numba path, spatialized I_o) matches the
    bridge's pure implementation."""
    pytest.importorskip("tvb.simulator.simulator")
    from tvb.simulator.models.wong_wang import ReducedWongWang

    from repare_cycle.tvb_bridge import RWW_DEFAULTS, rww_dfun

    rng = np.random.default_rng(13)
    state = rng.uniform(0.05, 0.8, size=4)
    model = ReducedWongWang(I_o=_RWW_IO.copy())
    model.configure()

    # TVB evaluates dfun(state, coupling) with coupling precomputed; feed
    # it the same linear coupling the bridge assumes.
    coupling_in = _RWW_G * (state @ _RWW_W)
    tvb_deriv = model.dfun(
        state[None, :, None], coupling_in[None, :, None]
    )[0, :, 0]
    ours = rww_dfun(state, _RWW_IO, _RWW_W, _RWW_G, params=RWW_DEFAULTS)
    assert np.allclose(tvb_deriv, ours, atol=1e-12)


def test_tvb_rww_ensemble_settles_to_newton_fixed_point():
    """Block-diagonal TVB batching settles every episode onto the Newton
    fixed point of its own input row."""
    pytest.importorskip("tvb.simulator.simulator")
    from repare_cycle.tvb_bridge import rww_fixed_point, simulate_rww_ensemble

    rng = np.random.default_rng(14)
    input_rows = _RWW_IO + rng.laplace(0.0, 0.02, size=(3, 4))
    # batch_size=2 exercises both a full batch and a remainder batch.
    result = simulate_rww_ensemble(
        _RWW_W, input_rows, g=_RWW_G, tol=1e-9, batch_size=2
    )
    assert bool(np.all(result.converged))
    assert float(np.max(result.residuals)) <= 1e-9
    for k in range(3):
        fp = rww_fixed_point(_RWW_W, input_rows[k], _RWW_G)
        assert fp.converged
        rel = (np.linalg.norm(result.states[k] - fp.state)
               / np.linalg.norm(fp.state))
        assert rel <= 1e-6, f"episode {k}: rel err {rel:.2e}"


def test_tvb_three_node_cycle_reaches_exact_equilibrium():
    """TVB integration of a 3-cycle settles onto the exact LiNG equilibrium."""
    pytest.importorskip("tvb.simulator.simulator")
    from repare_cycle.tvb_bridge import simulate_to_equilibrium

    weights = np.zeros((3, 3))
    weights[0, 1] = weights[1, 2] = weights[2, 0] = 0.5
    inputs = np.array([1.0, 2.0, -1.0])

    result = simulate_to_equilibrium(weights, inputs)
    assert result.converged
    assert result.terminal_residual <= 1e-7

    x_exact = exact_equilibrium(weights, inputs)
    rel_err = np.linalg.norm(result.x - x_exact) / np.linalg.norm(x_exact)
    assert rel_err <= 1e-5, f"relative equilibrium error {rel_err:.2e} > 1e-5"


def test_tvb_soft_intervention_matches_exact_response():
    """A cluster-level soft shift propagates through TVB as predicted."""
    pytest.importorskip("tvb.simulator.simulator")
    from repare_cycle.tvb_bridge import simulate_to_equilibrium

    # Two 2-SCCs {0,1} → {2,3} with one cross edge 1→2.
    weights = np.zeros((4, 4))
    weights[0, 1], weights[1, 0] = 0.6, 0.5
    weights[2, 3], weights[3, 2] = 0.6, 0.5
    weights[1, 2] = 0.4

    # Soft shift 1/sqrt(2) on the upstream SCC; baseline is zero, so the
    # intervened equilibrium IS the steady-state response.
    delta_e = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    expected = exact_equilibrium(weights, delta_e)
    assert np.all(np.abs(expected[2:]) > 0)  # response reaches descendants

    result = simulate_to_equilibrium(weights, delta_e)
    assert result.converged
    assert result.terminal_residual <= 1e-7
    rel_err = np.linalg.norm(result.x - expected) / np.linalg.norm(expected)
    assert rel_err <= 1e-5, f"relative response error {rel_err:.2e} > 1e-5"


def test_tvb_default_connectome_preflight_stats_valid():
    """The default 76-region connectome yields well-formed preflight stats."""
    pytest.importorskip("tvb.simulator.simulator")
    from repare_cycle.tvb_bridge import from_tvb_orientation as from_tvb
    from repare_cycle.tvb_bridge import load_default_connectivity

    conn = load_default_connectivity()
    assert conn.number_of_regions == 76

    rows = connectome_threshold_stats(from_tvb(conn.weights))
    assert len(rows) == 7
    for row in rows:
        assert 0.0 <= row["retained_fraction"] <= 1.0
        assert 0.0 <= row["reciprocity"] <= 1.0
        assert 1 <= row["num_sccs"] <= 76
        assert 0.0 < row["largest_scc_fraction"] <= 1.0
        assert row["n_condensation_edges"] >= 0
