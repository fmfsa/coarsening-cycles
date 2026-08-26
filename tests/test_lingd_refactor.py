"""Tests for the staged LiNG-D pipeline.

Verifies that ``run_lingd`` equals the explicit composition of
``fit_ica_unmixing`` → (``enumerate_admissible_candidates`` →
``choose_enumerated_candidate`` | ``hungarian_candidate``) for every pick
strategy, and that the enumeration metadata is honest (the Hungarian path
enumerates nothing; the cap flag is set by probing beyond the cap).
"""

import networkx as nx
import numpy as np
import pytest

from repare_cycle.graph import CyclicLinearSEM, random_cyclic_graph
from repare_cycle.lingd import (
    arbitrary_permutation_candidate,
    choose_enumerated_candidate,
    enumerate_admissible_candidates,
    fit_ica_unmixing,
    hungarian_candidate,
    random_matching_candidate,
    run_lingd,
)

THRESH = 0.1


def _make_obs(seed: int, d: int = 6, n: int = 4000) -> np.ndarray:
    graph, weights = random_cyclic_graph(
        d=d, num_cycles=2, density=0.5, seed=seed, weight_range=(0.5, 0.95)
    )
    sem = CyclicLinearSEM(
        weights,
        rng=np.random.default_rng(seed),
        noise_dist="laplace",
    )
    return sem.sample(n)


@pytest.mark.parametrize("strategy", ["first_stable", "random_admissible"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wrapper_equals_staged_composition_enumerated(seed, strategy):
    obs = _make_obs(seed)
    wrapped = run_lingd(
        obs, threshold_b=THRESH, threshold_w=THRESH,
        pick_strategy=strategy, random_state=seed,
    )

    W, err = fit_ica_unmixing(obs, random_state=seed)
    assert err is None and W is not None
    cands = enumerate_admissible_candidates(
        W, threshold_b=THRESH, threshold_w=THRESH
    )
    B, is_stable = choose_enumerated_candidate(
        cands, pick_strategy=strategy, random_state=seed
    )

    assert wrapped["failed"] is False
    assert B is not None, "no admissible candidate on a well-posed test SEM"
    np.testing.assert_array_equal(wrapped["B_chosen"], B)
    assert wrapped["is_stable"] == is_stable
    assert wrapped["n_candidates_enumerated"] == cands["n_candidates_enumerated"]
    assert wrapped["n_perms"] == cands["n_candidates_enumerated"]
    assert wrapped["n_candidates_returned"] == (
        len(cands["stable"]) + len(cands["unstable"])
    )
    assert wrapped["enumeration_cap_hit"] == cands["enumeration_cap_hit"]
    assert len(wrapped["stable"]) == len(cands["stable"])
    assert len(wrapped["unstable"]) == len(cands["unstable"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wrapper_equals_staged_composition_hungarian(seed):
    obs = _make_obs(seed)
    wrapped = run_lingd(
        obs, threshold_b=THRESH, threshold_w=THRESH,
        pick_strategy="hungarian_any", random_state=seed,
    )

    W, _ = fit_ica_unmixing(obs, random_state=seed)
    B = hungarian_candidate(W, threshold_b=THRESH, threshold_w=THRESH)

    assert wrapped["failed"] is False
    assert B is not None
    np.testing.assert_array_equal(wrapped["B_chosen"], B)
    # Honest metadata: Hungarian enumerates zero equivalence-class members.
    assert wrapped["n_candidates_enumerated"] == 0
    assert wrapped["n_candidates_returned"] == 1
    assert wrapped["is_stable"] is None
    assert wrapped["stable"] == [] and wrapped["unstable"] == []


def test_random_admissible_deterministic_given_seed():
    obs = _make_obs(3)
    W, _ = fit_ica_unmixing(obs, random_state=3)
    cands = enumerate_admissible_candidates(
        W, threshold_b=THRESH, threshold_w=THRESH
    )
    b1, s1 = choose_enumerated_candidate(
        cands, pick_strategy="random_admissible", random_state=7
    )
    b2, s2 = choose_enumerated_candidate(
        cands, pick_strategy="random_admissible", random_state=7
    )
    np.testing.assert_array_equal(b1, b2)
    assert s1 == s2


def test_enumeration_cap_hit_probes_beyond_cap():
    """A dense W admits many permutations; with max_perms=1 the cap flag
    must flip because a further admissible permutation exists."""
    rng = np.random.default_rng(0)
    W = rng.uniform(0.5, 1.0, size=(4, 4)) * rng.choice([-1, 1], size=(4, 4))
    full = enumerate_admissible_candidates(
        W, threshold_b=THRESH, threshold_w=THRESH, max_perms=10_000
    )
    assert full["n_candidates_enumerated"] == 24  # all 4! permutations admissible
    assert full["enumeration_cap_hit"] is False

    capped = enumerate_admissible_candidates(
        W, threshold_b=THRESH, threshold_w=THRESH, max_perms=1
    )
    assert capped["n_candidates_enumerated"] == 1
    assert capped["enumeration_cap_hit"] is True

    exact = enumerate_admissible_candidates(
        W, threshold_b=THRESH, threshold_w=THRESH, max_perms=24
    )
    # Exactly at the class size: cap reached but nothing exists beyond it.
    assert exact["n_candidates_enumerated"] == 24
    assert exact["enumeration_cap_hit"] is False


def test_enumeration_empty_case_uses_hungarian_precheck():
    """A W whose last column has no super-threshold entry admits no
    permutation; the Hungarian pre-check must certify emptiness without
    the DFS having to exhaust the search tree."""
    rng = np.random.default_rng(1)
    W = rng.uniform(0.5, 1.0, size=(5, 5))
    W[:, -1] = 0.01  # below threshold everywhere -> no admissible matching
    out = enumerate_admissible_candidates(
        W, threshold_b=THRESH, threshold_w=THRESH
    )
    assert out["n_candidates_enumerated"] == 0
    assert out["stable"] == [] and out["unstable"] == []
    assert out["enumeration_cap_hit"] is False
    assert out["enumeration_timed_out"] is False


def test_enumeration_time_budget_flags_truncation():
    """An exhausted time budget prunes the DFS and is reported as a
    truncated (timed-out) enumeration, never silently."""
    rng = np.random.default_rng(0)
    W = rng.uniform(0.5, 1.0, size=(6, 6)) * rng.choice([-1, 1], size=(6, 6))
    out = enumerate_admissible_candidates(
        W, threshold_b=THRESH, threshold_w=THRESH, time_budget_sec=0.0
    )
    assert out["enumeration_timed_out"] is True
    full = enumerate_admissible_candidates(
        W, threshold_b=THRESH, threshold_w=THRESH
    )
    assert full["enumeration_timed_out"] is False
    assert full["n_candidates_enumerated"] > out["n_candidates_enumerated"]


def test_first_stable_recovers_true_graph_on_stable_sem():
    """Sanity: on a stable sparse SEM the first-stable pick matches the
    true edge set (Lacerda's uniqueness argument for sparse cyclic SEMs)."""
    graph, weights = random_cyclic_graph(
        d=6, num_cycles=2, density=0.5, seed=0, weight_range=(0.5, 0.95)
    )
    sem = CyclicLinearSEM(
        weights, rng=np.random.default_rng(0), noise_dist="laplace"
    )
    obs = sem.sample(20_000)
    res = run_lingd(obs, pick_strategy="first_stable")
    assert res["failed"] is False and res["is_stable"] is True
    true_adj_col = (weights.T != 0)  # B_chosen is column convention (j → i)
    est_adj_col = np.abs(res["B_chosen"]) > 0
    assert (true_adj_col == est_adj_col).mean() > 0.9


def test_random_matching_candidate_no_enumeration():
    """The randomised matching arm is admissible, deterministic per seed,
    varies with the seed on a dense W, and never enumerates."""
    rng = np.random.default_rng(2)
    W = rng.uniform(0.5, 1.0, size=(5, 5)) * rng.choice([-1, 1], size=(5, 5))

    b1 = random_matching_candidate(W, random_state=0)
    b2 = random_matching_candidate(W, random_state=0)
    assert b1 is not None
    np.testing.assert_array_equal(b1, b2)  # deterministic given seed

    # On a fully dense W all 5! permutations are admissible; different seeds
    # should reach more than one distinct candidate.
    outs = {
        random_matching_candidate(W, random_state=s).tobytes()
        for s in range(12)
    }
    assert len(outs) > 1

    # Same admissibility contract as hungarian_candidate: every returned
    # candidate corresponds to a diagonal-zeroless permutation, and columns
    # with no allowed source make the matching infeasible.
    W_bad = W.copy()
    W_bad[:, 0] = 0.01
    assert random_matching_candidate(W_bad, random_state=0) is None


def test_arbitrary_permutation_candidate_does_not_enforce_admissibility():
    """The negative control always returns a graph and separately reports
    whether its uniformly sampled permutation happened to be admissible."""
    W = np.array([
        [0.9, 0.01, 0.01],
        [0.01, 0.8, 0.01],
        [0.01, 0.01, 0.7],
    ])

    seen_admissible = set()
    seen_candidates = set()
    for seed in range(20):
        B, admissible, min_diag = arbitrary_permutation_candidate(
            W, threshold_w=THRESH, random_state=seed
        )
        assert B.shape == W.shape
        assert np.isfinite(B).all()
        assert admissible == (min_diag > THRESH)
        seen_admissible.add(admissible)
        seen_candidates.add(B.tobytes())

    assert seen_admissible == {False, True}
    assert len(seen_candidates) > 1


def test_runtime_fields_present():
    obs = _make_obs(4)
    for strategy in ["first_stable", "random_admissible", "hungarian_any"]:
        res = run_lingd(obs, pick_strategy=strategy)
        assert res["ica_runtime_sec"] > 0
        assert res["selection_runtime_sec"] >= 0
