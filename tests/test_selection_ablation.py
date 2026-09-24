import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "selection_ablation", Path(__file__).resolve().parents[1] / "scripts/run_selection_ablation.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_exact_recovery_requires_edges_as_well_as_partition():
    truth = np.array([[0., 1., 1.], [1., 0., 0.], [0., 0., 0.]])
    estimate = truth.copy()
    assert benchmark.score_candidate(estimate.T, truth)["exact"] == 1
    estimate[0, 2] = 0
    metrics = benchmark.score_candidate(estimate.T, truth)
    assert metrics["partition_exact"] == 1
    assert metrics["ari"] == 1
    assert metrics["exact"] == 0
    assert benchmark.score_candidate(None, truth)["exact"] is None


def test_random_draws_are_averaged_within_seed(tmp_path):
    rows = []
    for seed, times in [(0, [1., 3.]), (1, [6., 6.])]:
        for draw, elapsed in enumerate(times):
            rows.append(dict(regime="hard", n=100, seed=seed, method="enum_uniform_random", draw=draw,
                             ari=float(seed), exact=seed, partition_exact=seed, ica_runtime_sec=2.,
                             selection_runtime_sec=elapsed, total_runtime_sec=2. + elapsed,
                             enumeration_cap_hit=False, enumeration_timed_out=False, status="ok"))
    result = benchmark.summarize(rows, tmp_path).iloc[0]
    assert result.seeds == 2
    assert result.exact_pct == 50
    assert result.selection_sec == 4
    assert result.dominance_count == 1
    cells = pd.read_csv(tmp_path / "cells.csv")
    np.testing.assert_allclose(cells.total_runtime_sec, cells.ica_runtime_sec + cells.selection_runtime_sec)


def test_total_enumeration_budget_is_shared_across_threshold_rounds(tmp_path, monkeypatch):
    clock = [0.]
    budgets = []
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(benchmark, "fit_ica_unmixing", lambda *a, **k: (np.eye(4), None))

    def enumerate_empty(W, **kwargs):
        budgets.append(kwargs["time_budget_sec"])
        clock[0] += min(40., kwargs["time_budget_sec"])
        return dict(stable=[], unstable=[], n_candidates_enumerated=0,
                    enumeration_cap_hit=False, enumeration_timed_out=clock[0] >= 60)

    monkeypatch.setattr(benchmark, "enumerate_admissible_candidates", enumerate_empty)
    cfg = dict(d=4, num_cycles=2, density=.5, noise_dist="laplace", threshold=.1,
               ica_max_iter=10, ica_tolerance=1e-6, ica_random_state=0, random_draws=1,
               max_perms=None, enumeration_budget_sec=60., enumeration_budget_scope="total")
    result = benchmark.run_cell(cfg, "hard", 10, 0, tmp_path)
    assert budgets == [60., 20.]
    first = next(r for r in result["rows"] if r["method"] == "enum_first_stable")
    assert first["enumeration_timed_out"]
    assert first["status"] == "no_estimate"
