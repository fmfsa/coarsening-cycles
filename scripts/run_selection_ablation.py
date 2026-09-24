"""Paired selection ablation with shared ICA, explicit stage times and resumable cells.

Run via run_ablation_d20.sh to fix numerical thread counts before imports.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA  # preload before timing the ICA stage
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment  # preload selection dependency

from repare_cycle.graph import CyclicLinearSEM, random_cyclic_graph
from repare_cycle.lingd import (
    arbitrary_permutation_candidate, choose_enumerated_candidate,
    enumerate_admissible_candidates, fit_ica_unmixing, hungarian_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["enum_first_stable", "enum_uniform_random", "ours", "arbitrary_control"]


def graph_signature(adj):
    graph = nx.from_numpy_array(np.asarray(adj, dtype=bool), create_using=nx.DiGraph)
    parts = {frozenset(c) for c in nx.strongly_connected_components(graph)}
    membership = {v: part for part in parts for v in part}
    edges = {(membership[u], membership[v]) for u, v in graph.edges if membership[u] != membership[v]}
    labels = np.zeros(len(adj), dtype=int)
    for i, part in enumerate(sorted(parts, key=lambda p: min(p))):
        labels[list(part)] = i
    return parts, edges, labels


def score_candidate(B, weights):
    if B is None:
        return dict(ari=None, exact=None, partition_exact=None)
    # Generator weights use i->j; the estimated B uses row-is-effect convention.
    truth = graph_signature(weights != 0)
    estimate = graph_signature(B.T != 0)
    return dict(ari=float(adjusted_rand_score(truth[2], estimate[2])),
                exact=int(truth[:2] == estimate[:2]), partition_exact=int(truth[0] == estimate[0]))


def run_cell(cfg, regime, n, seed, directory):
    _, weights = random_cyclic_graph(
        d=cfg["d"], num_cycles=cfg["num_cycles"], density=cfg["density"],
        seed=seed, weight_range=(0.5, 0.95),
        target_rho=1.5 if regime == "unstable" else None,
        intra_scc_density=cfg["density"],
    )
    sem = CyclicLinearSEM(weights, means=(-2., 2.), variances=(.5, 2.),
                          rng=np.random.default_rng(seed), noise_dist=cfg["noise_dist"],
                          allow_unstable=regime == "unstable")
    obs = sem.sample(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        W, error = fit_ica_unmixing(obs, ica_max_iter=cfg["ica_max_iter"],
                                  ica_tolerance=cfg["ica_tolerance"], random_state=cfg["ica_random_state"])
        ica_sec = time.perf_counter() - start
    np.savez_compressed(directory / "estimate.npz", weights=weights,
                        W=W if W is not None else np.empty((0, 0)))
    common = dict(regime=regime, n=n, seed=seed, ica_runtime_sec=ica_sec,
                  W_hash=hashlib.sha256(W.tobytes()).hexdigest() if W is not None else None)
    rows = []

    def record(method, draw, B, seconds, **extra):
        rows.append(dict(common, method=method, draw=draw,
                         status="ok" if B is not None else "no_estimate",
                         selection_runtime_sec=seconds, total_runtime_sec=ica_sec + seconds,
                         **score_candidate(B, weights), **extra))

    if W is None:
        for method in METHODS:
            for draw in range(cfg["random_draws"] if method in (METHODS[1], METHODS[3]) else 1):
                record(method, draw, None, 0., enumeration_cap_hit=False, enumeration_timed_out=False)
    else:
        # Rotate branch order across seeds to distribute cache/order effects.
        def ours():
            start = time.perf_counter()
            threshold = cfg["threshold"]
            while True:
                B = hungarian_candidate(W, threshold_b=cfg["threshold"], threshold_w=threshold)
                if B is not None or threshold <= .01:
                    break
                threshold /= 2
            elapsed = time.perf_counter() - start
            record("ours", 0, B, elapsed, threshold_w_used=threshold,
                   enumeration_cap_hit=False, enumeration_timed_out=False)

        def enumeration():
            start = time.perf_counter()
            total_deadline = (time.monotonic() + cfg["enumeration_budget_sec"]
                              if cfg.get("enumeration_budget_scope", "per_round") == "total" else None)
            threshold = cfg["threshold"]
            rounds = []
            while True:
                candidates = enumerate_admissible_candidates(
                    W, threshold_b=cfg["threshold"], threshold_w=threshold,
                    max_perms=cfg["max_perms"],
                    time_budget_sec=max(0., total_deadline - time.monotonic())
                    if total_deadline is not None else cfg["enumeration_budget_sec"])
                rounds.append(dict(threshold=threshold, **{k: v for k, v in candidates.items()
                                                          if k not in ("stable", "unstable")}))
                if (candidates["stable"] or candidates["unstable"] or threshold <= .01
                        or (total_deadline is not None and time.monotonic() >= total_deadline)):
                    break
                threshold /= 2
            enumeration_sec = time.perf_counter() - start
            metadata = dict(enumeration_runtime_sec=enumeration_sec, enumeration_rounds=rounds,
                            enumeration_cap_hit=any(r["enumeration_cap_hit"] for r in rounds),
                            enumeration_timed_out=(any(r["enumeration_timed_out"] for r in rounds)
                                or (total_deadline is not None and time.monotonic() >= total_deadline)),
                            n_candidates_enumerated=candidates["n_candidates_enumerated"],
                            n_stable_candidates=len(candidates["stable"]), threshold_w_used=threshold)
            for method, strategy, draws in [
                ("enum_first_stable", "first_stable", 1),
                ("enum_uniform_random", "random_admissible", cfg["random_draws"]),
            ]:
                for draw in range(draws):
                    start = time.perf_counter()
                    B, stable = choose_enumerated_candidate(candidates, pick_strategy=strategy,
                                                            random_state=seed * 100 + draw)
                    elapsed = enumeration_sec + time.perf_counter() - start
                    record(method, draw, B, elapsed, chosen_is_stable=stable, **metadata)

        def arbitrary():
            for draw in range(cfg["random_draws"]):
                start = time.perf_counter()
                B, admissible, min_diagonal = arbitrary_permutation_candidate(
                    W, threshold_b=cfg["threshold"], threshold_w=cfg["threshold"],
                    random_state=seed * 100 + draw)
                elapsed = time.perf_counter() - start
                record("arbitrary_control", draw, B, elapsed, permutation_admissible=admissible,
                       min_abs_selected_diagonal=min_diagonal,
                       enumeration_cap_hit=False, enumeration_timed_out=False)

        tasks = [ours, enumeration, arbitrary]
        shift = seed % len(tasks)
        for task in tasks[shift:] + tasks[:shift]:
            task()
    return dict(rows=rows, ica_error=error,
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                data_hash=hashlib.sha256(obs.tobytes()).hexdigest())


def summarize(rows, out):
    df = pd.DataFrame(rows)
    df.to_csv(out / "draws.csv", index=False)
    # The independent unit is the dataset/seed; five random draws are NOT five seeds.
    cells = df.groupby(["regime", "n", "seed", "method"], as_index=False).agg(
        ari=("ari", "mean"), exact=("exact", "mean"), partition_exact=("partition_exact", "mean"),
        ica_runtime_sec=("ica_runtime_sec", "first"),
        selection_runtime_sec=("selection_runtime_sec", "mean"),
        total_runtime_sec=("total_runtime_sec", "mean"),
        cap_hit=("enumeration_cap_hit", "max"), timed_out=("enumeration_timed_out", "max"),
        completed=("status", lambda s: bool((s == "ok").all())),
    )
    cells["selection_share"] = cells.selection_runtime_sec / cells.total_runtime_sec
    cells["selection_over_ica"] = cells.selection_runtime_sec / cells.ica_runtime_sec
    cells["selection_dominates"] = cells.selection_runtime_sec > cells.ica_runtime_sec
    cells.to_csv(out / "cells.csv", index=False)
    summary = cells.groupby(["regime", "n", "method"], as_index=False).agg(
        ari=("ari", "mean"), exact_pct=("exact", lambda s: 100 * s.mean()),
        partition_exact_pct=("partition_exact", lambda s: 100 * s.mean()),
        ica_sec=("ica_runtime_sec", "median"), selection_sec=("selection_runtime_sec", "median"),
        total_sec=("total_runtime_sec", "median"), selection_share=("selection_share", "median"),
        selection_over_ica=("selection_over_ica", "median"),
        dominance_count=("selection_dominates", "sum"), seeds=("seed", "size"),
        completed=("completed", "sum"), cap_count=("cap_hit", "sum"), timeout_count=("timed_out", "sum"),
    )
    summary.to_csv(out / "summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/ablation_d20.json")
    parser.add_argument("--output", type=Path, default=ROOT / "src/expt/workflow/results/ablation_d20")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["d"] < 2 * cfg["num_cycles"] or min(cfg["sample_sizes"]) < 2:
        raise ValueError("Invalid graph/sample dimensions")
    if min(cfg["seeds"], cfg["random_draws"], cfg["enumeration_budget_sec"]) <= 0:
        raise ValueError("Counts and enumeration budget must be positive")
    if cfg["max_perms"] is not None and cfg["max_perms"] <= 0:
        raise ValueError("max_perms must be positive or null (no count cap)")
    if cfg.get("enumeration_budget_scope", "per_round") not in ("per_round", "total"):
        raise ValueError("enumeration_budget_scope must be per_round or total")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "src/repare_cycle/lingd.py", ROOT / "src/repare_cycle/graph.py"]
    protocol = dict(config=cfg, source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    platform=platform.platform(), machine=platform.node(), python=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "networkx", "pandas"]},
                    threads={k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]})
    manifest = out / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text())["protocol"] != protocol:
        raise ValueError("Protocol/source/environment changed. Use a new --output directory; do not mix timings.")
    if not manifest.exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        manifest.write_text(json.dumps(dict(protocol=protocol, git_commit=commit,
                                            timing="preloaded imports; one shared ICA; selection excludes scoring and I/O"), indent=2))
    rows = []
    for regime in cfg["regimes"]:
        for n in cfg["sample_sizes"]:
            for seed in range(cfg["seeds"]):
                directory = out / f"regime={regime}" / f"n={n}" / f"seed={seed}"
                path = directory / "result.json"
                if not path.exists():
                    if args.summarize_only:
                        raise FileNotFoundError(f"Incomplete grid: {path}")
                    directory.mkdir(parents=True, exist_ok=True)
                    result = run_cell(cfg, regime, n, seed, directory)
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
                    temporary.replace(path)
                    first = next(r for r in result["rows"] if r["method"] == "enum_first_stable")
                    print(f"{regime} n={n} seed={seed}: ICA={first['ica_runtime_sec']:.3f}s, "
                          f"enum+selection={first['selection_runtime_sec']:.3f}s", flush=True)
                rows.extend(json.loads(path.read_text())["rows"])
    summary = summarize(rows, out)
    print(summary.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
