"""One isolated, timed fit. Imports/data loading are outside the fit budget.

Called by Snakemake; each result (including a timeout) is checkpointed as JSON.
Threads are set before numerical imports in both parent and spawned child.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import time
import traceback

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[4] / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")


def worker(conn, data_path, method, seed):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from repare_cycle.benchmark import group_lingam_adapter, groups_from_adjacency, condensation_metrics
    try:
        if method == "group_lingam":
            from lingam import GroupLiNGAM
        else:
            from repare_cycle.lingd import run_lingd
        with np.load(data_path) as data:
            obs, weights = data["obs"], data["weights"]
        np.random.seed(seed)
        conn.send({"ready": True})
        conn.recv()
        with threadpool_limits(limits=1):
            start, cpu_start = time.perf_counter(), time.process_time()
            try:
                diagnostics = {}
                if method == "group_lingam":
                    model = GroupLiNGAM(alpha=0.01).fit(obs)
                    raw_groups, b = model.causal_order_, model.adjacency_matrix_
                else:
                    result = run_lingd(obs, threshold_b=0.1, threshold_w=0.1,
                                       ica_max_iter=10000, ica_tolerance=1e-6,
                                       random_state=seed, pick_strategy="hungarian_any")
                    diagnostics = {k: result[k] for k in
                                   ("ica_runtime_sec", "selection_runtime_sec", "n_candidates_enumerated", "n_candidates_returned")}
                    if result["failed"] or result["n_candidates_returned"] == 0:
                        raise RuntimeError(result.get("last_error") or "No admissible candidate")
                    b = result["B_chosen"]
                    raw_groups = groups_from_adjacency(np.abs(b.T) > 0)
                elapsed, cpu = time.perf_counter() - start, time.process_time() - cpu_start
            except Exception:
                conn.send({"status": "error", "fit_runtime_sec": time.perf_counter()-start,
                           "fit_cpu_sec": time.process_time()-cpu_start, "error": traceback.format_exc()})
                return
        ev = time.perf_counter()
        groups, adj = group_lingam_adapter(raw_groups, b)
        metrics = condensation_metrics(weights, groups, adj)
        conn.send({"status": "ok", "fit_runtime_sec": elapsed, "fit_cpu_sec": cpu,
                   "evaluation_sec": time.perf_counter()-ev, **metrics, **diagnostics,
                   "groups": [sorted(g) for g in groups],
                   "known_edges": np.argwhere(adj).tolist(),
                   "unknown_coefficients": int(np.isnan(b).sum())})
    except Exception:
        conn.send({"status": "error", "error": traceback.format_exc()})
    finally:
        conn.close()


def timed_fit(data_path, method, seed, timeout_sec):
    import psutil
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=worker, args=(child, str(data_path), method, seed))
    started = time.perf_counter()
    process.start()
    child.close()
    monitor = psutil.Process(process.pid)
    peak = 0
    try:
        if not parent.poll(120):
            return {"status": "startup_error", "error": "Worker startup exceeded 120 s"}
        initial = parent.recv()
        startup = time.perf_counter()-started
        if not initial.get("ready"):
            return {**initial, "startup_sec": startup}
        fit_started = time.perf_counter()
        parent.send("fit")
        while True:
            try:
                peak = max(peak, monitor.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if parent.poll(0.05):
                result = parent.recv()
                break
            if time.perf_counter()-fit_started >= timeout_sec:
                try:
                    times = monitor.cpu_times()
                    cpu = times.user + times.system
                except psutil.NoSuchProcess:
                    cpu = None
                result = {"status": "timeout", "fit_runtime_sec": None,
                          "censor_time_sec": timeout_sec, "worker_cpu_sec": cpu,
                          "error": "Fit exceeded time budget; no partial model scored"}
                break
            if not process.is_alive():
                result = {"status": "error", "error": "Worker exited without a result"}
                break
        return {**result, "startup_sec": startup, "peak_rss_mb": peak/1024**2,
                "worker_elapsed_sec": time.perf_counter()-fit_started}
    except EOFError:
        return {"status": "error", "error": "Worker pipe closed unexpectedly"}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["hungarian", "group_lingam"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--regime", choices=["hard", "unstable"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    root = Path(__file__).resolve().parents[4]
    sources = [Path(__file__), root / "src/repare_cycle/benchmark.py", root / "src/repare_cycle/lingd.py", root / "src/repare_cycle/graph.py"]
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "lingam", "networkx")}
    result = timed_fit(args.data, args.method, args.seed, args.timeout)
    result.update(method=args.method, regime=args.regime, seed=args.seed, samp_size=args.n,
                  d=10, num_cycles=4, density=0.5, noise="laplace", timeout_sec=args.timeout,
                  dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  versions=versions, python=platform.python_version(), platform=platform.platform(),
                  cpu=platform.machine(), numerical_threads=1,
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    # Successful exact recovery requires an actual returned model, not an empty fallback.
    if result["status"] != "ok":
        result["exact_condensation"] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    tmp.replace(args.output)
    print(f"{args.method} {args.regime} n={args.n} seed={args.seed}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
