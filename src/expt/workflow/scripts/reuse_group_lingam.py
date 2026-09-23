"""Copy compatible completed fits into a new sweep without changing originals.

Run from the repository root before starting Snakemake. Timed-out attempts
are never reused. Data, implementation hashes, versions, and fit settings
must match; provenance of each reused result remains explicit.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def compatible(result, expected, source_hashes, versions, dataset_hash, timeout):
    return (result.get("status") == "ok"
            and all(result.get(k) == v for k, v in expected.items())
            and result.get("source_sha256") == source_hashes
            and result.get("versions") == versions
            and result.get("dataset_sha256") == dataset_hash
            and result.get("python") == platform.python_version()
            and result.get("platform") == platform.platform()
            and result.get("numerical_threads") == 1
            and 0 <= result.get("fit_runtime_sec", float("inf")) <= timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--sources", nargs="+", default=["pilot", "feasibility", "large_probe"])
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    if args.profile in args.sources or "/" in args.profile or args.profile in (".", ".."):
        parser.error("Destination must be a new profile name")
    root = Path(__file__).resolve().parents[4]
    workflow = root / "src/expt/workflow"
    results = workflow / "results/group_lingam"
    files = [workflow / "scripts/benchmark_group_lingam.py",
             root / "src/repare_cycle/benchmark.py", root / "src/repare_cycle/lingd.py",
             root / "src/repare_cycle/graph.py"]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "lingam", "networkx")}
    copied = 0
    for regime in ("hard", "unstable"):
        for n in args.sizes:
            for seed in range(args.seeds):
                data = workflow / f"results/synth/regime={regime}/d=10/num_cycles=4/density=0.5/samp_size={n}/seed={seed}/dataset.npz"
                if not data.exists():
                    continue
                dataset_hash = hashlib.sha256(data.read_bytes()).hexdigest()
                for method in ("hungarian", "group_lingam"):
                    relative = Path(f"regime={regime}/n={n}/seed={seed}/method={method}.json")
                    dest = results / args.profile / relative
                    if dest.exists():
                        continue
                    expected = dict(regime=regime, samp_size=n, seed=seed, method=method,
                                    d=10, num_cycles=4, density=.5, noise="laplace")
                    for source in args.sources:
                        path = results / source / relative
                        if not path.exists():
                            continue
                        result = json.loads(path.read_text())
                        if not compatible(result, expected, hashes, versions, dataset_hash, args.timeout):
                            continue
                        result["reused_from"] = str(path.relative_to(root))
                        result["reused_result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                        result["original_timeout_sec"] = result["timeout_sec"]
                        result["timeout_sec"] = args.timeout
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        temp = dest.with_suffix(".tmp")
                        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
                        temp.replace(dest)
                        copied += 1
                        break
    print(f"Reused {copied} compatible completed fits in {args.profile}; originals unchanged.")


if __name__ == "__main__":
    main()
