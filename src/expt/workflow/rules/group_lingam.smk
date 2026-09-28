# GroupLiNGAM (lingam 1.13.0) vs ours over sample size at d=10, on the same
# datasets. Each fit is a timed, single-threaded worker checkpointed as JSON.
# Not part of `rule all` (GroupLiNGAM takes hours per fit at n=5000): launch
# with scripts/run_group_lingam.sh, which passes src/expt/config/group_lingam.yaml.
import sys

gl_config = config.get("group_lingam", {})
gl_sizes = gl_config.get("sizes", [100, 500, 1000, 2000, 5000, 10000])
gl_seeds = range(gl_config.get("seeds", 10))
gl_max_n = gl_config.get("group_lingam_max_n", 5000)  # ours runs on every size
gl_timeouts = {int(n): float(s) for n, s in gl_config.get("timeout_by_n", {}).items()}

gl_fit_path = "results/group_lingam/regime={regime}/n={samp_size}/seed={seed}/method={method}.json"


rule group_lingam_fit:
    input:
        data=lambda wc: synth_data_path.format(regime=wc.regime, d=10, num_cycles=4,
                                               density=0.5, samp_size=wc.samp_size, seed=wc.seed) + "dataset.npz",
        runner="scripts/benchmark_group_lingam.py",
        adapter="../../repare_cycle/benchmark.py",
        estimator="../../repare_cycle/lingd.py",
    output:
        gl_fit_path,
    log:
        "results/group_lingam/logs/regime={regime}/n={samp_size}/seed={seed}/method={method}.log",
    wildcard_constraints:
        method="hungarian|group_lingam",
        regime="hard|unstable",
    params:
        python=sys.executable,
        timeout=lambda wc: gl_timeouts[int(wc.samp_size)],
    resources:
        benchmark_slot=1,  # the launcher caps the total at parallel_fits
    shell:
        "{params.python:q} {input.runner:q} --data {input.data:q} --output {output:q} "
        "--method {wildcards.method} --regime {wildcards.regime} --seed {wildcards.seed} "
        "--n {wildcards.samp_size} --timeout {params.timeout} > {log:q} 2>&1"


rule group_lingam_collect:
    input:
        [gl_fit_path.format(regime=regime, samp_size=n, seed=seed, method=method)
         for regime in ["hard", "unstable"] for n in gl_sizes for seed in gl_seeds
         for method in ["hungarian", "group_lingam"] if method == "hungarian" or n <= gl_max_n],
    output:
        "results/group_lingam_metrics.csv",
    run:
        # One row per fit; groups/edges (lists) stay in the per-fit JSONs.
        import json
        import pandas as pd
        rows = [{k: v for k, v in json.load(open(path)).items() if not isinstance(v, (dict, list))}
                for path in input]
        (pd.DataFrame(rows).sort_values(["regime", "samp_size", "seed", "method"])
         .to_csv(output[0], index=False))


rule group_lingam_plot:
    input:
        "results/group_lingam_metrics.csv",
    output:
        "results/group_lingam_sample_sizes.pdf",
    params:
        max_n=gl_max_n,
    script:
        "../scripts/plot_group_lingam_sample_sizes.py"
