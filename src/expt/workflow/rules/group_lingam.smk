"""GroupLiNGAM (lingam 1.13.0) vs ours over sample size at d=10.

Both methods see the same dataset per regime/n/seed. Every fit runs in its own
timed worker with one numerical thread (scripts/benchmark_group_lingam.py) and
is checkpointed as JSON, so the sweep resumes after interruption.

Not part of `rule all`: GroupLiNGAM takes hours per fit at n=5000. Launch with
scripts/run_group_lingam.sh, which passes src/expt/config/group_lingam.yaml.
"""
import sys

gl_config = config.get("group_lingam", {})
gl_sizes = gl_config.get("sizes", [100, 500, 1000, 2000, 5000, 10000])
gl_seed_count = gl_config.get("seeds", 10)
# GroupLiNGAM is fitted only up to this n; ours runs on every size.
gl_max_n = gl_config.get("group_lingam_max_n", 5000)
gl_timeouts = {int(n): float(seconds) for n, seconds in gl_config.get(
    "timeout_by_n", {100: 600, 500: 600, 1000: 1800, 2000: 7200, 5000: 86400, 10000: 86400}
).items()}
if (not gl_sizes or any(not isinstance(n, int) or n <= 0 for n in gl_sizes)
        or len(set(gl_sizes)) != len(gl_sizes)
        or not isinstance(gl_seed_count, int) or gl_seed_count < 1
        or not isinstance(gl_max_n, int) or gl_max_n < min(gl_sizes)
        or any(n not in gl_timeouts or gl_timeouts[n] <= 0 for n in gl_sizes)):
    raise ValueError("Invalid GroupLiNGAM grid, size cap, or missing/nonpositive time limit")

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
        # Caps concurrent timed fits (the launcher sets the total from parallel_fits).
        benchmark_slot=1,
    shell:
        "{params.python:q} {input.runner:q} --data {input.data:q} --output {output:q} "
        "--method {wildcards.method} --regime {wildcards.regime} --seed {wildcards.seed} "
        "--n {wildcards.samp_size} --timeout {params.timeout} > {log:q} 2>&1"


rule group_lingam_collect:
    input:
        [gl_fit_path.format(regime=regime, samp_size=n, seed=seed, method=method)
         for regime in ["hard", "unstable"] for n in gl_sizes for seed in range(gl_seed_count)
         for method in ["hungarian", "group_lingam"] if method == "hungarian" or n <= gl_max_n],
    output:
        "results/group_lingam_metrics.csv",
    script:
        "../scripts/collect_group_lingam.py"


rule group_lingam_plot:
    input:
        "results/group_lingam_metrics.csv",
    output:
        pdf="results/group_lingam_sample_sizes.pdf",
        summary="results/group_lingam_sample_sizes_summary.csv",
        report="results/group_lingam_sample_sizes_table.md",
    params:
        max_n=gl_max_n,
        protocol_note=gl_config.get("protocol_note", ""),
    script:
        "../scripts/plot_group_lingam_sample_sizes.py"
