"""Paper revision targets. Large benchmarks are explicit opt-in targets."""
import sys

paper_fig3_sizes = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]


rule paper_ablation_replot:
    input:
        "../../../data/reference/synth_ablation.csv.gz",
    output:
        "results/paper_ablation_selection.pdf",
    script:
        "../scripts/plot_ablation.py"


rule paper_ablation_accuracy_runtime:
    input:
        "../../../data/reference/synth_ablation.csv.gz",
    output:
        pdf="results/ablation_accuracy_runtime.pdf",
        cells="results/ablation_accuracy_runtime_cells.csv",
        summary="results/ablation_accuracy_runtime_summary.csv",
        caption="results/ablation_accuracy_runtime_caption.md",
    script:
        "../scripts/plot_ablation_accuracy_runtime.py"


rule paper_fig3_collect:
    input:
        saved="../../../data/reference/synth_results.csv.gz",
        fresh=expand(synth_model_path + "metrics.csv", regime=synth_regimes, d=[10],
               num_cycles=synth_num_cycles, density=synth_densities,
               samp_size=[10000, 50000, 100000], seed=synth_seeds, method=["lacerda"]),
    output:
        "results/paper_fig3.csv",
    script:
        "../scripts/collect_paper_fig3.py"


rule paper_fig3_plot:
    input:
        "results/paper_fig3.csv",
    output:
        "results/paper_fig3.pdf",
    script:
        "../scripts/plot_synth_combined.py"


full_config = config.get("group_lingam_full", {})
full_sizes = full_config.get("sizes", [100, 500, 1000, 2000, 5000, 10000])
full_seed_count = full_config.get("seeds", 10)
full_timeouts = {int(n): float(seconds) for n, seconds in full_config.get(
    "timeout_by_n", {100: 600, 500: 600, 1000: 1800, 2000: 7200, 5000: 21600, 10000: 86400}
).items()}
if (not full_sizes or any(not isinstance(n, int) or n <= 0 for n in full_sizes)
        or len(set(full_sizes)) != len(full_sizes)
        or not isinstance(full_seed_count, int) or full_seed_count < 1
        or any(n not in full_timeouts or full_timeouts[n] <= 0 for n in full_sizes)):
    raise ValueError("Invalid full GroupLiNGAM grid or missing/nonpositive time limit")

group_profiles = {
    "pilot": {"sizes": [100, 500, 1000], "seeds": range(3), "timeout": 120},
    # Runtime feasibility only: predeclared seed 0 in both regimes.
    "feasibility": {"sizes": [1000, 2000], "seeds": range(1), "timeout": 3600},
    "large_probe": {"sizes": [5000], "seeds": range(1), "timeout": 7200},
    "full": {"sizes": full_sizes, "seeds": range(full_seed_count)},
}
group_path = "results/group_lingam/{profile}/regime={regime}/n={samp_size}/seed={seed}/method={method}.json"


rule group_lingam_fit:
    input:
        data=lambda wc: synth_data_path.format(regime=wc.regime, d=10, num_cycles=4,
                                               density=0.5, samp_size=wc.samp_size, seed=wc.seed) + "dataset.npz",
        runner="scripts/benchmark_group_lingam.py",
        adapter="../../repare_cycle/benchmark.py",
        estimator="../../repare_cycle/lingd.py",
    output:
        group_path,
    wildcard_constraints:
        profile="pilot|feasibility|large_probe",
        method="hungarian|group_lingam",
        regime="hard|unstable",
    params:
        python=sys.executable,
        timeout=lambda wc: group_profiles[wc.profile]["timeout"],
    resources:
        # Prevent competing timed fits even when the workflow uses several cores.
        benchmark_slot=1,
    shell:
        "{params.python:q} {input.runner:q} --data {input.data:q} --output {output:q} "
        "--method {wildcards.method} --regime {wildcards.regime} --seed {wildcards.seed} "
        "--n {wildcards.samp_size} --timeout {params.timeout}"


rule group_lingam_full_fit:
    input:
        data=lambda wc: synth_data_path.format(regime=wc.regime, d=10, num_cycles=4,
                                               density=0.5, samp_size=wc.samp_size, seed=wc.seed) + "dataset.npz",
        runner="scripts/benchmark_group_lingam.py",
        adapter="../../repare_cycle/benchmark.py",
        estimator="../../repare_cycle/lingd.py",
    output:
        "results/group_lingam/full/regime={regime}/n={samp_size}/seed={seed}/method={method}.json",
    log:
        "results/group_lingam/full/logs/regime={regime}/n={samp_size}/seed={seed}/method={method}.log",
    wildcard_constraints:
        method="hungarian|group_lingam",
        regime="hard|unstable",
    params:
        python=sys.executable,
        timeout=lambda wc: full_timeouts[int(wc.samp_size)],
    resources:
        benchmark_slot=1,
    shell:
        "{params.python:q} {input.runner:q} --data {input.data:q} --output {output:q} "
        "--method {wildcards.method} --regime {wildcards.regime} --seed {wildcards.seed} "
        "--n {wildcards.samp_size} --timeout {params.timeout} > {log:q} 2>&1"


rule group_lingam_collect:
    input:
        lambda wc: expand(group_path, profile=[wc.profile], regime=["hard", "unstable"],
                          samp_size=group_profiles[wc.profile]["sizes"],
                          seed=group_profiles[wc.profile]["seeds"], method=["hungarian", "group_lingam"]),
    output:
        "results/group_lingam/{profile}/metrics.csv",
    script:
        "../scripts/collect_group_lingam.py"


rule group_lingam_plot:
    input:
        "results/group_lingam/{profile}/metrics.csv",
    output:
        pdf="results/group_lingam/{profile}/comparison.pdf",
        summary="results/group_lingam/{profile}/summary.csv",
        report="results/group_lingam/{profile}/report.md",
    script:
        "../scripts/plot_group_lingam.py"


rule group_lingam_sample_sizes_plot:
    input:
        "results/group_lingam/{profile}/metrics.csv",
    output:
        pdf="results/group_lingam/{profile}/sample_sizes.pdf",
        summary="results/group_lingam/{profile}/sample_sizes_summary.csv",
        report="results/group_lingam/{profile}/sample_sizes_table.md",
    script:
        "../scripts/plot_group_lingam_sample_sizes.py"
