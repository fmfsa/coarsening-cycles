# ─────────────────────────────────────────────────────────────────────────────
# Whole-SCC hard-intervention experiment.
#
# do(X_S = c) on the largest non-trivial true SCC with downstream blocks,
# c_j = μ̂_j + σ̂_j. Compares Δμ = μ^{do} − μ^{obs} on downstream-SCC block
# averages: analytic truth vs (i) oracle-condensation block regression and
# (ii) end-to-end recovered condensation (Hungarian primary, first-stable
# secondary), with strict structural-validity accounting. Reuses the Grid A
# dataset rule at fixed d=10, κ=4, density=0.5 (n=50000 cells and seeds
# 10-19 are generated on demand by synth_generate).
# ─────────────────────────────────────────────────────────────────────────────

inter_regimes    = ["hard", "unstable"]
inter_samp_sizes = [100, 500, 1000, 5000, 50000]
inter_seeds      = range(20)
inter_d          = 10
inter_num_cycles = 4
inter_density    = 0.5

inter_data_path = (
    f"results/synth/regime={{regime}}/d={inter_d}/num_cycles={inter_num_cycles}/"
    f"density={inter_density}/samp_size={{samp_size}}/seed={{seed}}/"
)
inter_out_path = (
    "results/intervention/regime={regime}/samp_size={samp_size}/seed={seed}/"
)


rule intervention_run:
    input:
        data=inter_data_path + "dataset.npz",
    output:
        inter_out_path + "metrics.csv",
    params:
        threshold=0.1,
        max_iter=10000,
    script:
        "../scripts/intervention_run.py"


rule intervention_collect:
    input:
        [
            inter_out_path.format(regime=r, samp_size=n, seed=s) + "metrics.csv"
            for r in inter_regimes
            for n in inter_samp_sizes
            for s in inter_seeds
        ],
    output:
        results="results/intervention_effects.csv",
    script:
        "../scripts/collect.py"


rule intervention_plot:
    input:
        "results/intervention_effects.csv",
    output:
        pdf="results/intervention_effects.pdf",
        table="results/intervention_table.txt",
    script:
        "../scripts/plot_intervention.py"
