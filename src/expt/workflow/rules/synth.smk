# ─────────────────────────────────────────────────────────────────────────────
# Main synthetic experiment — single-method illustration of the
# condensation-identifiability theorem.
#
# Method: `lacerda` (FastICA → B̂ = I − Â⁻¹ → threshold → SCCs via Tarjan →
# inter-SCC edges from B̂). The story is not "which method wins" but rather:
# inter-SCC F₁ → 1 with n while variable-level F₁ saturates strictly below 1
# because intra-SCC structure is genuinely unidentifiable from observational
# data.
# ─────────────────────────────────────────────────────────────────────────────

synth_data_path  = "results/synth/regime={regime}/d={d}/num_cycles={num_cycles}/density={density}/samp_size={samp_size}/seed={seed}/"
synth_model_path = synth_data_path + "method={method}/"

synth_methods      = ["lacerda"]
synth_regimes      = ["easy"]
synth_d            = [10]
synth_num_cycles   = [3, 4, 5]
synth_densities    = [0.3, 0.5, 0.8]
synth_samp_sizes   = [50, 100, 500, 1000, 5000, 10000]
synth_seeds        = range(3)


rule synth_generate:
    output:
        synth_data_path + "dataset.npz",
    params:
        noise_dist="laplace",
    script:
        "../scripts/generate_synth.py"


rule synth_fit:
    input:
        data=synth_data_path + "dataset.npz",
    output:
        synth_model_path + "model.pkl",
    params:
        threshold=0.1,
        max_iter=10000,
        random_state=0,
    script:
        "../scripts/fit.py"


rule synth_evaluate:
    input:
        data=synth_data_path + "dataset.npz",
        model=synth_model_path + "model.pkl",
    output:
        synth_model_path + "metrics.csv",
    script:
        "../scripts/evaluate_synth.py"


rule synth_collect:
    input:
        expand(
            synth_model_path + "metrics.csv",
            method=synth_methods,
            regime=synth_regimes,
            d=synth_d,
            num_cycles=synth_num_cycles,
            density=synth_densities,
            samp_size=synth_samp_sizes,
            seed=synth_seeds,
        ),
    output:
        results="results/synth_results.csv",
    script:
        "../scripts/collect.py"


# Plot rules depend only on the slice of metrics they actually plot — one
# (regime, density) at a time — so building a single PDF doesn't transitively
# require every fit in the grid.
def _plot_slice_inputs(wildcards):
    return expand(
        synth_model_path + "metrics.csv",
        method=synth_methods,
        regime=[wildcards.regime],
        d=synth_d,
        num_cycles=synth_num_cycles,
        density=[wildcards.density],
        samp_size=synth_samp_sizes,
        seed=synth_seeds,
    )


rule synth_plot:
    input:
        _plot_slice_inputs,
    output:
        "results/synth_main_regime={regime}_density={density}.pdf",
    script:
        "../scripts/plot_synth.py"
