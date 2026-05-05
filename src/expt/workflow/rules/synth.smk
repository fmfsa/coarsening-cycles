# ─────────────────────────────────────────────────────────────────────────────
# Synthetic experiments.
#
# Two grids are run side-by-side:
#
# (A) MAIN d=10 grid — demonstrates the condensation-identifiability gap
#     (stable vs unstable regime) for our procedure on small graphs.
#       lacerda     — ICA-LiNG-D + first-stable filter.
#       lacerda_rnd — ICA-LiNG-D + uniform draw from full equivalence class.
#
# (B) SCALABILITY grid — d ∈ {20, 50, 100, 500} at fixed κ=10 SCCs.
#     Compares our method against `disjointCycles` (Drton, Garrote-Lopez,
#     Nikov, Robeva, Wang), the cycle-aware higher-order-moment baseline.
# ─────────────────────────────────────────────────────────────────────────────

synth_data_path  = "results/synth/regime={regime}/d={d}/num_cycles={num_cycles}/density={density}/samp_size={samp_size}/seed={seed}/"
synth_model_path = synth_data_path + "method={method}/"

# (A) Main d=10 grid
synth_methods    = ["lacerda", "lacerda_rnd"]
synth_regimes    = ["hard", "unstable"]
synth_d          = [10]
synth_num_cycles = [3, 4, 5]
synth_densities  = [0.3, 0.5, 0.8]
synth_samp_sizes = [50, 100, 500, 1000, 5000]
synth_seeds      = range(10)

# (B) Scalability grid
scal_methods     = ["lacerda", "disjointcycles"]
scal_regime      = "hard"
scal_density     = 0.5
scal_num_cycles  = 10                    # k = 10 SCCs, fixed
scal_d           = [20, 50, 100, 500]
scal_samp_sizes  = [50, 100, 500, 1000, 5000]
scal_seeds       = range(10)


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
        pick_strategy=lambda wc: "random_admissible" if wc.method == "lacerda_rnd" else "first_stable",
        alpha=0.01,
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


def _all_metrics_inputs():
    """Cartesian product over the d=10 main grid."""
    return [
        synth_model_path.format(
            method=m, regime=r, d=dd, num_cycles=k,
            density=rho, samp_size=n, seed=s,
        ) + "metrics.csv"
        for m in synth_methods
        for r in synth_regimes
        for dd in synth_d
        for k in synth_num_cycles
        for rho in synth_densities
        for n in synth_samp_sizes
        for s in synth_seeds
    ]


def _scalability_metrics_inputs():
    """Cartesian product over the scalability grid.

    Cells with samp_size < d are skipped: ICA-LiNG-D requires at least d
    samples to estimate a d×d unmixing matrix (sklearn's FastICA silently
    caps n_components at min(n, d), which breaks the downstream
    B = I - W identity). The same constraint is reasonable for
    disjointCycles' higher-order moment tests.
    """
    return [
        synth_model_path.format(
            method=m, regime=scal_regime, d=dd, num_cycles=scal_num_cycles,
            density=scal_density, samp_size=n, seed=s,
        ) + "metrics.csv"
        for m in scal_methods
        for dd in scal_d
        for n in scal_samp_sizes
        if n >= dd
        for s in scal_seeds
    ]


rule synth_collect:
    input:
        _all_metrics_inputs(),
    output:
        results="results/synth_results.csv",
    script:
        "../scripts/collect.py"


rule synth_scalability_collect:
    input:
        _scalability_metrics_inputs(),
    output:
        results="results/synth_scalability.csv",
    script:
        "../scripts/collect.py"


# Plot rules depend only on the slice of metrics they actually plot — one
# (regime, density) at a time — so building a single PDF doesn't transitively
# require every fit in the grid.
def _plot_slice_inputs(wildcards):
    return [
        synth_model_path.format(
            method=m, regime=wildcards.regime, d=dd, num_cycles=k,
            density=wildcards.density, samp_size=n, seed=s,
        ) + "metrics.csv"
        for m in synth_methods
        for dd in synth_d
        for k in synth_num_cycles
        for n in synth_samp_sizes
        for s in synth_seeds
    ]


rule synth_plot:
    input:
        _plot_slice_inputs,
    output:
        "results/synth_main_regime={regime}_density={density}.pdf",
    script:
        "../scripts/plot_synth.py"


rule synth_plot_combined:
    input:
        "results/synth_results.csv",
    output:
        "results/synth_main_combined.pdf",
    script:
        "../scripts/plot_synth_combined.py"


rule synth_plot_scalability:
    input:
        "results/synth_scalability.csv",
    output:
        "results/scalability_disjointcycles.pdf",
    script:
        "../scripts/plot_scalability.py"
