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
scal_d           = [20, 50, 100]#, 500]
scal_samp_sizes  = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]
scal_seeds       = range(10)

# (C) Strict disjoint-cycles micro grid (d=20, κ=5).
# Each SCC is a single Hamilton cycle (no intra-SCC chord edges), so every
# directed simple cycle is pairwise vertex-disjoint by construction —
# the regime where Drton et al.'s assumption holds exactly. The (d, κ)
# pair is unused by the other two grids, so wildcard-driven dispatch
# below is collision-free.
disjoint_methods    = ["lacerda", "disjointcycles"]
disjoint_regime     = "hard"
disjoint_density    = 0.5
disjoint_d          = 20
disjoint_num_cycles = 5
disjoint_samp_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
disjoint_seeds      = range(10)


def _is_scal_cell(wc) -> bool:
    return int(wc.num_cycles) == 10


def _is_disjoint_cell(wc) -> bool:
    return int(wc.d) == 20 and int(wc.num_cycles) == 5


rule synth_generate:
    output:
        synth_data_path + "dataset.npz",
    params:
        # Main d=10 grid uses Laplace (symmetric, non-Gaussian — fine for
        # ICA-LiNG-D). The scalability and disjoint-micro grids need
        # ASYMMETRIC noise because disjointCycles' tests rely on non-zero
        # higher-order cumulants; Laplace has zero third moment and
        # silently fails. Exponential is non-Gaussian + skewed, so both
        # methods can exploit it on the same data.
        noise_dist=lambda wc: (
            "exponential" if _is_scal_cell(wc) or _is_disjoint_cell(wc)
            else "laplace"
        ),
        # Disjoint-micro grid suppresses intra-SCC chord edges; every
        # other grid keeps the legacy behaviour (chord prob = density).
        intra_scc_density=lambda wc: 0.0 if _is_disjoint_cell(wc) else None,
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
        # Pick-strategy dispatch:
        #   lacerda_rnd       — uniform over equivalence class (used by §5.2 figure
        #                       to expose the variable-level identifiability gap).
        #   lacerda @ k=10    — scalability grid. Use hungarian_any: O(d³) fast
        #                       path that returns ONE admissible representative,
        #                       which is sufficient because Theorem 1 makes the
        #                       condensation (= what ARI/F1 measure) invariant
        #                       across the equivalence class. N-rooks enumeration
        #                       blows up combinatorially at d≥50.
        #   lacerda @ k≠10    — d=10 main grid. Keep first_stable (cheap at d=10
        #                       and supports the §5.2 stable/unstable narrative).
        pick_strategy=lambda wc: (
            "random_admissible" if wc.method == "lacerda_rnd"
            else "hungarian_any" if _is_scal_cell(wc) or _is_disjoint_cell(wc)
            else "first_stable"
        ),
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


def _disjoint_metrics_inputs():
    """Cartesian product over the strict-disjoint-cycles micro grid."""
    return [
        synth_model_path.format(
            method=m, regime=disjoint_regime, d=disjoint_d,
            num_cycles=disjoint_num_cycles, density=disjoint_density,
            samp_size=n, seed=s,
        ) + "metrics.csv"
        for m in disjoint_methods
        for n in disjoint_samp_sizes
        if n >= disjoint_d
        for s in disjoint_seeds
    ]


rule synth_disjoint_collect:
    input:
        _disjoint_metrics_inputs(),
    output:
        results="results/synth_disjoint.csv",
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


rule synth_plot_disjoint:
    input:
        "results/synth_disjoint.csv",
    output:
        "results/disjoint_micro.pdf",
    script:
        "../scripts/plot_disjoint.py"
