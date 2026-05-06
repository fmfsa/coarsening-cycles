# ─────────────────────────────────────────────────────────────────────────────
# Threshold-sensitivity sweep (App. C.2 — "Threshold sensitivity").
#
# Probes the two failure modes of Prop. 3 by sweeping τ at several sample
# sizes in the standard "hard" regime (w ∈ [0.5, 0.95]) at κ=4, λ=0.5, d=10.
# Reuses scripts/generate_synth.py, fit.py, evaluate_synth.py, collect.py —
# only adds a {threshold} wildcard to the fit/evaluate/collect path. The fit
# script already routes the threshold parameter into run_lingd().
# ─────────────────────────────────────────────────────────────────────────────

thr_data_path  = (
    "results/synth_threshold/regime={regime}/d={d}/num_cycles={num_cycles}/"
    "density={density}/samp_size={samp_size}/seed={seed}/"
)
thr_model_path = thr_data_path + "method={method}/threshold={threshold}/"

thr_regimes      = ["hard"]
thr_d            = 10
thr_num_cycles   = 4
thr_density      = 0.5
thr_samp_sizes   = [500, 5000, 50000]
thr_thresholds   = ["0.001", "0.005", "0.01", "0.05", "0.1", "0.2", "0.3", "0.5", "0.7", "1.0"]
thr_seeds        = range(10)
thr_methods      = ["lacerda"]


rule threshold_generate:
    output:
        thr_data_path + "dataset.npz",
    params:
        # Match the main d=10 grid: symmetric Laplace noise, no chord
        # suppression. Both weight regimes are stable (target_rho=None) so
        # CyclicLinearSEM uses its default ρ < 1 path.
        noise_dist="laplace",
        intra_scc_density=None,
    script:
        "../scripts/generate_synth.py"


rule threshold_fit:
    input:
        data=thr_data_path + "dataset.npz",
    output:
        thr_model_path + "model.pkl",
    params:
        threshold=lambda wc: float(wc.threshold),
        max_iter=10_000,
        random_state=0,
        # First-stable filter at d=10 to mirror the main-experiment recipe;
        # the sweep is about τ, not about the equivalence-class member chosen.
        pick_strategy="first_stable",
        alpha=0.01,
    script:
        "../scripts/fit.py"


rule threshold_evaluate:
    input:
        data=thr_data_path + "dataset.npz",
        model=thr_model_path + "model.pkl",
    output:
        thr_model_path + "metrics.csv",
    script:
        "../scripts/evaluate_synth.py"


def _threshold_metrics_inputs():
    return [
        thr_model_path.format(
            method=m, regime=r, d=thr_d, num_cycles=thr_num_cycles,
            density=thr_density, samp_size=n, seed=s, threshold=t,
        ) + "metrics.csv"
        for m in thr_methods
        for r in thr_regimes
        for n in thr_samp_sizes
        for s in thr_seeds
        for t in thr_thresholds
    ]


rule threshold_collect:
    input:
        _threshold_metrics_inputs(),
    output:
        results="results/synth_threshold.csv",
    script:
        "../scripts/collect.py"


rule threshold_plot:
    input:
        "results/synth_threshold.csv",
    output:
        "results/synth_threshold.pdf",
    script:
        "../scripts/plot_threshold.py"
