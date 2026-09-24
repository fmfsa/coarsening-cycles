# ─────────────────────────────────────────────────────────────────────────────
# TVB linear-equilibrium pilot.
#
# Controlled benchmark: d=20, five non-trivial SCCs, inter/intra-SCC density
# 0.3, spectral radius rescaled to 0.9, standardized Laplace inputs.
# Observations are exact equilibria X = E (I - W)^{-1} — the quantity TVB's
# linear model (gamma=-1, linear coupling a=1, zero delays, constant regional
# stimulus) converges to; tvb_validate confirms this numerically for seed 0.
# One 5000-sample dataset per seed; fits run on nested prefixes via the
# sample_limit param, so the grid is 5 seeds × 4 sample sizes = 20 fits.
#
# Only tvb_validate and tvb_preflight need TVB (install the `.[tvb]` extra);
# generation, fitting, and evaluation are TVB-free, and the default
# `rule all` does not depend on any of this.
# ─────────────────────────────────────────────────────────────────────────────

tvb_data_path = "results/tvb/seed={seed}/"
tvb_model_path = tvb_data_path + "samp_size={samp_size}/method={method}/"

tvb_seeds = range(5)
tvb_samp_sizes = [100, 500, 2000, 5000]
tvb_methods = ["lacerda"]

tvb_d = 20
tvb_num_cycles = 5
tvb_density = 0.3
tvb_n_obs = 5000


rule tvb_generate:
    output:
        tvb_data_path + "dataset.npz",
    params:
        d=tvb_d,
        num_cycles=tvb_num_cycles,
        density=tvb_density,
        intra_scc_density=tvb_density,
        weight_lo=0.5,
        weight_hi=0.95,
        target_rho=0.9,
        n_obs=tvb_n_obs,
    script:
        "../scripts/generate_tvb.py"


rule tvb_fit:
    input:
        data=tvb_data_path + "dataset.npz",
    output:
        tvb_model_path + "model.pkl",
    params:
        threshold=0.1,
        max_iter=10_000,
        random_state=0,
        # Hungarian representative: O(d³) at d=20, and Prop. 3 guarantees any
        # admissible member of the equivalence class has the right condensation.
        pick_strategy="hungarian_any",
        # Fit on a prefix of the single 5000-sample dataset.
        sample_limit=lambda wc: int(wc.samp_size),
    script:
        "../scripts/fit.py"


rule tvb_evaluate:
    input:
        data=tvb_data_path + "dataset.npz",
        model=tvb_model_path + "model.pkl",
    output:
        tvb_model_path + "metrics.csv",
    params:
        d=tvb_d,
        num_cycles=tvb_num_cycles,
        density=tvb_density,
        threshold=0.1,
    script:
        "../scripts/evaluate_tvb.py"


def _tvb_metrics_inputs():
    return [
        tvb_model_path.format(seed=s, samp_size=n, method=m) + "metrics.csv"
        for s in tvb_seeds
        for n in tvb_samp_sizes
        for m in tvb_methods
    ]


rule tvb_collect:
    input:
        _tvb_metrics_inputs(),
    output:
        results="results/tvb_results.csv",
    script:
        "../scripts/collect.py"


rule tvb_validate:
    input:
        data="results/tvb/seed=0/dataset.npz",
    output:
        "results/tvb_equilibrium_validation.csv",
    params:
        n_episodes=32,
        dt=0.0625,
        chunk_ms=8.0,
        # The slowest seed-0 mode decays at ~0.089/ms, so the heaviest-tail
        # Laplace episodes cross the 1e-7 residual right around 200 ms
        # (worst observed: 200.7 ms). 250 ms gives every episode margin;
        # the bridge default stays at 200 ms.
        max_ms=250.0,
        tol=1e-7,
    script:
        "../scripts/validate_tvb.py"


rule tvb_preflight:
    output:
        "results/tvb_default_preflight.csv",
    script:
        "../scripts/preflight_tvb.py"


rule tvb_intervention_descendants:
    input:
        data="results/tvb/seed=0/dataset.npz",
        validation="results/tvb_equilibrium_validation.csv",
        models=[
            tvb_model_path.format(seed=0, samp_size=n, method="lacerda")
            + "model.pkl"
            for n in tvb_samp_sizes
        ],
    output:
        "results/tvb_intervention_descendants.csv",
    params:
        samp_sizes=tvb_samp_sizes,
        # TVB descendant responses are O(0.1+); non-descendants are zero up
        # to the 1e-7 integration residual. 1e-3 splits them with margin.
        resp_threshold=1e-3,
    script:
        "../scripts/intervention_descendants_tvb.py"


rule tvb_plot:
    input:
        results="results/tvb_results.csv",
        validation="results/tvb_equilibrium_validation.csv",
        preflight="results/tvb_default_preflight.csv",
        interventions="results/tvb_intervention_descendants.csv",
    output:
        "results/tvb_pilot.pdf",
    script:
        "../scripts/plot_tvb.py"


rule tvb_plot_topology:
    input:
        data="results/tvb/seed=0/dataset.npz",
        model=tvb_model_path.format(seed=0, samp_size=5000, method="lacerda")
        + "model.pkl",
    output:
        "results/tvb_topology.pdf",
    params:
        samp_size=5000,
    script:
        "../scripts/plot_tvb_topology.py"


rule tvb_all:
    input:
        "results/tvb_results.csv",
        "results/tvb_default_preflight.csv",
        "results/tvb_equilibrium_validation.csv",
        "results/tvb_intervention_descendants.csv",
        "results/tvb_pilot.pdf",
        "results/tvb_topology.pdf",
