# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Reduced Wong-Wang (nonlinear neural-mass dynamics).
#
# Same benchmark graphs as Experiment 1 (reuses results/tvb/seed=*/dataset.npz
# for the ground-truth weights), one Reduced Wong-Wang population per node.
# A TVB-free calibration stage picks a (G, I_o) regime with one stable fixed
# point for every seed graph; TVB then generates EVERY observation by
# integrating the nonlinear network to steady state under truncated-Laplace
# per-region drive shifts at three amplitudes (small ≈ locally linear,
# large → saturation). Fits run the unchanged LiNG-D pipeline on centered
# settled states; held-out ± cluster stimulations validate the recovered
# C-DAG's descendant predictions against actual nonlinear TVB responses.
#
# Everything except tvb_rww_generate and tvb_rww_stimulate is TVB-free.
# Invoke with `snakemake tvb_rww_all --cores all` (needs the `.[tvb]` extra).
# ─────────────────────────────────────────────────────────────────────────────

tvb_rww_data_path = "results/tvb_rww/seed={seed}/amp={amp}/"
tvb_rww_model_path = tvb_rww_data_path + "samp_size={samp_size}/method={method}/"

tvb_rww_seeds = range(5)
tvb_rww_amps = ["small", "medium", "large"]
# Laplace scale [nA]. Calibrated against measured linearization error of the
# exact (Newton) equilibrium map at the chosen regime: the near-critical
# stability margin (~0.003/ms) amplifies inputs ~60x/nA along the slow mode,
# so local linearity requires far smaller drives than the raw parameter
# scale suggests. small ≈ 2% nonlinear, medium ≈ 9%, large ≈ 23%
# (recovery collapses at `large` — that cliff is the experiment's point).
tvb_rww_sigmas = {"small": 0.0005, "medium": 0.002, "large": 0.005}
tvb_rww_samp_sizes = [100, 500, 2000, 5000]
tvb_rww_methods = ["lacerda"]
tvb_rww_trunc_k = 3.0  # truncate shifts at +/- k * scale


rule tvb_rww_calibrate:
    input:
        [tvb_data_path.format(seed=s) + "dataset.npz" for s in tvb_rww_seeds],
    output:
        scan="results/tvb_rww_calibration.csv",
        chosen="results/tvb_rww/chosen_params.json",
    params:
        g_grid=[0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
        io_grid=[0.29, 0.31, 0.33, 0.35],
        margin_floor=0.002,  # 1/ms
        # Worst case: every region shifted to the truncation corner of the
        # largest amplitude simultaneously.
        robust_shift=tvb_rww_trunc_k * tvb_rww_sigmas["large"],
        # Linearized-oracle scoring: the unchanged pipeline threshold, on
        # standardized samples of the exactly-linearized equilibrium map.
        oracle_threshold=0.1,
        oracle_n=5000,  # the experiment's largest fit size — a fair ceiling
    script:
        "../scripts/calibrate_rww_tvb.py"


rule tvb_rww_verify:
    input:
        datasets=[
            tvb_data_path.format(seed=s) + "dataset.npz" for s in tvb_rww_seeds
        ],
        chosen="results/tvb_rww/chosen_params.json",
    output:
        "results/tvb_rww_verification.csv",
    script:
        "../scripts/verify_rww_tvb.py"


rule tvb_rww_generate:
    input:
        data=tvb_data_path + "dataset.npz",
        chosen="results/tvb_rww/chosen_params.json",
    output:
        tvb_rww_data_path + "dataset.npz",
    params:
        sigma=lambda wc: tvb_rww_sigmas[wc.amp],
        amp_id=lambda wc: tvb_rww_amps.index(wc.amp),
        trunc_k=tvb_rww_trunc_k,
        n_obs=tvb_n_obs,
        dt=1.0,
        chunk_ms=512.0,
        max_ms=20_000.0,
        tol=1e-7,
        batch_size=100,
    script:
        "../scripts/generate_rww_tvb.py"


rule tvb_rww_fit:
    input:
        data=tvb_rww_data_path + "dataset.npz",
    output:
        tvb_rww_model_path + "model.pkl",
    params:
        threshold=0.1,
        max_iter=10_000,
        random_state=0,
        pick_strategy="hungarian_any",
        sample_limit=lambda wc: int(wc.samp_size),
        # Settled Wong-Wang states have a nonzero baseline and
        # dynamics-reshaped per-node scales; standardize per fit (a
        # positive-diagonal similarity transform — support, SCCs, and the
        # condensation are preserved exactly).
        standardize=True,
    script:
        "../scripts/fit.py"


rule tvb_rww_evaluate:
    input:
        data=tvb_rww_data_path + "dataset.npz",
        model=tvb_rww_model_path + "model.pkl",
    output:
        tvb_rww_model_path + "metrics.csv",
    params:
        d=tvb_d,
        num_cycles=tvb_num_cycles,
        density=tvb_density,
        threshold=0.1,
        regime="tvb_rww",
    script:
        "../scripts/evaluate_tvb.py"


def _tvb_rww_metrics_inputs():
    return [
        tvb_rww_model_path.format(seed=s, amp=a, samp_size=n, method=m)
        + "metrics.csv"
        for s in tvb_rww_seeds
        for a in tvb_rww_amps
        for n in tvb_rww_samp_sizes
        for m in tvb_rww_methods
    ]


rule tvb_rww_collect:
    input:
        _tvb_rww_metrics_inputs(),
    output:
        results="results/tvb_rww_results.csv",
    script:
        "../scripts/collect.py"


rule tvb_rww_stimulate:
    input:
        data=tvb_data_path.format(seed=0) + "dataset.npz",
        chosen="results/tvb_rww/chosen_params.json",
    output:
        "results/tvb_rww/amp={amp}/stimulation.csv",
    params:
        sigma=lambda wc: tvb_rww_sigmas[wc.amp],
        # Stimulate at the truncation corner of the amplitude level, split
        # 1/sqrt(|SCC|) across members (mirrors Experiment 1's convention).
        stim_scale=tvb_rww_trunc_k,
        dt=1.0,
        chunk_ms=512.0,
        max_ms=40_000.0,
        # Tight tolerance: non-descendant responses are exactly zero in the
        # dynamics, so integration drift (~residual/margin) must stay well
        # below the response threshold used by score_rww_descendants.
        tol=1e-10,
    script:
        "../scripts/stimulate_rww_tvb.py"


rule tvb_rww_descendants:
    input:
        data=tvb_data_path.format(seed=0) + "dataset.npz",
        stimulations=expand(
            "results/tvb_rww/amp={amp}/stimulation.csv", amp=tvb_rww_amps
        ),
        models=[
            tvb_rww_model_path.format(seed=0, amp=a, samp_size=n, method="lacerda")
            + "model.pkl"
            for a in tvb_rww_amps
            for n in tvb_rww_samp_sizes
        ],
    output:
        "results/tvb_rww_descendants.csv",
    params:
        amps=tvb_rww_amps,
        samp_sizes=tvb_rww_samp_sizes,
        resp_threshold=1e-6,
    script:
        "../scripts/score_rww_descendants.py"


rule tvb_rww_plot:
    input:
        results="results/tvb_rww_results.csv",
        data=tvb_rww_data_path.format(seed=0, amp="medium") + "dataset.npz",
        model=tvb_rww_model_path.format(
            seed=0, amp="medium", samp_size=5000, method="lacerda"
        )
        + "model.pkl",
    output:
        recovery="results/tvb_rww.pdf",
        topology="results/tvb_rww_topology.pdf",
    params:
        samp_size=5000,
        amp="medium",
    script:
        "../scripts/plot_tvb_rww.py"


rule tvb_rww_all:
    input:
        "results/tvb_rww_calibration.csv",
        "results/tvb_rww_verification.csv",
        "results/tvb_rww_results.csv",
        "results/tvb_rww_descendants.csv",
        "results/tvb_rww.pdf",
        "results/tvb_rww_topology.pdf",
