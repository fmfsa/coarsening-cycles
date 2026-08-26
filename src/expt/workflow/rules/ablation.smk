# ─────────────────────────────────────────────────────────────────────────────
# Candidate-selection ablation (fixed, verified ICA estimate).
#
# Every procedure uses the same fixed FastICA settings; stored W hashes verify
# that their separately staged fits produce a bit-identical unmixing matrix W:
#   abl_first_stable — enumeration + first-stable choice (experimental
#                      baseline; cap-truncated where indicated);
#   abl_random_r{r}  — enumeration + uniform draw r ∈ 0..4 from the same
#                      enumerated candidate set (draws share one dataset/W:
#                      average within-cell before across-seed CIs);
#   abl_hungarian    — single Hungarian representative, ZERO enumeration
#                      (the polynomial-time path of Algorithm 1).
#   abl_arbitrary_r{r} — uniform random row permutation without enforcing
#                        admissibility (negative control, 5 draws).
#
# Reuses Grid A datasets and the stock synth_evaluate rule (the branch
# directories follow the method= convention). A small d ∈ {5, 10, 20} arm
# measures enumeration cost growth; cells whose
# enumeration hits the max_perms cap are flagged (enumeration_cap_hit) and
# represent a TRUNCATED candidate list, not the full equivalence class.
# ─────────────────────────────────────────────────────────────────────────────

abl_n_random_draws = 5
# Collected/reported branches. The enumeration+random-pick branches
# (method=abl_random_r*) are still produced by ablation_fit for provenance
# but are deliberately NOT collected: their role (sensitivity to the choice
# of an enumerated representative) is superseded by abl_randhun_r* — a
# randomised admissible representative obtained WITHOUT enumeration via a
# random-cost matching (the literal "random solution instead of the full
# enumeration"). Four reported selection procedures:
#   abl_first_stable — enumeration baseline (original ICA-LiNG-D recipe);
#   abl_hungarian    — Algorithm 1's deterministic no-enumeration path;
#   abl_randhun_r{r} — randomised no-enumeration representative, 5 draws.
#   abl_arbitrary_r{r} — unconstrained uniform row permutations, 5 draws;
#                        admissibility is measured but never enforced.
abl_branches = (
    ["abl_first_stable", "abl_hungarian"]
    + [f"abl_random_r{r}" for r in range(abl_n_random_draws)]
    + [f"abl_randhun_r{r}" for r in range(abl_n_random_draws)]
    + [f"abl_arbitrary_r{r}" for r in range(abl_n_random_draws)]
)

# Ablation-only extension of the sample-size axis: one additional point at
# n=10000 beyond the main grid's 5000, without touching the §5 grids. The
# (κ=5, λ=0.5, hard, n=10000) cells coincide with the cost arm's (10, 5)
# cells and are shared on disk.
abl_samp_sizes = list(synth_samp_sizes) + [10000]

# Enumeration-cost arm: explicit (d, κ) pairs, κ scaling with d because
# random_cyclic_graph needs d ≥ 2κ. d=10/κ=5 anchors the arm to the main
# grid's setting; d=20/κ=10 reuses scalability-grid-style datasets (the
# num_cycles=10 dispatch gives those cells exponential noise, the smaller
# ones Laplace. Because noise affects W_hat and hence the admissible matching
# set, cross-d runtime comparisons are indicative rather than noise-controlled.
# d ≥ 30 is excluded from the reported grid; the reproducible in-grid timing
# and truncation fields are the only scaling numbers cited in the rebuttal.
abl_cost_cells      = [(5, 2), (10, 5), (20, 10)]   # (d, num_cycles)
abl_scal_samp_sizes = [1000, 10000, 50000]
abl_scal_seeds      = range(10)
abl_scal_regimes    = ["hard", "unstable"]

# Cost-arm branches reported in the rebuttal's d=20 table: enumeration +
# first-stable (baseline), enumeration + uniformly random admissible
# candidate (the reviewer's random-selection ablation; still requires
# enumeration), magnitude-based Hungarian (Algorithm 1, no enumeration),
# and the arbitrary-permutation control (not admissible, negative control).
abl_scal_branches = (
    ["abl_first_stable", "abl_hungarian"]
    + [f"abl_random_r{r}" for r in range(abl_n_random_draws)]
    + [f"abl_arbitrary_r{r}" for r in range(abl_n_random_draws)]
)


rule ablation_fit:
    input:
        data=synth_data_path + "dataset.npz",
    output:
        first_stable=synth_data_path + "method=abl_first_stable/model.pkl",
        hungarian=synth_data_path + "method=abl_hungarian/model.pkl",
        random_r0=synth_data_path + "method=abl_random_r0/model.pkl",
        random_r1=synth_data_path + "method=abl_random_r1/model.pkl",
        random_r2=synth_data_path + "method=abl_random_r2/model.pkl",
        random_r3=synth_data_path + "method=abl_random_r3/model.pkl",
        random_r4=synth_data_path + "method=abl_random_r4/model.pkl",
    params:
        threshold=0.1,
        max_iter=10000,
        max_perms=10000,
        n_random_draws=abl_n_random_draws,
    script:
        "../scripts/ablation_fit.py"


rule ablation_randhun_fit:
    input:
        data=synth_data_path + "dataset.npz",
    output:
        randhun_r0=synth_data_path + "method=abl_randhun_r0/model.pkl",
        randhun_r1=synth_data_path + "method=abl_randhun_r1/model.pkl",
        randhun_r2=synth_data_path + "method=abl_randhun_r2/model.pkl",
        randhun_r3=synth_data_path + "method=abl_randhun_r3/model.pkl",
        randhun_r4=synth_data_path + "method=abl_randhun_r4/model.pkl",
    params:
        threshold=0.1,
        max_iter=10000,
        n_random_draws=abl_n_random_draws,
    script:
        "../scripts/ablation_randhun_fit.py"


rule ablation_arbitrary_fit:
    input:
        data=synth_data_path + "dataset.npz",
    output:
        arbitrary_r0=synth_data_path + "method=abl_arbitrary_r0/model.pkl",
        arbitrary_r1=synth_data_path + "method=abl_arbitrary_r1/model.pkl",
        arbitrary_r2=synth_data_path + "method=abl_arbitrary_r2/model.pkl",
        arbitrary_r3=synth_data_path + "method=abl_arbitrary_r3/model.pkl",
        arbitrary_r4=synth_data_path + "method=abl_arbitrary_r4/model.pkl",
    params:
        threshold=0.1,
        max_iter=10000,
        n_random_draws=abl_n_random_draws,
    script:
        "../scripts/ablation_arbitrary_fit.py"


# method=abl_* model.pkl paths must never fall through to synth_fit
# (fit.py would reject the unknown method).
ruleorder: ablation_fit > synth_fit
ruleorder: ablation_randhun_fit > synth_fit
ruleorder: ablation_arbitrary_fit > synth_fit


def _ablation_metrics_inputs():
    """Ablation branches over the full Grid A (main d=10) grid, with the
    ablation-only n=10000 extension."""
    return [
        synth_model_path.format(
            method=m, regime=r, d=dd, num_cycles=k,
            density=rho, samp_size=n, seed=s,
        ) + "metrics.csv"
        for m in abl_branches
        for r in synth_regimes
        for dd in synth_d
        for k in synth_num_cycles
        for rho in synth_densities
        for n in abl_samp_sizes
        for s in synth_seeds
    ]


def _ablation_scal_metrics_inputs():
    """Enumeration-cost arm over the (d, κ) pairs of abl_cost_cells,
    both regimes, with the reported branch set."""
    return [
        synth_model_path.format(
            method=m, regime=r, d=dd, num_cycles=k,
            density=scal_density, samp_size=n, seed=s,
        ) + "metrics.csv"
        for m in abl_scal_branches
        for r in abl_scal_regimes
        for dd, k in abl_cost_cells
        for n in abl_scal_samp_sizes
        for s in abl_scal_seeds
    ]


rule ablation_collect:
    input:
        _ablation_metrics_inputs(),
    output:
        results="results/synth_ablation.csv",
    script:
        "../scripts/collect.py"


rule ablation_scal_collect:
    input:
        _ablation_scal_metrics_inputs(),
    output:
        results="results/synth_ablation_scal.csv",
    script:
        "../scripts/collect.py"


rule ablation_plot:
    input:
        "results/synth_ablation.csv",
    output:
        "results/ablation_selection.pdf",
    script:
        "../scripts/plot_ablation.py"


rule split_merge_plot:
    input:
        "results/synth_results.csv",
    output:
        "results/split_merge.pdf",
    script:
        "../scripts/plot_split_merge.py"
