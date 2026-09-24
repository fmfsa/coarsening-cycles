"""TVB linear-equilibrium pilot figure (four panels).

  (a) Default-connectome preflight: largest-SCC fraction and directed
      reciprocity versus retained-edge fraction — the realism case study,
      showing why the recovery claims live on the controlled benchmark.
  (b) Recovery on the controlled benchmark: ARI of the SCC partition,
      projected cluster-DAG F1, and DIRECT C-DAG F1 (exact-match edges of
      the recovered `model.dag`) versus sample size (median + 95% CI over
      5 seeds). The direct metric is 0 whenever the estimate collapses to
      one cluster, unlike the projected one.
  (c) Exact versus TVB-integrated equilibria and intervention responses,
      with the identity line — the numerical-validation panel (seed 0).
  (d) Descendant validation: graph-reachability F1 (5 seeds) and the
      TVB-intervention-validated descendant F1 (seed 0 — C-DAG response
      predictions scored against actual TVB stimulation responses).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

ROSE_WINE = {
    "pale_rose":  "#E9C5C2",
    "rose":       "#AE6B91",
    "deep_wine":  "#2C1E3D",
}

sns.set_context("paper", font_scale=1.8)
sns.set_style("white")
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

try:
    results_path = snakemake.input.results        # type: ignore[name-defined]
    validation_path = snakemake.input.validation  # type: ignore[name-defined]
    preflight_path = snakemake.input.preflight    # type: ignore[name-defined]
    interventions_path = snakemake.input.interventions  # type: ignore[name-defined]
    pdf_path = snakemake.output[0]                # type: ignore[name-defined]
except NameError:
    results_path = "results/tvb_results.csv"
    validation_path = "results/tvb_equilibrium_validation.csv"
    preflight_path = "results/tvb_default_preflight.csv"
    interventions_path = "results/tvb_intervention_descendants.csv"
    pdf_path = "results/tvb_pilot.pdf"

results = pd.read_csv(results_path)
results = results[results["method"] == "lacerda"].copy()
validation = pd.read_csv(validation_path)
preflight = pd.read_csv(preflight_path)
interventions = pd.read_csv(interventions_path)

fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.6))

# ---------------------------------------------------------------- (a)
# Default connectome: SCC structure vs thresholding. Descriptive only.
ax = axes[0, 0]
pf = preflight.sort_values("retained_fraction")
ax.plot(pf["retained_fraction"], pf["largest_scc_fraction"],
        color=ROSE_WINE["deep_wine"], marker="o", markersize=8, linewidth=2.4)
ax.plot(pf["retained_fraction"], pf["reciprocity"],
        color=ROSE_WINE["rose"], marker="o", markersize=8, linewidth=2.4)
# Ties in the weight distribution can map several quantiles to the same
# threshold; annotate each distinct point once with the quantile range.
for (x, y), grp in pf.groupby(["retained_fraction", "largest_scc_fraction"]):
    qs = sorted(grp["quantile"])
    label = f"q={qs[0]:g}" if len(qs) == 1 else f"q={qs[0]:g}–{qs[-1]:g}"
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, -18),
                fontsize=10, ha="center", color="0.45")
ax.set_xscale("log")
ax.set_xlabel("retained-edge fraction")
ax.set_ylim(-0.05, 1.05)
ax.axhline(1.0, color="0.85", linewidth=1.0, zorder=0)
ax.set_title("default connectome (descriptive)", fontsize=14)
ax.legend(handles=[
    Line2D([0], [0], color=ROSE_WINE["deep_wine"], lw=2.4, marker="o",
           label="largest-SCC fraction"),
    Line2D([0], [0], color=ROSE_WINE["rose"], lw=2.4, marker="o",
           label="reciprocity"),
], frameon=False, fontsize=12, loc="lower right")

# ---------------------------------------------------------------- (b)
# Controlled benchmark: partition + condensation recovery vs sample size.
ax = axes[0, 1]
for ycol, color, style in [("ari_scc", ROSE_WINE["rose"], "-"),
                           ("fscore", ROSE_WINE["deep_wine"], "-"),
                           ("cdag_fscore", ROSE_WINE["deep_wine"], "--")]:
    sns.lineplot(
        data=results, x="samp_size", y=ycol,
        color=color, estimator="median", errorbar=("ci", 95),
        marker="o", markersize=8, linewidth=2.4, linestyle=style, ax=ax,
    )
ax.set_xscale("log")
ax.set_xlabel(r"$n$")
ax.set_ylabel("")
ax.set_ylim(-0.05, 1.05)
ax.axhline(1.0, color="0.85", linewidth=1.0, zorder=0)
ax.set_title("recovery on TVB equilibria", fontsize=14)
ax.legend(handles=[
    Line2D([0], [0], color=ROSE_WINE["rose"], lw=2.4, marker="o",
           label=r"ARI $\uparrow$ (SCC partition)"),
    Line2D([0], [0], color=ROSE_WINE["deep_wine"], lw=2.4, marker="o",
           label=r"$F_1$ $\uparrow$ (projected cluster DAG)"),
    Line2D([0], [0], color=ROSE_WINE["deep_wine"], lw=2.4, marker="o",
           linestyle="--", label=r"$F_1$ $\uparrow$ (direct C-DAG)"),
], frameon=False, fontsize=12, loc="lower right")

# ---------------------------------------------------------------- (c)
# Exact vs TVB-integrated states: equilibria + intervention responses.
ax = axes[1, 0]
obs_rows = validation[validation["episode_type"] == "observational"]
ivn_rows = validation[validation["episode_type"] == "intervention"]
lo = min(validation["x_exact"].min(), validation["x_tvb"].min())
hi = max(validation["x_exact"].max(), validation["x_tvb"].max())
pad = 0.05 * (hi - lo)
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
        color="0.75", linewidth=1.2, zorder=0)
ax.scatter(obs_rows["x_exact"], obs_rows["x_tvb"], s=22, alpha=0.5,
           color=ROSE_WINE["rose"], edgecolors="none")
ax.scatter(ivn_rows["x_exact"], ivn_rows["x_tvb"], s=42, marker="D",
           color=ROSE_WINE["deep_wine"], edgecolors="none")
ax.set_xlabel(r"exact $x^\ast = e\,(I-W)^{-1}$")
ax.set_ylabel("TVB-integrated")
ax.set_aspect("equal", adjustable="box")
ax.set_title("equilibria and responses (seed 0)", fontsize=14)
ax.legend(handles=[
    Line2D([0], [0], color=ROSE_WINE["rose"], lw=0, marker="o",
           alpha=0.5, label="observational equilibria"),
    Line2D([0], [0], color=ROSE_WINE["deep_wine"], lw=0, marker="D",
           label="intervention responses"),
], frameon=False, fontsize=12, loc="upper left")

# ---------------------------------------------------------------- (d)
# Descendant validation: graph reachability (5 seeds) + TVB-intervention-
# validated C-DAG response predictions (seed 0, CI over interventions).
ax = axes[1, 1]
sns.lineplot(
    data=results, x="samp_size", y="desc_fscore",
    color=ROSE_WINE["deep_wine"], estimator="median", errorbar=("ci", 95),
    marker="o", markersize=8, linewidth=2.4, ax=ax,
)
sns.lineplot(
    data=interventions, x="samp_size", y="ivn_desc_fscore",
    color=ROSE_WINE["rose"], estimator="median", errorbar=("ci", 95),
    marker="D", markersize=8, linewidth=2.4, ax=ax,
)
ax.set_xscale("log")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$F_1$ $\uparrow$ (descendants)")
ax.set_ylim(-0.05, 1.05)
ax.axhline(1.0, color="0.85", linewidth=1.0, zorder=0)
ax.set_title("held-out perturbation validation", fontsize=14)
ax.legend(handles=[
    Line2D([0], [0], color=ROSE_WINE["deep_wine"], lw=2.4, marker="o",
           label="graph reachability (5 seeds)"),
    Line2D([0], [0], color=ROSE_WINE["rose"], lw=2.4, marker="D",
           label="TVB stimulation (seed 0)"),
], frameon=False, fontsize=12, loc="lower right")

fig.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(f"Saved {pdf_path}")

# Console summary of the numerical acceptance criteria (results, not gates).
episode_stats = validation.drop_duplicates(["episode_type", "episode_id"])
n_conv = int(episode_stats["converged"].sum())
print(
    f"validation: {n_conv}/{len(episode_stats)} episodes converged, "
    f"max terminal residual {episode_stats['terminal_residual'].max():.2e}, "
    f"max relative error {episode_stats['rel_err_l2'].max():.2e}"
)
if not bool(np.all(episode_stats["converged"])):
    print("WARNING: some TVB episodes did not converge — see the CSV.")
