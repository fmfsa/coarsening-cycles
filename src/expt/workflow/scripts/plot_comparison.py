"""Comparison plots: RePaRe-Cycle (IVN) vs OICA.

Produces a 2×3 figure (like Figure 2 of the paper) showing:
  Row 1 — ARI ↑ (each method vs its natural ground truth)
  Row 2 — F-score ↑ (edge recovery)
  Cols   — density = 0.2 | 0.5 | 0.8

Plus a single overlay plot showing both methods on the same axes.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.8)

ivn_df = pd.read_csv(snakemake.input.ivn)
oica_df = pd.read_csv(snakemake.input.oica)

# Each method uses its natural ground truth for ARI:
#   IVN  → ari_ivn  (intervention-determined coarsening)
#   OICA → ari_scc  (SCC / Tarjan coarsening, finest observable)
ivn_df["ari"] = ivn_df["ari_ivn"]
oica_df["ari"] = oica_df["ari_scc"]

num_intervs = ivn_df["num_intervs"].iloc[0]
densities = sorted(ivn_df["density"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: side-by-side 2×3  (density as columns)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")
fig.suptitle(
    f"ER graphs  |  n=10 nodes  |  {num_intervs} interventions  |  10 seeds",
    fontsize=12,
)

for col, density in enumerate(densities):
    ivn_sub = ivn_df[ivn_df.density == density]
    oica_sub = oica_df[oica_df.density == density]

    for row, (metric, ylabel) in enumerate([
        ("ari",    "ARI ↑"),
        ("fscore", "F-score ↑"),
    ]):
        ax = axes[row][col]

        for sub, label, color in [
            (ivn_sub, "RePaRe-Cycle", "tab:blue"),
            (oica_sub, "OICA", "tab:orange"),
        ]:
            agg = sub.groupby("samp_size")[metric].agg(["median", "std"]).reset_index()
            ax.semilogx(agg["samp_size"], agg["median"], "o-", label=label, color=color)
            ax.fill_between(
                agg["samp_size"],
                (agg["median"] - agg["std"]).clip(0, 1),
                (agg["median"] + agg["std"]).clip(0, 1),
                alpha=0.15, color=color,
            )

        ax.set_ylim(0, 1)
        ax.set_xlabel("sample size (n)")
        if col == 0:
            ax.set_ylabel(ylabel)
        if row == 0:
            ax.set_title(f"density = {density}")
        if row == 0 and col == 2:
            ax.legend(fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(snakemake.output.grid, bbox_inches="tight", pad_inches=0.02)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: single ARI overlay  (hue = density, style = method)
# ─────────────────────────────────────────────────────────────────────────────
combined = pd.concat([
    ivn_df[["density", "samp_size", "ari", "fscore"]].assign(method="RePaRe-Cycle"),
    oica_df[["density", "samp_size", "ari", "fscore"]].assign(method="OICA"),
], ignore_index=True)

fig, (ax_ari, ax_fs) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"RePaRe-Cycle vs OICA  |  ER  |  {num_intervs} interventions", fontsize=12)

for ax, metric, ylabel in [
    (ax_ari, "ari",    "ARI ↑"),
    (ax_fs,  "fscore", "F-score ↑"),
]:
    sns.lineplot(
        data=combined, x="samp_size", y=metric,
        hue="density", style="method",
        markers=True, estimator="median", errorbar="ci", ax=ax,
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("sample size (n)")
    ax.set_ylabel(ylabel)
    ax.legend(title="density / method", fontsize=8)

plt.tight_layout()
plt.savefig(snakemake.output.overlay, bbox_inches="tight", pad_inches=0.02)
plt.close()
