"""Plot multi-SCC experiment results.

Three figures:
1. ARI vs sample size, one panel per n_sccs, hue = scc_size
2. F-score vs sample size, same layout
3. Heatmap: ARI at largest sample size, rows = n_sccs, cols = scc_size
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", palette="colorblind")

df = pd.read_csv(snakemake.input[0])

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: ARI vs sample size, faceted by n_sccs, hue = scc_size
# ─────────────────────────────────────────────────────────────────────────────
n_sccs_vals = sorted(df["n_sccs_param"].unique())
n_panels = len(n_sccs_vals)

fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5), sharey=True)
if n_panels == 1:
    axes = [axes]

for ax, n_sccs in zip(axes, n_sccs_vals):
    sub = df[df["n_sccs_param"] == n_sccs]
    sns.lineplot(
        data=sub, x="samp_size", y="ari_scc",
        hue="scc_size", style="scc_size",
        markers=True, dashes=False, ax=ax,
        estimator="median", errorbar=("ci", 95),
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("sample size (n)")
    ax.set_ylabel("ARI" if ax == axes[0] else "")
    ax.set_title(f"{n_sccs} SCCs")
    ax.legend(title="SCC size", fontsize=7, title_fontsize=8)

fig.suptitle("SCC Partition Recovery (ARI) — Multi-SCC Graphs", y=1.02)
fig.tight_layout()
fig.savefig(snakemake.output.ari, bbox_inches="tight", dpi=150)
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: F-score vs sample size, faceted by n_sccs, hue = scc_size
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5), sharey=True)
if n_panels == 1:
    axes2 = [axes2]

for ax, n_sccs in zip(axes2, n_sccs_vals):
    sub = df[df["n_sccs_param"] == n_sccs]
    sns.lineplot(
        data=sub, x="samp_size", y="fscore",
        hue="scc_size", style="scc_size",
        markers=True, dashes=False, ax=ax,
        estimator="median", errorbar=("ci", 95),
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("sample size (n)")
    ax.set_ylabel("F-score" if ax == axes2[0] else "")
    ax.set_title(f"{n_sccs} SCCs")
    ax.legend(title="SCC size", fontsize=7, title_fontsize=8)

fig2.suptitle("Condensation Edge Recovery (F-score) — Multi-SCC Graphs", y=1.02)
fig2.tight_layout()
fig2.savefig(snakemake.output.fscore, bbox_inches="tight", dpi=150)
plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Heatmap — ARI at largest sample size
# ─────────────────────────────────────────────────────────────────────────────
max_samp = df["samp_size"].max()
sub_max = df[df["samp_size"] == max_samp]
pivot = sub_max.pivot_table(
    values="ari_scc", index="n_sccs_param", columns="scc_size",
    aggfunc="median",
)

fig3, ax3 = plt.subplots(figsize=(5, 3.5))
sns.heatmap(
    pivot, annot=True, fmt=".2f", cmap="YlGn", vmin=0, vmax=1,
    ax=ax3, cbar_kws={"label": "ARI"},
)
ax3.set_xlabel("SCC size")
ax3.set_ylabel("Number of SCCs")
ax3.set_title(f"ARI at n = {max_samp:,} samples")
fig3.tight_layout()
fig3.savefig(snakemake.output.heatmap, bbox_inches="tight", dpi=150)
plt.close(fig3)
