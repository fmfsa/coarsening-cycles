"""Plot the main synthetic experiment — one PDF per (regime, density).

3 × n_cycles grid, single method (`lacerda`):
  - Row 1: ARI(SCC partition) — partition recovery, → 1 with n.
  - Row 2: F (inter-SCC edges) — the identifiable target, → 1 with n
           (Theorem 2 in the paper).
  - Row 3: F (full DAG, var-level) — saturates strictly below 1; the gap
           is the unidentifiable intra-SCC mass and is not a defect of
           the method.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid")

df = pd.concat([pd.read_csv(f) for f in snakemake.input], ignore_index=True)
df = df[df["num_cycles"] >= 1].copy()
density = float(snakemake.wildcards.density)
regime = snakemake.wildcards.regime
df = df[(df["density"] == density) & (df["regime"] == regime)]

num_cycles_vals = sorted(df["num_cycles"].unique())
n_cols = max(len(num_cycles_vals), 1)

ROWS = [
    ("ari_scc",          "ARI (SCC partition)"),
    ("inter_scc_fscore", "F (inter-SCC, identifiable)"),
    ("var_fscore",       "F (full DAG; intra-SCC unidentifiable)"),
]

LINE_COLOR = "tab:red"

fig, axes = plt.subplots(len(ROWS), n_cols, figsize=(4 * n_cols, 8.5), sharey="row")
if n_cols == 1:
    axes = axes.reshape(len(ROWS), 1)

for j, kappa in enumerate(num_cycles_vals):
    sub = df[df["num_cycles"] == kappa]
    for i, (ycol, ylabel) in enumerate(ROWS):
        ax = axes[i, j]
        sns.lineplot(
            data=sub, x="samp_size", y=ycol,
            color=LINE_COLOR,
            marker="o",
            ax=ax, estimator="median", errorbar=("ci", 95),
            linewidth=1.8,
        )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("sample size" if i == len(ROWS) - 1 else "")
        ax.set_ylabel(ylabel if j == 0 else "")
        if i == 0:
            ax.set_title(rf"$\kappa = {kappa}$")

fig.suptitle(
    f"Lacerda recipe — d={int(df['d'].iloc[0])}   density = {density}   regime = {regime}",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(snakemake.output[0], bbox_inches="tight", dpi=150)
plt.close(fig)
