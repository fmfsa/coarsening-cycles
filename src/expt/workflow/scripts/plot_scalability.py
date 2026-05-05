"""Scalability figure: ours vs disjointCycles across d ∈ {20, 50, 100, 500}.

3 × len(d) grid. One column per d value; two solid lines per panel:
lacerda (ours) and disjointcycles. Restricted to regime=hard,
density=0.5, num_cycles=10.

Rows:
    1. ARI vs true SCC partition
    2. cluster-DAG F1 (`fscore`)
    3. wall-clock fit time (s, log y)
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(context="paper", style="whitegrid", font_scale=1.6)
plt.rcParams.update({
    "axes.titlesize":  16,
    "axes.labelsize":  14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
})

try:
    csv_path = snakemake.input[0]   # type: ignore[name-defined]
    pdf_path = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    csv_path = "results/synth_scalability.csv"
    pdf_path = "results/scalability_disjointcycles.pdf"

df = pd.read_csv(csv_path)

df = df[
    (df["regime"] == "hard")
    & (df["density"] == 0.5)
    & (df["num_cycles"] == 10)
    & (df["method"].isin(["lacerda", "disjointcycles"]))
].copy()

d_vals = sorted(df["d"].unique())
n_cols = max(len(d_vals), 1)

ROWS = [
    ("ari_scc",     "ARI — SCC partition", False),
    ("fscore",      "F₁ — cluster DAG",    False),
    ("runtime_sec", "fit time (s)",        True),
]

METHOD_COLORS = {
    "lacerda":        "#C0395A",   # rose
    "disjointcycles": "#1F8F73",   # teal
}
METHOD_LABEL = {
    "lacerda":        "ours (ICA-LiNG-D + Tarjan)",
    "disjointcycles": "disjointCycles",
}

fig, axes = plt.subplots(len(ROWS), n_cols, figsize=(4.6 * n_cols, 10), sharey="row")
if n_cols == 1:
    axes = axes.reshape(len(ROWS), 1)

for j, dd in enumerate(d_vals):
    sub = df[df["d"] == dd]
    for i, (ycol, ylabel, ylog) in enumerate(ROWS):
        ax = axes[i, j]
        for method, color in METHOD_COLORS.items():
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            sns.lineplot(
                data=msub, x="samp_size", y=ycol,
                color=color, linestyle="-",
                marker="o", markersize=6,
                ax=ax, estimator="median", errorbar=("ci", 95),
                linewidth=2.4, label=None,
            )
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("sample size" if i == len(ROWS) - 1 else "")
        ax.set_ylabel(ylabel if j == 0 else "")
        if i == 0:
            ax.set_title(rf"$d = {dd}$")

legend_elements = [
    Line2D([0], [0], color=METHOD_COLORS[m], lw=2.4, label=METHOD_LABEL[m])
    for m in ("lacerda", "disjointcycles")
]
if n_cols > 0:
    axes[0, 0].legend(handles=legend_elements, loc="lower right",
                      frameon=True, ncol=1)

fig.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved {pdf_path}")
