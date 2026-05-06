"""Scalability figure: ours vs disjointCycles across d ∈ {20, 50, 100, ...}.

1×3 grid (one panel per metric). ``d`` is encoded as colour intensity
(light rose → deep wine) and method as linestyle (solid = ours, dashed =
disjointCycles). Restricted to regime=hard, density=0.5, num_cycles=10.

Panels:
    1. ARI vs true SCC partition
    2. cluster-DAG F1 (`fscore`)
    3. wall-clock fit time (s, log y)
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

D_PALETTE = {
    20:  "#D9A6BB",   # light rose
    50:  "#AE6B91",   # rose (focal hue)
    100: "#5C2E48",   # deep wine
}

METHOD_LABEL = {"lacerda": "ours", "disjointcycles": "disjointCycles"}

try:
    csv_path = snakemake.input[0]   # type: ignore[name-defined]
    pdf_path = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    csv_path = "results/synth_scalability.csv"
    pdf_path = "results/scalability_disjointcycles.pdf"


sns.set_context("paper", font_scale=2.3)
sns.set_style("white")
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

df = pd.read_csv(csv_path)
df = df[
    (df["regime"] == "hard")
    & (df["density"] == 0.5)
    & (df["num_cycles"] == 10)
    & (df["method"].isin(["lacerda", "disjointcycles"]))
].copy()
df["method"] = df["method"].map(METHOD_LABEL)
df = df[df["d"].isin(D_PALETTE)]

# Restrict to sample sizes that ALL plotted d values have data for, so every
# line starts at the same x. Cells with n < d are mathematically ill-posed
# for ICA-LiNG-D (FastICA caps n_components at min(n, d), so the unmixing
# matrix isn't square) and are skipped at the snakemake level — this filter
# just keeps the figure visually consistent. Currently drops n = 50
# (missing for d = 100).
_n_per_d = {dd: set(df.loc[df["d"] == dd, "samp_size"].unique())
            for dd in sorted(D_PALETTE)}
_common_n = set.intersection(*_n_per_d.values()) if _n_per_d else set()
df = df[df["samp_size"].isin(_common_n)]

method_order = ["ours", "disjointCycles"]
d_order = sorted(D_PALETTE.keys())

PANELS = [
    ("ari_scc",     r"ARI $\uparrow$ — SCC partition", False),
    ("fscore",      r"$F_1$ $\uparrow$ — cluster DAG", False),
    ("runtime_sec", r"fit time (s) $\downarrow$",      True),
]

fig, axes = plt.subplots(1, len(PANELS), figsize=(18.0, 5.2))

for ax, (ycol, ylabel, ylog) in zip(axes, PANELS):
    sns.lineplot(
        data=df, x="samp_size", y=ycol,
        hue="d", style="method",
        hue_order=d_order, style_order=method_order,
        palette=D_PALETTE,
        dashes={"ours": "", "disjointCycles": (4, 2)},
        markers=False,
        estimator="median", errorbar=("ci", 95),
        linewidth=2.0,
        ax=ax, legend=False,
    )
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    else:
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"sample size $n$")
    ax.set_ylabel(ylabel)

# Single shared legend split into two groups: d (colour) and method (style).
d_handles = [
    Line2D([0], [0], color=D_PALETTE[d], lw=2.4, label=rf"$d={d}$")
    for d in d_order
]
method_handles = [
    Line2D([0], [0], color="0.25", lw=2.0,
           linestyle="-",  label="ours"),
    Line2D([0], [0], color="0.25", lw=2.0,
           linestyle=(0, (4, 2)), label="disjointCycles"),
]
fig.legend(
    handles=d_handles + method_handles,
    loc="upper center", bbox_to_anchor=(0.5, 1.04),
    ncol=len(d_handles) + len(method_handles),
    frameon=False,
    fontsize=18,
    handlelength=2.0,
    handletextpad=0.5,
    labelspacing=0.3,
    borderpad=0.3,
    columnspacing=1.2,
)

fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(f"Saved {pdf_path}")
