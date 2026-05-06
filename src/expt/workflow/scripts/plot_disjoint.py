"""Strict disjoint-cycles micro experiment.

Three methods at d=20, κ=5, every SCC a single Hamilton cycle (no
intra-SCC chord edges) — the regime where Drton et al.'s disjoint-
cycles assumption holds exactly by construction.

Panels:
    1. ARI vs true SCC partition
    2. cluster-DAG F1 (`fscore`)
    3. wall-clock fit time (s, log y)
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

# Rose-to-wine palette anchors (same hex as plot_synth_combined.py).
ROSE_WINE = {
    "pale_rose": "#E9C5C2",
    "rose":      "#AE6B91",
    "deep_wine": "#2C1E3D",
}

# Method → colour mapping.
# - ours         → deep wine  (anchor / focal hue for "our procedure")
# - disjointCycles → rose     (matches its colour in the sister scalability fig)
METHOD_COLOUR = {
    "ours":           ROSE_WINE["deep_wine"],
    "disjointCycles": ROSE_WINE["rose"],
}
METHOD_LABEL = {
    "lacerda":        "ours",
    "disjointcycles": "disjointCycles",
}

try:
    csv_path = snakemake.input[0]   # type: ignore[name-defined]
    pdf_path = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    csv_path = "results/synth_disjoint.csv"
    pdf_path = "results/disjoint_micro.pdf"


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
    & (df["num_cycles"] == 5)
    & (df["d"] == 20)
    & (df["method"].isin(METHOD_LABEL))
].copy()
df["method"] = df["method"].map(METHOD_LABEL)

method_order = ["ours", "disjointCycles"]

PANELS = [
    ("ari_scc",     r"ARI $\uparrow$ — SCC partition", False),
    ("fscore",      r"$F_1$ $\uparrow$ — cluster DAG", False),
    ("runtime_sec", r"fit time (s) $\downarrow$",      True),
]

fig, axes = plt.subplots(1, len(PANELS), figsize=(18.0, 5.2))

for ax, (ycol, ylabel, ylog) in zip(axes, PANELS):
    sns.lineplot(
        data=df, x="samp_size", y=ycol,
        hue="method", hue_order=method_order,
        palette=METHOD_COLOUR,
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

method_handles = [
    Line2D([0], [0], color=METHOD_COLOUR[m], lw=2.4, label=m)
    for m in method_order
]
fig.legend(
    handles=method_handles,
    loc="upper center", bbox_to_anchor=(0.5, 1.04),
    ncol=len(method_handles),
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
