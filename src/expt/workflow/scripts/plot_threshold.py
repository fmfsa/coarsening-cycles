"""Threshold-sensitivity figure (App. C.2).

Three panels (one row): ARI of the recovered SCC partition (left),
F1 of the recovered cluster-DAG edges (middle), and the size of the
estimated partition |Π̂| as a sanity diagnostic for the two failure
modes of Prop. 3 (right):

  - very small τ ⇒ |Π̂| collapses to 1: too many spurious edges close
    paths between distinct true SCCs, merging everything (Mode b).
  - very large τ ⇒ |Π̂| grows toward d: true intra-SCC edges drop
    below threshold, splitting the SCC into singletons (Mode a).

The middle of the τ-axis is the safe band where ARI and F1 saturate
at 1; the two extremes are explicitly marked as failure regimes.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

ROSE_WINE = {
    "pale_rose":  "#E9C5C2",
    "rose":       "#AE6B91",
    "deep_wine":  "#2C1E3D",
}
N_PALETTE = {
    500:    ROSE_WINE["pale_rose"],
    5000:   ROSE_WINE["rose"],
    50000:  ROSE_WINE["deep_wine"],
}

sns.set_context("paper", font_scale=1.8)
sns.set_style("white")
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

try:
    csv_path = snakemake.input[0]   # type: ignore[name-defined]
    pdf_path = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    csv_path = "results/synth_threshold.csv"
    pdf_path = "results/synth_threshold.pdf"

df = pd.read_csv(csv_path)
df = df[(df["method"] == "lacerda") & (df["regime"] == "hard")].copy()
df["threshold"] = df["threshold"].astype(float)

samp_sizes = sorted([n for n in df["samp_size"].unique() if n in N_PALETTE])

# Three explicit panels; each gets a distinct y-axis (no sharey).
PANELS = [
    ("ari_scc",       r"ARI $\uparrow$" "\n" r"(SCC partition)",     False, (-0.05, 1.05)),
    ("fscore",        r"$F_1$ $\uparrow$" "\n" r"(cluster-DAG edges)", False, (-0.05, 1.05)),
    ("num_parts_est", r"$|\widehat{\Pi}|$" "\n" r"(number of estimated clusters)", False, None),
]

fig, axes = plt.subplots(1, len(PANELS), figsize=(5.6 * len(PANELS), 4.6))

# True partition has 4 SCCs (κ=4 nontrivial cycles + however many singletons).
# We don't store that exactly, but for d=10, κ=4 with density 0.5 the
# expected ground-truth |Π| is around 5-6 — annotate the band.
for ax, (ycol, ylabel, ylog, ylim) in zip(axes, PANELS):
    for n in samp_sizes:
        sub = df[df["samp_size"] == n]
        sns.lineplot(
            data=sub, x="threshold", y=ycol,
            color=N_PALETTE[n],
            estimator="median", errorbar=("ci", 95),
            marker="o", markersize=8, linewidth=2.4,
            ax=ax,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
        ax.axhline(1.0, color="0.85", linewidth=1.0, zorder=0)
    if ycol == "num_parts_est":
        # |Π| = 1 means everything merged; |Π| = d means all-singletons.
        ax.axhline(1.0,  color="0.85", linewidth=1.0, zorder=0)
        ax.axhline(10.0, color="0.85", linewidth=1.0, zorder=0)
        ax.set_ylim(0.5, 10.5)

handles = [
    Line2D([0], [0], color=N_PALETTE[n], lw=2.4, marker="o", label=rf"$n={n}$")
    for n in samp_sizes
]
fig.legend(handles=handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.04), ncol=len(handles),
           frameon=False, fontsize=15, handlelength=2.0)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(f"Saved {pdf_path}")
