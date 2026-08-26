"""Candidate-selection ablation figure (§5.4).

Rows: ARI (SCC partition), cluster-DAG F₁, variable-level F₁.
Columns: regime (stable | unstable).
Hue/style: selection branch — first-stable (enumeration baseline), Hungarian
(deterministic, no enumeration), random-matching (randomised admissible
representative via random-cost matching, no enumeration; 5 draws per dataset,
AVERAGED WITHIN-CELL first so cells stay the unit of replication).

All branches of a cell share one FastICA estimate, so differences isolate
candidate selection. Density and κ are pooled (medians across the grid).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BRANCH_COLOR = {
    "first-stable": "#2C1E3D",
    "random-matching": "#AE6B91",
    "hungarian": "#4C8577",
}
BRANCH_DASH = {
    "first-stable": (1, 0),
    "random-matching": (4, 2),
    "hungarian": (1, 1),
}

sns.set_context("paper", font_scale=2.0)
sns.set_style("white")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
})

try:
    csv_path = snakemake.input[0]   # type: ignore[name-defined]
    pdf_path = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    csv_path = "results/synth_ablation.csv"
    pdf_path = "results/ablation_selection.pdf"

df = pd.read_csv(csv_path)

df["branch"] = df["method"].map(
    lambda m: "first-stable" if m == "abl_first_stable"
    else "hungarian" if m == "abl_hungarian"
    else "random-matching" if m.startswith("abl_randhun")
    else "random-enumerated"
)
# The enumeration+random-pick branches are not part of the reported ablation.
df = df[df["branch"] != "random-enumerated"]
df["regime"] = df["regime"].replace({"hard": "stable"})

# The 5 random draws share one dataset/W: average within-cell first so each
# cell contributes one observation per branch (no pseudo-replication).
CELL = ["regime", "num_cycles", "density", "samp_size", "seed", "branch"]
num_cols = ["ari_scc", "fscore", "var_fscore", "split_rate", "merge_rate"]
cells = df.groupby(CELL, as_index=False)[num_cols].mean()

ROWS = [
    ("ari_scc", "ARI ↑ — SCC partition"),
    ("fscore", "F₁ ↑ — cluster DAG"),
    ("var_fscore", "F₁ ↑ — variable-level DAG"),
]
REGIMES = ["stable", "unstable"]

fig, axes = plt.subplots(
    len(ROWS), len(REGIMES), figsize=(5.6 * len(REGIMES), 11), sharey="row"
)

for j, regime in enumerate(REGIMES):
    sub = cells[cells["regime"] == regime]
    for i, (ycol, ylabel) in enumerate(ROWS):
        ax = axes[i, j]
        sns.lineplot(
            data=sub, x="samp_size", y=ycol,
            hue="branch", style="branch",
            palette=BRANCH_COLOR, dashes=BRANCH_DASH, markers=True,
            estimator="median", errorbar=("ci", 95),
            linewidth=2.4, markersize=8,
            ax=ax, legend=(i == 0 and j == 0),
        )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("sample size $n$" if i == len(ROWS) - 1 else "")
        ax.set_ylabel(ylabel if j == 0 else "")
        if i == 0:
            ax.set_title(regime)

leg = axes[0, 0].get_legend()
if leg is not None:
    axes[0, 0].legend(
        leg.legend_handles, [t.get_text() for t in leg.get_texts()],
        loc="lower right", frameon=False, fontsize=15,
        handlelength=2.0, handletextpad=0.5, labelspacing=0.3, borderpad=0.3,
        title=None,
    )

fig.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {pdf_path}")
