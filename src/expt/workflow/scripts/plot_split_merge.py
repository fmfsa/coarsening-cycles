"""Split/merge failure-mode figure (extends Prop. 2 / App. C.2 with direct metrics).

Left column: stacked bars of run classification (exact / split-only /
merge-only / mixed) vs sample size, per regime.
Right column: median split and merge rates vs sample size, per regime.
First-stable method only (the paper's §5 default), all densities and κ pooled.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_context("paper", font_scale=1.9)
sns.set_style("white")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ERROR_ORDER = ["exact", "split_only", "merge_only", "mixed"]
ERROR_COLOR = {
    "exact": "#4C8577",
    "split_only": "#E9C5C2",
    "merge_only": "#AE6B91",
    "mixed": "#2C1E3D",
}

try:
    csv_path = snakemake.input[0]   # type: ignore[name-defined]
    pdf_path = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    csv_path = "results/synth_results.csv"
    pdf_path = "results/split_merge.pdf"

df = pd.read_csv(csv_path)
df = df[df["method"] == "lacerda"].copy()
df["regime"] = df["regime"].replace({"hard": "stable"})
REGIMES = ["stable", "unstable"]

fig, axes = plt.subplots(len(REGIMES), 2, figsize=(12.5, 8.5))

for i, regime in enumerate(REGIMES):
    sub = df[df["regime"] == regime]

    # Left: stacked run-classification proportions per sample size.
    ax = axes[i, 0]
    comp = (
        sub.groupby("samp_size")["error_type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=ERROR_ORDER, fill_value=0.0)
        .sort_index()
    )
    x = np.arange(len(comp.index))
    bottom = np.zeros(len(comp.index))
    for etype in ERROR_ORDER:
        vals = comp[etype].to_numpy()
        ax.bar(
            x, vals, bottom=bottom, width=0.7,
            color=ERROR_COLOR[etype], label=etype if i == 0 else None,
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n):,}" for n in comp.index], rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"{regime}\nproportion of runs")
    if i == len(REGIMES) - 1:
        ax.set_xlabel("sample size $n$")
    if i == 0:
        ax.legend(frameon=False, fontsize=13, ncol=2, loc="lower left")
        ax.set_title("run classification")

    # Right: median split/merge rates.
    ax = axes[i, 1]
    long = sub.melt(
        id_vars=["samp_size"], value_vars=["split_rate", "merge_rate"],
        var_name="rate", value_name="value",
    ).dropna(subset=["value"])
    sns.lineplot(
        data=long, x="samp_size", y="value", hue="rate",
        palette={"split_rate": "#E9C5C2", "merge_rate": "#AE6B91"},
        estimator="median", errorbar=("ci", 95),
        markers=True, style="rate", dashes=False,
        linewidth=2.4, markersize=8, ax=ax, legend=(i == 0),
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("pair error rate")
    if i == len(REGIMES) - 1:
        ax.set_xlabel("sample size $n$")
    if i == 0:
        ax.legend(frameon=False, fontsize=13, title=None)
        ax.set_title("split vs merge rate")

fig.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {pdf_path}")
