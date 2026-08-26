"""Direct split/merge rates across the threshold sweep (App. C.2 extension).

Shows where Prop. 2's two failure modes cross over as τ grows: merges
(spurious surviving entries fuse SCCs) dominate at small τ, splits (dropped
true intra-SCC edges break reachability) at large τ. Median pair-error rates
per (τ, n); hard regime, d=10, κ=4, λ=0.5 (the App. C.2 sweep grid).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("paper", font_scale=1.9)
sns.set_style("white")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
})

try:
    csv_path = snakemake.input[0]   # type: ignore[name-defined]
    pdf_path = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    csv_path = "results/synth_threshold.csv"
    pdf_path = "results/threshold_split_merge.pdf"

df = pd.read_csv(csv_path)
samp_sizes = sorted(df["samp_size"].unique())

fig, axes = plt.subplots(1, len(samp_sizes), figsize=(5.2 * len(samp_sizes), 4.2), sharey=True)

for j, n in enumerate(samp_sizes):
    ax = axes[j]
    sub = df[df["samp_size"] == n]
    long = sub.melt(
        id_vars=["threshold"], value_vars=["split_rate", "merge_rate"],
        var_name="rate", value_name="value",
    ).dropna(subset=["value"])
    sns.lineplot(
        data=long, x="threshold", y="value", hue="rate", style="rate",
        palette={"split_rate": "#4C8577", "merge_rate": "#AE6B91"},
        dashes=False, markers=True,
        estimator="median", errorbar=("ci", 95),
        linewidth=2.4, markersize=8, ax=ax, legend=(j == 0),
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"threshold $\tau$")
    ax.set_ylabel("pair error rate" if j == 0 else "")
    ax.set_title(f"$n = {n:,}$")
    if j == 0:
        ax.legend(frameon=False, fontsize=13, title=None)

fig.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {pdf_path}")
