"""Plot saved accuracy and candidate-selection time, Fig. 6 style."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv(snakemake.input[0], low_memory=False)
df["branch"] = df.method.map(lambda m:
    "first-stable" if m == "abl_first_stable" else
    "ours" if m == "abl_hungarian" else
    "random-matching" if m.startswith("abl_randhun_") else None)
df = df[df.branch.notna()].copy()
fields = ["ari_scc", "fscore", "runtime_sec", "ica_runtime_sec", "selection_runtime_sec"]
if df[fields].isna().any().any() or (df.runtime_sec <= 0).any() or (df.selection_runtime_sec <= 0).any():
    raise ValueError("Missing accuracy/timing data or nonpositive fitting time")
if not np.allclose(df.runtime_sec, df.ica_runtime_sec + df.selection_runtime_sec):
    raise ValueError("Saved fit time must equal ICA plus candidate-selection time")
# The five random matchings are repeated choices on the same dataset.
# Average them within each dataset before plotting or estimating uncertainty.
keys = ["regime", "num_cycles", "density", "samp_size", "seed", "branch"]
cells = df.groupby(keys, as_index=False)[fields].mean()
cells.to_csv(snakemake.output.cells, index=False)
summary = cells.groupby(["regime", "samp_size", "branch"])[fields].median().reset_index()
summary.to_csv(snakemake.output.summary, index=False)

sns.set_context("paper", font_scale=2.3)
sns.set_style("white")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
order = ["ours", "first-stable", "random-matching"]
# Fig. 3's three palette anchors, now distinguishing selection strategies.
colors = {"ours": "#2C1E3D", "first-stable": "#AE6B91", "random-matching": "#E9C5C2"}
dashes = {"ours": "", "first-stable": (4, 2), "random-matching": (1, 2)}
panels = [("ari_scc", r"ARI $\uparrow$ — SCC partition"),
          ("fscore", r"$F_1$ $\uparrow$ — cluster DAG"),
          ("selection_runtime_sec", r"selection time (s) $\downarrow$")]
fig, axes = plt.subplots(2, 3, figsize=(18, 9.3), sharey="col")
for i, regime in enumerate(["hard", "unstable"]):
    data = cells[cells.regime == regime]
    for j, (metric, label) in enumerate(panels):
        ax = axes[i, j]
        sns.lineplot(data=data, x="samp_size", y=metric,
                     hue="branch", hue_order=order, style="branch", style_order=order,
                     palette=colors, dashes=dashes, markers=False,
                     estimator="median", errorbar=("ci", 95), seed=0,
                     linewidth=2, ax=ax, legend=False)
        ax.set_xscale("log")
        if j == 2:
            ax.set_yscale("log")
            ax.set_title("log scale", fontsize=16, pad=16)
        else:
            ax.set_ylim(-.05, 1.05)
            ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_xlabel(r"sample size $n$")
        ax.set_ylabel(label)
        if j == 1:
            ax.set_title("stable" if regime == "hard" else "unstable", pad=16)
handles = [Line2D([0], [0], color=colors[name], lw=2.4,
                  linestyle="-" if not dashes[name] else (0, dashes[name]), label=name)
           for name in order]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.02),
           ncol=3, frameon=False, fontsize=18, handlelength=2.5,
           handletextpad=.5, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, .97), h_pad=2)
fig.savefig(snakemake.output.pdf, bbox_inches="tight", pad_inches=.04)
plt.close(fig)

Path(snakemake.output.caption).write_text(
    "Candidate-selection ablation: accuracy and selection time from saved results. "
    "d=10; κ∈{3,4,5}; density∈{0.3,0.5,0.8}; n∈{50,100,500,1000,5000,10000}; "
    "10 seeds per setting; Laplace noise. Top: stable; bottom: unstable. "
    "Solid deep wine: ours. Dashed rose: enumeration with first-stable selection. "
    "Dotted pale rose: random-cost admissible matching (five draws averaged within each dataset). "
    "Ours uses magnitude-based Hungarian matching; only the display label is shortened. "
    "Curves show medians and shaded bands bootstrap 95% intervals across the pooled "
    "κ/density/seed cells, following the existing ablation plot. "
    "Cluster-DAG F₁ is computed using the true partition. The right-hand panels "
    "show selection_runtime_sec on a logarithmic axis, excluding ICA and including enumeration for "
    "first-stable. Shared stages are charged to each strategy as "
    "standalone costs; these rows must not be summed to estimate batch elapsed time. "
    "First-stable enumeration is capped at 10,000 candidates. The cap is reached in "
    "180/180 cells at n=50, 180/180 at n=100, 58/180 at n=500, 5/180 at n=1000, "
    "and 0/180 at n=5000 and 10000. These are capped-enumeration timings, not "
    "unrestricted exhaustive-search timings. "
    "Historical hardware/thread settings and original execution logs are unavailable. "
    "The companion summary CSV also retains ICA and total fit time separately. "
    "No new experimental fits were run.\n")
