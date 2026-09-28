"""GroupLiNGAM vs ours over sample size: ARI, cluster-DAG F1, and fit time.

One row per regime (stable above unstable by default), up to the largest n
at which GroupLiNGAM was run. Timeouts are never plotted as zero accuracy;
they appear as open triangles at their time limit on the time panel.
"""
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
import seaborn as sns


def median_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    median = float(np.median(values))
    if len(values) < 2:
        return median, np.nan, np.nan
    rng = np.random.default_rng(0)
    resampled = rng.choice(values, size=(4000, len(values)), replace=True)
    lo, hi = np.quantile(np.median(resampled, axis=1), [.025, .975])
    return median, float(lo), float(hi)


params = getattr(snakemake, "params", SimpleNamespace())
df = pd.read_csv(snakemake.input[0])
# Stop at the largest size where both methods were run.
max_n = getattr(params, "max_n", None) or int(df[df.method == "group_lingam"].samp_size.max())
df = df[df.samp_size <= int(max_n)]
regimes = list(getattr(params, "regimes", ["hard", "unstable"]))

# Match the paper's disjoint-cycles figure: ours solid deep wine, baseline dashed rose.
methods = {"hungarian": ("ours", "#2C1E3D", "-"),
           "group_lingam": ("GroupLiNGAM", "#AE6B91", (0, (4, 2)))}
metrics = [("ari_scc", r"ARI $\uparrow$" "\n" r"(SCC partition)"),
           ("oracle_cluster_f1", r"$F_1$ $\uparrow$" "\n" r"(cluster DAG)"),
           ("fit_runtime_sec", r"fit time (s) $\downarrow$")]
rows = []
for (regime, n, method), group in df.groupby(["regime", "samp_size", "method"]):
    ok = group[group.status == "ok"]
    row = dict(regime=regime, samp_size=n, method=method, attempted=len(group),
               completed=len(ok), timed_out=int((group.status == "timeout").sum()),
               errors=int((~group.status.isin(["ok", "timeout"])).sum()),
               exact_successes=int(ok.exact_condensation.sum()),
               timeout_sec=float(group.timeout_sec.max()),
               minimum_timeout_sec=float(group.timeout_sec.min()),
               median_peak_rss_mb=group.peak_rss_mb.median(),
               median_fit_cpu_sec=ok.fit_cpu_sec.median())
    for metric, _ in metrics:
        row[metric], row[metric+"_lo"], row[metric+"_hi"] = median_interval(ok[metric])
    rows.append(row)
summary = pd.DataFrame(rows)

sns.set_context("paper", font_scale=2.3)
sns.set_style("white")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(len(regimes), 3, figsize=(18, 5.2*len(regimes)), sharey="col", squeeze=False)
sizes = sorted(df.samp_size.unique())
for i, regime in enumerate(regimes):
    for j, (metric, title) in enumerate(metrics):
        ax = axes[i, j]
        for method, (label, color, style) in methods.items():
            ss = summary[(summary.regime == regime) & (summary.method == method)].sort_values("samp_size")
            ax.plot(ss.samp_size, ss[metric], linestyle=style,
                    color=color, label=label, linewidth=2)
            ax.fill_between(ss.samp_size, ss[metric+"_lo"], ss[metric+"_hi"], color=color, alpha=.16, linewidth=0)
            if j == 2:
                timed = df[(df.regime == regime) & (df.method == method) & (df.status == "timeout")]
                ax.scatter(timed.samp_size, timed.timeout_sec, edgecolors=color,
                           facecolors="none", marker="^", s=65, zorder=4)
        ax.set_xscale("log")
        if j == 2:
            ax.set_yscale("log")
        else:
            lower = min(-.05, float(df.ari_scc.min())-.05) if j == 0 else -.05
            ax.set_ylim(lower, 1.05)
            ax.set_yticks([0, .25, .5, .75, 1])
        # A single-regime figure drops the row title (the caption names it).
        if j == 1 and len(regimes) > 1:
            ax.set_title("stable" if regime == "hard" else "unstable", pad=16)
        ax.set_xlabel(r"sample size $n$")
        ax.set_ylabel(title)
        # Label exactly the sample sizes that were run.
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) if n < 1000 else f"{n//1000}k" for n in sizes])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(min(sizes)/1.15, max(sizes)*1.15)

handles = [Line2D([0], [0], color=color, linestyle=style, lw=2.4, label=label)
           for label, color, style in methods.values()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.02 if len(regimes) > 1 else 1.04),
           ncol=2, frameon=False, fontsize=18, handlelength=2, handletextpad=.5,
           labelspacing=.3, borderpad=.3, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, .97 if len(regimes) > 1 else .92), h_pad=2)
fig.savefig(snakemake.output[0], bbox_inches="tight", pad_inches=.04)
plt.close(fig)

