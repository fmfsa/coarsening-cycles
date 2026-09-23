"""Fig. 6-style comparison; missing/timeout accuracy is never plotted as zero."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


df = pd.read_csv(snakemake.input[0])
preview = bool(getattr(snakemake.params, "preview", False))
planned_sizes = list(getattr(snakemake.params, "planned_sizes", []))
# Match the rendered paper Fig. 6: a shared rose, distinguished by line style.
methods = {"hungarian": ("ours", "#AE6B91", "-"),
           "group_lingam": ("GroupLiNGAM", "#AE6B91", "--")}
metrics = [("ari_scc", r"ARI $\uparrow$ — SCC partition"),
           ("oracle_cluster_f1", r"$F_1$ $\uparrow$ — cluster DAG"),
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
summary.to_csv(snakemake.output.summary, index=False)

# Match Fig. 6's typography and layout.
sns.set_context("paper", font_scale=2.3)
sns.set_style("white")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(18, 9.3), sharey="col")
sizes = sorted(df.samp_size.unique())
display_sizes = sorted(set(sizes) | set(planned_sizes))
for i, regime in enumerate(["hard", "unstable"]):
    for j, (metric, title) in enumerate(metrics):
        ax = axes[i, j]
        if planned_sizes:
            boundary = np.sqrt(max(sizes) * min(planned_sizes))
            ax.axvspan(boundary, max(display_sizes)*1.15, color="#F0F0F0", zorder=0)
            ax.text(.79, .48, "Planned\nnot yet run", transform=ax.transAxes,
                    ha="center", va="center", color="#737373", fontsize=10)
        for method, (label, color, style) in methods.items():
            ss = summary[(summary.regime == regime) & (summary.method == method)].sort_values("samp_size")
            ax.plot(ss.samp_size, ss[metric], linestyle=style,
                    color=color, label=label, linewidth=2)
            ax.fill_between(ss.samp_size, ss[metric+"_lo"], ss[metric+"_hi"], color=color, alpha=.16, linewidth=0)
            if j == 2:
                # Individual censoring limits can differ in an interim preview.
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
        if j == 1:
            ax.set_title("stable" if regime == "hard" else "unstable", pad=16)
        ax.set_xlabel(r"sample size $n$")
        ax.set_ylabel(title)
        if planned_sizes:
            ax.set_xticks(display_sizes)
            ax.set_xticklabels([str(n) if n < 1000 else f"{n//1000}k" for n in display_sizes])
            ax.set_xlim(min(display_sizes)/1.15, max(display_sizes)*1.15)

attempted = sorted(summary.attempted.unique())
seed_note = f"{attempted[0]} paired seed{'s' if attempted[0] != 1 else ''}/cell" if len(attempted) == 1 else f"{attempted[0]}–{attempted[-1]} paired seeds/cell; see table"
stage_note = "Runtime feasibility" if str(snakemake.wildcards.profile) in ("feasibility", "large_probe") else "Sample-size comparison"
title = "Layout preview: d = 10, measured results and planned sample sizes" if preview else f"{stage_note}: d = 10, {seed_note}"
handles = [Line2D([0], [0], color=color, linestyle=style, lw=2.4, label=label)
           for label, color, style in methods.values()]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.02),
           ncol=2, frameon=False, fontsize=18, handlelength=2, handletextpad=.5,
           labelspacing=.3, borderpad=.3, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, .97), h_pad=2)
if preview:
    fig.suptitle(title, y=1.08, fontsize=18)
fig.savefig(snakemake.output.pdf, bbox_inches="tight", pad_inches=.04)
plt.close(fig)

report = ["# Sample-size comparison: companion table", "",
          "The two methods receive identical observations per regime, n, and seed. "
          "d=10, 4 non-trivial SCCs, density 0.5, Laplace noise; one numerical thread and one timed fit at a time. "
          "GroupLiNGAM 1.13.0: alpha=0.01, native edge estimation. Ours: Hungarian selection, threshold 0.1. "
          "Fit wall/CPU times exclude worker setup, data loading, and evaluation; they include lazy initialization inside the method call.", "",
          "ARI and F1 summarize completed fits only. F1 projects known predicted edges onto the true partition; "
          "it is an oracle diagnostic, not end-to-end condensation accuracy. Exact recovery requires both variable "
          "memberships and condensation edges. Timeouts are unknown accuracy, and unsuccessful operational exact "
          "recovery within the declared budget. Completed-only summaries may be biased if other fits time out. "
          "Bootstrap intervals are descriptive with few seeds; no interval is estimated from a single seed.", "",
          "| Regime | n | Method | Completed/attempts | Timeouts | Errors | ARI median | Oracle F1 median | Exact within budget/attempts | Wall s median | CPU s median | Budget s |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
def fmt(value):
    return "—" if pd.isna(value) else f"{value:.3g}"
for r in summary.itertuples():
    regime = "stable" if r.regime == "hard" else r.regime
    budget = f"{r.timeout_sec:g}" if r.minimum_timeout_sec == r.timeout_sec else f"{r.minimum_timeout_sec:g}–{r.timeout_sec:g}"
    report.append(f"| {regime} | {r.samp_size} | {methods[r.method][0]} | {r.completed}/{r.attempted} | {r.timed_out} | {r.errors} | {fmt(r.ari_scc)} | {fmt(r.oracle_cluster_f1)} | {r.exact_successes}/{r.attempted} | {fmt(r.fit_runtime_sec)} | {fmt(r.median_fit_cpu_sec)} | {budget} |")
if preview:
    report += ["", "LAYOUT PREVIEW ONLY: three pilot attempts per cell, with the completed "
               "stable seed-0 GroupLiNGAM fit at n=1000 replaced by its longer-budget result. "
               "The original 120-second pilot attempt is retained separately. The other n=1000 "
               "baseline entries remain the original 120-second timeouts. Thus this preview "
               "mixes budgets and has only one completed baseline fit at stable n=1000. "
               "No new n=2000, 5000, or 10000 observations or accuracy results are implied "
               "by the shaded planned region. The interrupted unstable feasibility attempt "
               "is not scored or substituted for its original pilot record."]
report += ["", "Peak RSS and interval endpoints are provided in the companion CSV. "
           "Raw execution logs retain estimator convergence warnings. "
           "This fixed-d experiment does not establish dimension scaling or a universal method ranking.", ""]
Path(snakemake.output.report).write_text("\n".join(report))
