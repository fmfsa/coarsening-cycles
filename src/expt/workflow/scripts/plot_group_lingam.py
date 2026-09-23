"""Accuracy, completion and censored runtime; never hide unsuccessful attempts."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

df = pd.read_csv(snakemake.input[0])
colors = {"hungarian": "#2C1E3D", "group_lingam": "#AE6B91"}
labels = {"hungarian": "Ours (Hungarian)", "group_lingam": "GroupLiNGAM"}
rows = []
for (regime, n, method), group in df.groupby(["regime", "samp_size", "method"]):
    ok = group[group.status == "ok"]
    k, count = int(group.exact_condensation.sum()), len(group)
    z = norm.ppf(.975)
    p = k/count
    center = (p+z*z/(2*count))/(1+z*z/count)
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    rows.append(dict(regime=regime, samp_size=n, method=method, attempted=count,
                     completed=len(ok), timed_out=int((group.status == "timeout").sum()),
                     errors=int((~group.status.isin(["ok", "timeout"])).sum()),
                     median_ari=ok.ari_scc.median() if len(ok) else np.nan,
                     median_oracle_f1=ok.oracle_cluster_f1.median() if len(ok) else np.nan,
                     exact_successes=k, exact_rate=p, exact_lo=max(0, center-half), exact_hi=min(1, center+half),
                     median_fit_sec=ok.fit_runtime_sec.median() if len(ok) else np.nan,
                     median_fit_cpu_sec=ok.fit_cpu_sec.median() if len(ok) else np.nan,
                     median_peak_rss_mb=group.peak_rss_mb.median()))
summary = pd.DataFrame(rows)
summary.to_csv(snakemake.output.summary, index=False)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
panels = [("median_ari", "Partition ARI (completed fits)"),
          ("median_oracle_f1", r"True-partition-projected $F_1$ (completed)"),
          ("exact_rate", "Exact within budget / all attempts"),
          ("median_fit_sec", "Fit wall time (s); triangles = timeouts")]
for i, regime in enumerate(["hard", "unstable"]):
    for j, (col, title) in enumerate(panels):
        ax = axes[i, j]
        for method in colors:
            ss = summary[(summary.regime == regime) & (summary.method == method)].sort_values("samp_size")
            ax.plot(ss.samp_size, ss[col], marker="o", linestyle="--" if method == "group_lingam" else "-",
                    color=colors[method], label=labels[method], markersize=4)
            raw = df[(df.regime == regime) & (df.method == method)]
            if j in (0, 1):
                ok = raw[raw.status == "ok"]
                metric = "ari_scc" if j == 0 else "oracle_cluster_f1"
                ax.scatter(ok.samp_size, ok[metric], color=colors[method], alpha=.4, s=12)
            elif j == 2:
                ax.fill_between(ss.samp_size, ss.exact_lo, ss.exact_hi, color=colors[method], alpha=.12)
            elif j == 3:
                ok = raw[raw.status == "ok"]
                ax.scatter(ok.samp_size, ok.fit_runtime_sec, color=colors[method], alpha=.4, s=12)
                timed = raw[raw.status == "timeout"]
                ax.scatter(timed.samp_size, timed.timeout_sec, color=colors[method], marker="^", s=45)
        ax.set_title(title, fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("Sample size n")
        if j == 3:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-.1 if j == 0 else -.03, 1.05)
        if j == 0:
            ax.set_ylabel("Stable" if regime == "hard" else "Unstable")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
fig.suptitle("Paired d=10 comparison: exploratory pilot (3 seeds/cell)" if str(snakemake.wildcards.profile) == "pilot" else "Paired d=10 comparison (10 seeds/cell)", y=.94)
fig.tight_layout(rect=(0, .04, 1, .90))
fig.text(.5, .015, "Accuracy medians use completed fits; exact-within-budget uses all attempts. Bands: Wilson 95% intervals. Timeout runtime is a lower bound.", ha="center", fontsize=9)
fig.savefig(snakemake.output.pdf, bbox_inches="tight")
plt.close(fig)

report = ["# GroupLiNGAM comparison", "",
          "Same observations per method; d=10, 4 non-trivial SCCs, density 0.5, Laplace noise. "
          "One numerical thread; timed fits run sequentially. GroupLiNGAM 1.13.0 uses default HSIC alpha=0.01 "
          "and native adaptive-lasso edge estimation. Ours uses Hungarian selection and threshold 0.1.", "",
          "Timing excludes worker setup and data loading, includes each method's native structure estimation "
          "and any lazy initialization inside the timed call, and excludes evaluation. "
          "Sampled peak RSS (50 ms polling) includes the worker's imported libraries. "
          "Timeouts have no accuracy estimate and count as unsuccessful exact recovery within budget. "
          "Completed-fit accuracy and runtime can be selection-biased when other fits time out.", "",
          "True-partition-projected F1 is an oracle diagnostic: unknown within-estimated-group edges are "
          "excluded. Exact recovery requires the correct variable memberships AND condensation edges. "
          "No variable-level graph recovery is claimed for GroupLiNGAM.", "",
          "| Regime | n | Method | Completed / attempts | Timeouts | ARI median | Oracle F1 median | Exact within budget / attempts | Fit seconds median |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
def fmt(x):
    return "—" if pd.isna(x) else f"{x:.3g}"
for row in summary.itertuples():
    regime_label = "stable" if row.regime == "hard" else "unstable"
    report.append(f"| {regime_label} | {row.samp_size} | {labels[row.method]} | {row.completed}/{row.attempted} | {row.timed_out} | {fmt(row.median_ari)} | {fmt(row.median_oracle_f1)} | {row.exact_successes}/{row.attempted} | {fmt(row.median_fit_sec)} |")
report += ["", "Pilot results are exploratory, not publication evidence. Runtime at fixed d=10 does not establish dimension scaling. "
           "Do not interpret an unsuccessful fit as a statistical failure of the population identification result.", ""]
Path(snakemake.output.report).write_text("\n".join(report))
