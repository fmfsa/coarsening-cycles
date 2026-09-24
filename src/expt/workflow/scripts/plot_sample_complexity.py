"""Empirical $1/n^2$ rate plot — companion to App. A.X.

Two panels:
  Left   E[Hamming(supp(B̂), supp(B))] vs n on log-log. The
         proposition predicts E[Hamming] ≲ d² K / n², so a slope of −2
         is the empirical signature of the rate.
  Right  P[supp(B̂) = supp(B)] vs n. Same data, viewed as exact-recovery.
         Steeper than the Hamming panel because it's a binary indicator.

A reference line of slope −2 is overlaid on the left panel for eyeball
verification, plus a fit slope on the same panel computed via OLS in
log-log over the n range where E[Hamming] is non-zero.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

ROSE_WINE = {
    "rose": "#AE6B91",
    "deep_wine": "#2C1E3D",
    "pale_rose": "#E9C5C2",
}

sns.set_context("paper", font_scale=1.9)
sns.set_style("white")
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


try:
    csv_path = snakemake.input[0]  # type: ignore[name-defined]
    pdf_path = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    csv_path = "results/sample_complexity.csv"
    pdf_path = "results/sample_complexity.pdf"


def _wilson(k, n, alpha=0.05):
    """Wilson 95% CI on a binomial proportion p̂ = k/n."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


df = pd.read_csv(csv_path)

agg = (
    df.groupby("n")
    .agg(
        n_runs=("seed", "size"),
        mean_hamming=("hamming", "mean"),
        sem_hamming=("hamming", lambda s: float(np.std(s, ddof=1) / np.sqrt(len(s)))),
        n_exact_supp=("support_match", "sum"),
    )
    .reset_index()
    .sort_values("n")
)
ci = agg.apply(
    lambda r: _wilson(r["n_runs"] - r["n_exact_supp"], r["n_runs"]),
    axis=1,
    result_type="expand",
).rename(columns={0: "p_err_lo", 1: "p_err_hi"})
agg = pd.concat([agg, ci], axis=1)
agg["p_err"] = 1.0 - agg["n_exact_supp"] / agg["n_runs"]

# ---------------- Plot ----------------
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

# Left: log E[Hamming] vs log n with reference slope −2 and fitted slope.
nz = agg[agg["mean_hamming"] > 0].copy()
ax = axes[0]
# 95% normal-approximation CI using SEM on the mean — for small samples this
# under-covers, but it's the right order-of-magnitude visualisation.
sem_lo = (agg["mean_hamming"] - 1.96 * agg["sem_hamming"]).clip(lower=1e-3)
sem_hi = (agg["mean_hamming"] + 1.96 * agg["sem_hamming"]).clip(lower=1e-3)
ax.fill_between(
    agg["n"], sem_lo, sem_hi, color=ROSE_WINE["rose"], alpha=0.25, linewidth=0
)
ax.plot(
    agg["n"],
    agg["mean_hamming"],
    "-o",
    color=ROSE_WINE["deep_wine"],
    markersize=8,
    linewidth=2.4,
    label=r"empirical $\mathbb{E}[d_H(\hat S, S)]$",
)

# Slope-fit region: drop cells where the failure rate is below 2% (in
# which case the mean Hamming is dominated by 1–5 outlier seeds, not the
# rate) and the leading saturated cells near the d²-d ceiling.
nz_filt = nz.merge(
    agg[["n", "n_exact_supp", "n_runs"]],
    on=["n"],
    how="left",
    suffixes=("", "_dup"),
)
fail_rate = (nz_filt["n_runs"] - nz_filt["n_exact_supp"]) / nz_filt["n_runs"]
nz_filt = nz_filt[fail_rate >= 0.02].copy()
if len(nz_filt) > 0:
    h_max = float(nz_filt["mean_hamming"].max())
    nz_filt = nz_filt[nz_filt["mean_hamming"] < 0.75 * h_max].copy()

# Reference slope of −2 anchored at the slowest non-saturated point.
anchor_df = nz_filt if len(nz_filt) > 0 else nz
if len(anchor_df) > 0:
    n_anchor = anchor_df["n"].iloc[0]
    h_anchor = anchor_df["mean_hamming"].iloc[0]
    n_ref = np.array(agg["n"], dtype=float)
    ax.plot(
        n_ref,
        h_anchor * (n_ref / n_anchor) ** -2,
        color="0.4",
        linestyle=(0, (4, 2)),
        linewidth=1.6,
        label=r"slope $-2$ (Prop. 4)",
    )

if len(nz_filt) >= 2:
    slope, intercept, r_value, _, _ = stats.linregress(
        np.log(nz_filt["n"]),
        np.log(nz_filt["mean_hamming"]),
    )
    ax.text(
        0.05,
        0.06,
        f"fit slope: {slope:.2f}  ($R^2={r_value**2:.2f}$)",
        transform=ax.transAxes,
        fontsize=14,
        color="0.2",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"sample size $n$")
ax.set_ylabel(r"$\mathbb{E}[\,d_H(\mathrm{supp}(\hat B),\, \mathrm{supp}(B))\,]$")
ax.legend(frameon=False, fontsize=14, loc="upper right")

# Right: P[exact support recovery error] vs n with Wilson CI bands.
ax = axes[1]
ax.fill_between(
    agg["n"],
    1.0 - agg["p_err_hi"],
    1.0 - agg["p_err_lo"],
    color=ROSE_WINE["rose"],
    alpha=0.25,
    linewidth=0,
)
ax.plot(
    agg["n"],
    1.0 - agg["p_err"],
    "-o",
    color=ROSE_WINE["deep_wine"],
    markersize=8,
    linewidth=2.4,
)
ax.set_xscale("log")
ax.set_xlabel(r"sample size $n$")
ax.set_ylabel(r"$\Pr[\,\mathrm{supp}(\hat B_n) = \mathrm{supp}(B)\,]$")
ax.set_ylim(-0.05, 1.05)

fig.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(f"Saved {pdf_path}")
