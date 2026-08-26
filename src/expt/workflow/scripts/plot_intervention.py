"""Whole-SCC hard-intervention figure + LaTeX-ready summary table (§5.5).

Panel A: normalized block-effect error of Δμ vs n — oracle condensation vs
end-to-end recovered condensation (Hungarian; conditional on structural
validity), per regime.
Panel B: end-to-end structural-recovery rate vs n (strict: exact partition AND
exact target→outcome ancestral condensation edges). This is a recovery rate,
not a "success" rate — no effect-error threshold is folded into it.
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

ARM_COLOR = {"oracle": "#4C8577", "hungarian": "#2C1E3D"}
REGIME_DASH = {"stable": (1, 0), "unstable": (4, 2)}

try:
    csv_path = snakemake.input[0]      # type: ignore[name-defined]
    pdf_path = snakemake.output.pdf    # type: ignore[name-defined]
    table_path = snakemake.output.table  # type: ignore[name-defined]
except NameError:
    csv_path = "results/intervention_effects.csv"
    pdf_path = "results/intervention_effects.pdf"
    table_path = "results/intervention_table.txt"

df = pd.read_csv(csv_path)
df = df[df.get("skipped", 0) == 0].copy()
df["regime"] = df["regime"].replace({"hard": "stable"})

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# ── Panel A: conditional effect error ────────────────────────────────────
# Aggregate WITHIN each dataset first (median over its downstream blocks),
# so datasets — not outcome blocks — are the unit of replication for the
# cross-dataset median and CI.
eff = df[df["arm"].isin(["oracle", "hungarian"])].copy()
eff = eff[eff["structurally_valid"] == True]  # noqa: E712 — CSV bools
eff = (
    eff.groupby(["regime", "arm", "samp_size", "seed"], as_index=False)
    ["norm_err"].median()
)
sns.lineplot(
    data=eff, x="samp_size", y="norm_err",
    hue="arm", style="regime",
    palette=ARM_COLOR, dashes=REGIME_DASH, markers=True,
    estimator="median", errorbar=("ci", 95),
    linewidth=2.2, markersize=7, ax=axes[0],
)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("sample size $n$")
axes[0].set_ylabel("normalized block-effect error")
axes[0].set_title("effect error (structurally exact runs)")
axes[0].legend(frameon=False, fontsize=11)

# ── Panel B: end-to-end structural-recovery rate (per dataset) ───────────
# Validity is constant across a dataset's rows (full-DAG criterion), so
# collapse to one value per dataset before averaging across datasets.
succ = (
    df[df["arm"] == "hungarian"]
    .groupby(["regime", "samp_size", "seed"], as_index=False)
    ["structurally_valid"].all()
    .groupby(["regime", "samp_size"], as_index=False)["structurally_valid"]
    .mean()
)
sns.lineplot(
    data=succ, x="samp_size", y="structurally_valid",
    hue="regime", style="regime",
    palette={"stable": "#4C8577", "unstable": "#AE6B91"},
    dashes=REGIME_DASH, markers=True,
    linewidth=2.2, markersize=7, ax=axes[1],
)
axes[1].set_xscale("log")
axes[1].set_ylim(-0.05, 1.05)
axes[1].set_xlabel("sample size $n$")
axes[1].set_ylabel("structural-recovery rate")
axes[1].set_title("end-to-end recovery (strict)")
axes[1].legend(frameon=False, fontsize=11)

fig.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {pdf_path}")

# ── Summary table (median ± IQR at the largest n per regime/arm) ─────────
lines = []
n_max = int(df["samp_size"].max())
lines.append(f"Whole-SCC hard-intervention summary at n={n_max:,}")
lines.append("=" * 64)
for regime in ["stable", "unstable"]:
    for arm in ["oracle", "hungarian", "first_stable"]:
        sub = df[
            (df["regime"] == regime) & (df["arm"] == arm)
            & (df["samp_size"] == n_max)
        ]
        # Dataset-level: within-dataset median first, then across datasets.
        val = (
            sub[sub["structurally_valid"] == True]  # noqa: E712
            .groupby("seed")["norm_err"].median()
        )
        rate = sub.groupby("seed")["structurally_valid"].all().mean() if len(sub) else float("nan")
        med = val.median() if len(val) else float("nan")
        q1, q3 = (val.quantile(0.25), val.quantile(0.75)) if len(val) else (float("nan"),) * 2
        lines.append(
            f"{regime:9s} {arm:13s} norm_err median {med:7.4f} "
            f"[IQR {q1:7.4f}, {q3:7.4f}]   run-level structural recovery {rate:5.1%}"
        )
with open(table_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved {table_path}")
