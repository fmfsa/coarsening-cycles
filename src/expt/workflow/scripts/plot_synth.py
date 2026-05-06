"""Plot the main synthetic experiment — one PDF per (regime, density).

3 × n_cycles grid, two methods compared:

  ``lacerda``     — stability filter picks the first stable B̂ (ρ < 1).
                    For sparse cyclic SEMs this uniquely recovers the true
                    variable-level graph, so all three metrics → 1 with n.

  ``lacerda_rnd`` — draws uniformly from the *full* distributional equivalence
                    class (stable + unstable members combined). The condensation
                    is invariant across all members (Lacerda Theorem 4), so
                    ARI(SCC) AND cluster-DAG F₁ still → 1; but the variable-level
                    F₁ saturates strictly below 1 — the identifiability gap.

Rows:
  1. ARI (SCC partition)       — identifiable target. → 1 for both methods.
  2. F (cluster-DAG)           — identifiable target. → 1 for both methods,
                                   regardless of which equivalence-class
                                   member is picked.
  3. F (full DAG, var-level)   — ``lacerda`` → 1; ``lacerda_rnd`` saturates
                                   < 1  (the variable-level identifiability
                                   gap).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context="paper", style="whitegrid")

df = pd.concat([pd.read_csv(f) for f in snakemake.input], ignore_index=True)
df = df[df["num_cycles"] >= 1].copy()
density = float(snakemake.wildcards.density)
regime = snakemake.wildcards.regime
df = df[(df["density"] == density) & (df["regime"] == regime)]

num_cycles_vals = sorted(df["num_cycles"].unique())
n_cols = max(len(num_cycles_vals), 1)

ROWS = [
    ("ari_scc",    "ARI — SCC partition\n(identifiable ✓)"),
    ("fscore",     "F₁ — cluster DAG\n(identifiable ✓)"),
    ("var_fscore", "F₁ — full variable-level DAG\n(not identifiable ✗)"),
]

METHOD_STYLE = {
    "lacerda":     dict(color="tab:blue",   label="first-stable (ρ<1 filter)"),
    "lacerda_rnd": dict(color="tab:orange", label="random admissible (gap demo)"),
}

fig, axes = plt.subplots(len(ROWS), n_cols, figsize=(4 * n_cols, 8.5), sharey="row")
if n_cols == 1:
    axes = axes.reshape(len(ROWS), 1)

for j, kappa in enumerate(num_cycles_vals):
    sub = df[df["num_cycles"] == kappa]
    for i, (ycol, ylabel) in enumerate(ROWS):
        ax = axes[i, j]
        for method, style in METHOD_STYLE.items():
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            sns.lineplot(
                data=msub, x="samp_size", y=ycol,
                color=style["color"],
                label=style["label"] if (i == 0 and j == 0) else None,
                marker="o",
                ax=ax, estimator="median", errorbar=("ci", 95),
                linewidth=1.8,
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("sample size" if i == len(ROWS) - 1 else "")
        ax.set_ylabel(ylabel if j == 0 else "")
        if i == 0:
            ax.set_title(rf"$\kappa = {kappa}$")

# Legend in the top-left panel only
if n_cols > 0:
    axes[0, 0].legend(fontsize=7, loc="lower right", frameon=True)

fig.tight_layout()
fig.savefig(snakemake.output[0], bbox_inches="tight", dpi=150)
plt.close(fig)
