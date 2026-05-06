"""Combined synthetic figure: stable vs unstable regime, all densities, first-stable only.

3 × κ grid. Each panel shows the (density × regime) cross product.
Colors drive density, linestyle + marker drive regime — both auto-assigned
by seaborn from the "colorblind" palette and default style cycle.

Key visual: in row 3 (variable-level F₁), stable lines → 1 but unstable saturate below 1.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROSE_WINE = {
    "pale_rose":  "#E9C5C2",   # sparsest / lightest end
    "rose":       "#AE6B91",   # mid
    "deep_wine":  "#2C1E3D",   # densest / near-black
}

sns.set_palette([ROSE_WINE["pale_rose"], ROSE_WINE["rose"], ROSE_WINE["deep_wine"]])
sns.set_context("paper", font_scale=2.3)
sns.set_style("white")
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Density encoded by lightness (sparser = lighter), regime by linestyle.
DENSITY_COLOR = {
    0.3: ROSE_WINE["pale_rose"],
    0.5: ROSE_WINE["rose"],
    0.8: ROSE_WINE["deep_wine"],
}
REGIME_DASH = {
    "stable":   (1, 0),       # solid
    "unstable": (4, 2),       # dashed
}

try:
    csv_path = snakemake.input[0]   # type: ignore[name-defined]
    pdf_path = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    csv_path = "results/synth_results.csv"
    pdf_path = "results/synth_main_combined.pdf"

df = pd.read_csv(csv_path)
df = df[(df["method"] == "lacerda") & (df["num_cycles"] >= 1)].copy()

# Display label only — internal "hard" stays the same on disk and in metrics.csv;
# we just rename for the figure legend so it reads as "stable vs unstable".
df["regime"] = df["regime"].replace({"hard": "stable"})

num_cycles_vals = sorted(df["num_cycles"].unique())
n_cols = max(len(num_cycles_vals), 1)

ROWS = [
    ("ari_scc",    "ARI ↑ — SCC partition"),
    ("fscore",     "F₁ ↑ — cluster DAG"),
    ("var_fscore", "F₁ ↑ — variable-level DAG"),
]

fig, axes = plt.subplots(len(ROWS), n_cols, figsize=(5.4 * n_cols, 12), sharey="row")
if n_cols == 1:
    axes = axes.reshape(len(ROWS), 1)

for j, kappa in enumerate(num_cycles_vals):
    sub = df[df["num_cycles"] == kappa]
    for i, (ycol, ylabel) in enumerate(ROWS):
        ax = axes[i, j]
        sns.lineplot(
            data=sub, x="samp_size", y=ycol,
            hue="density", style="regime",
            palette=DENSITY_COLOR,
            dashes=REGIME_DASH,
            markers=True,
            estimator="median", errorbar=("ci", 95),
            linewidth=2.4, markersize=8,
            ax=ax, legend=(i == 0 and j == 0),
        )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("sample size $n$" if i == len(ROWS) - 1 else "")
        ax.set_ylabel(ylabel if j == 0 else "")
        if i == 0:
            ax.set_title(rf"$\kappa = {kappa}$")

# Reposition the single combined legend (only drawn once on the top-left panel).
# Shrink it: the global font_scale=2.3 inflates everything, so we override
# the legend to a more compact footprint.
leg = axes[0, 0].get_legend()
if leg is not None:
    leg.set_frame_on(False)
    axes[0, 0].legend(
        leg.legend_handles, [t.get_text() for t in leg.get_texts()],
        loc="lower right", frameon=False, ncol=1,
        fontsize=18,           # was "small" (≈ font_scale × default = ~21pt); now fixed
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.3,
        borderpad=0.3,
        title=None,
    )

fig.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {pdf_path}")
