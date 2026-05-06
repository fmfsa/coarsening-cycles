"""Partial-data scalability plotter (1x3 layout).

Standalone helper for iterating on the scalability figure while the
snakemake sweep is still running. Globs every `metrics.csv` under
`results/synth/regime=hard/.../num_cycles=10/density=0.5/...` that exists
on disk right now, concatenates them, and renders a compact 3-panel
figure: one panel per metric (ARI, cluster-DAG F1, fit time), with
``d`` encoded as colour intensity and method as linestyle (solid =
ours, dashed = disjointCycles).

Run directly (not via snakemake)::

    cd src/expt/workflow
    python scripts/plot_scalability_partial.py            # → results/scalability_disjointcycles_partial.pdf
    python scripts/plot_scalability_partial.py path.pdf   # custom output

The figure auto-shrinks to whichever (d, method, n) cells have produced
metrics so far - empty cells are silently dropped.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
WORKFLOW_ROOT = HERE.parent
REPO_ROOT = WORKFLOW_ROOT.parent.parent.parent
RESULTS_ROOT = WORKFLOW_ROOT / "results" / "synth"

DEFAULT_OUT = WORKFLOW_ROOT / "results" / "scalability_disjointcycles_partial.pdf"
out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT


# Three-step rose-to-wine ramp keyed by d. Hand-picked to read clearly
# alongside the main results figure (which uses density as a similar ramp).
D_PALETTE = {
    20:  "#D9A6BB",   # light rose
    50:  "#AE6B91",   # rose
    100: "#5C2E48",   # deep wine
}

METHOD_DASH = {
    "ours":           "",        # solid
    "disjointCycles": (4, 2),    # dashed
}
METHOD_LABEL = {"lacerda": "ours", "disjointcycles": "disjointCycles"}


def _collect_metrics() -> pd.DataFrame:
    """Concatenate every metrics.csv on disk under the scalability slice."""
    pattern = str(
        RESULTS_ROOT
        / "regime=hard" / "d=*" / "num_cycles=10" / "density=0.5"
        / "samp_size=*" / "seed=*" / "method=*" / "metrics.csv"
    )
    paths = glob.glob(pattern)
    if not paths:
        sys.exit(
            f"No metrics.csv files found under {RESULTS_ROOT}. "
            "Has the sweep produced anything yet?"
        )

    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:
            continue  # in-flight write
    if not frames:
        sys.exit("Found metrics.csv files but all were empty/in-flight.")

    df = pd.concat(frames, ignore_index=True)
    return df[df["method"].isin(["lacerda", "disjointcycles"])].copy()


def _render(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_context("paper", font_scale=2.3)
    sns.set_style("white")
    plt.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    df = df.copy()
    df["method"] = df["method"].map(METHOD_LABEL)
    df = df[df["d"].isin(D_PALETTE)]
    method_order = ["ours", "disjointCycles"]
    d_order = sorted(D_PALETTE.keys())

    # Keep only sample sizes that every plotted d has, so all lines share
    # an x-range. Cells with n < d are skipped upstream (ICA-ill-posed).
    _n_per_d = {dd: set(df.loc[df["d"] == dd, "samp_size"].unique())
                for dd in d_order}
    _common_n = set.intersection(*_n_per_d.values()) if _n_per_d else set()
    df = df[df["samp_size"].isin(_common_n)]

    PANELS = [
        ("ari_scc",     r"ARI $\uparrow$ — SCC partition", False),
        ("fscore",      r"$F_1$ $\uparrow$ — cluster DAG", False),
        ("runtime_sec", r"fit time (s) $\downarrow$",      True),
    ]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(18.0, 5.2))

    for ax, (ycol, ylabel, ylog) in zip(axes, PANELS):
        sns.lineplot(
            data=df, x="samp_size", y=ycol,
            hue="d", style="method",
            hue_order=d_order, style_order=method_order,
            palette=D_PALETTE,
            dashes={"ours": "", "disjointCycles": (4, 2)},
            markers=False,
            estimator="median", errorbar=("ci", 95),
            linewidth=2.0,
            ax=ax, legend=False,
        )
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r"sample size $n$")
        ax.set_ylabel(ylabel)

    # Single shared legend split into two groups: d (colour) and method (style).
    d_handles = [
        Line2D([0], [0], color=D_PALETTE[d], lw=2.4, label=rf"$d={d}$")
        for d in d_order
    ]
    method_handles = [
        Line2D([0], [0], color="0.25", lw=2.0,
               linestyle="-",  label="ours"),
        Line2D([0], [0], color="0.25", lw=2.0,
               linestyle=(0, (4, 2)), label="disjointCycles"),
    ]
    fig.legend(
        handles=d_handles + method_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=len(d_handles) + len(method_handles),
        frameon=False,
        fontsize=18,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.3,
        borderpad=0.3,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _coverage_summary(df: pd.DataFrame) -> str:
    """Print a cells-completed-per-method summary so you know what's missing."""
    lines = []
    for method in sorted(df["method"].unique()):
        msub = df[df["method"] == method]
        for dd in sorted(msub["d"].unique()):
            dsub = msub[msub["d"] == dd]
            counts_by_n = dsub.groupby("samp_size")["seed"].nunique().to_dict()
            cell_str = ", ".join(f"n={n}: {c}/10" for n, c in sorted(counts_by_n.items()))
            lines.append(f"  {method:<16s} d={dd:<3d}  {cell_str}")
    return "\n".join(lines)


def main() -> None:
    df = _collect_metrics()
    print(f"Collected {len(df)} rows from disk; coverage:")
    print(_coverage_summary(df))
    _render(df, out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
