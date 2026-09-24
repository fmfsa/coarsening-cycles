"""Experiment 2 figures (Reduced Wong-Wang).

Produces two focused figures:

  * tvb_rww.pdf: recovery against known ground truth, with SCC ARI and
    direct C-DAG F1 versus sample size for each input amplitude.
  * tvb_rww_topology.pdf: a large true-versus-recovered C-DAG comparison
    for seed 0, medium amplitude, n=5000.

The linear comparison and held-out stimulation diagnostic are intentionally
excluded: the controlled benchmark has known SCC and C-DAG ground truth, so
the recovery metrics are the clearest primary result.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from _graph_draw import draw_cdag, layered_positions, scc_coloring
from repare_cycle.tvb_bridge import condensation_from_adjacency

ROSE_WINE = {
    "pale_rose":  "#E9C5C2",
    "rose":       "#AE6B91",
    "deep_wine":  "#2C1E3D",
}
AMPS = ["small", "medium", "large"]
AMP_COLOR = dict(zip(AMPS, sns.color_palette("flare", len(AMPS))))
AMP_LABEL = {
    "small": "small (0.0005 nA)",
    "medium": "medium (0.002 nA)",
    "large": "large (0.005 nA)",
}
SAMPLE_SIZES = [100, 500, 2_000, 5_000]

sns.set_context("paper", font_scale=1.8)
sns.set_style("white")
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

try:
    rww_results_path = snakemake.input.results          # type: ignore[name-defined]
    data_path = snakemake.input.data                    # type: ignore[name-defined]
    model_path = snakemake.input.model                  # type: ignore[name-defined]
    recovery_pdf_path = snakemake.output.recovery       # type: ignore[name-defined]
    topology_pdf_path = snakemake.output.topology       # type: ignore[name-defined]
    topo_samp_size = int(snakemake.params.samp_size)    # type: ignore[name-defined]
    topo_amp = str(snakemake.params.amp)                # type: ignore[name-defined]
except NameError:
    rww_results_path = "results/tvb_rww_results.csv"
    data_path = "results/tvb_rww/seed=0/amp=medium/dataset.npz"
    model_path = ("results/tvb_rww/seed=0/amp=medium/samp_size=5000/"
                  "method=lacerda/model.pkl")
    recovery_pdf_path = "results/tvb_rww.pdf"
    topology_pdf_path = "results/tvb_rww_topology.pdf"
    topo_samp_size = 5000
    topo_amp = "medium"

rww = pd.read_csv(rww_results_path)
rww = rww[rww["method"] == "lacerda"].copy()


def _recovery_panel(ax, y_col, ylabel, title):
    for amp in AMPS:
        sns.lineplot(
            data=rww[rww["amp"] == amp], x="samp_size", y=y_col,
            color=AMP_COLOR[amp], estimator="median", errorbar=("ci", 95),
            marker="o", markersize=8, linewidth=2.4, ax=ax,
        )
    ax.set_xscale("log")
    ax.set_xticks(SAMPLE_SIZES, labels=["100", "500", "2,000", "5,000"])
    ax.set_xlabel(r"number of samples $n$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="0.85", linewidth=1.0, zorder=0)
    ax.set_title(title, fontsize=14)


# Figure 1: recovery against the known ground truth.
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
_recovery_panel(
    axes[0], "ari_scc", r"ARI $\uparrow$", "feedback-cluster recovery",
)
_recovery_panel(
    axes[1], "cdag_fscore", r"$F_1$ $\uparrow$", "C-DAG recovery",
)
handles = [
    Line2D([0], [0], color=AMP_COLOR[a], lw=2.4, marker="o",
           label=AMP_LABEL[a])
    for a in AMPS
]
axes[1].legend(handles=handles, frameon=False, fontsize=11,
               loc="upper left", title="input scale", title_fontsize=11)
fig.tight_layout()
fig.savefig(recovery_pdf_path, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

# Figure 2: true and recovered C-DAGs, large enough to inspect.
data = np.load(data_path, allow_pickle=True)
true_adj = (data["weights"] != 0).astype(int)
np.fill_diagonal(true_adj, 0)
model = pickle.load(open(model_path, "rb"))

true_cdag = condensation_from_adjacency(true_adj)
true_pos = layered_positions(true_cdag)
cluster_face, _ = scc_coloring(true_cdag)
recovered_pos = layered_positions(model.dag)
fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
draw_cdag(axes[0], true_cdag, true_pos, cluster_face, "true C-DAG")
draw_cdag(
    axes[1], model.dag, recovered_pos, cluster_face,
    f"recovered C-DAG ({topo_amp} input, n={topo_samp_size})",
)
fig.suptitle("Wong-Wang topology example (seed 0)", fontsize=16, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(topology_pdf_path, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(f"Saved {recovery_pdf_path}")
print(f"Saved {topology_pdf_path}")
