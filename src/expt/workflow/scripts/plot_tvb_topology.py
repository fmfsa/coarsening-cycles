"""Graph-topology figure for Experiment 1 (seed 0).

Three panels showing the actual objects, not performance curves:

  (a) true cyclic variable graph — nodes colored by true SCC, laid out by
      the condensation's topological generations so the feedback clusters
      and their causal ordering are visible;
  (b) true C-DAG — one node per SCC, drawn as a colored cluster;
  (c) recovered C-DAG (`model.dag`, the algorithm's scientific output) —
      clusters that exactly match a true SCC inherit its color; mismatched
      clusters are outlined in rose.

The variable graph is cyclic by design; the C-DAG is its acyclic
abstraction — the identification target. The selected LiNG-D
representative (`model.full_adj_ij`) is deliberately NOT drawn: it is an
intermediate object used to find the SCCs, not a recovered identifiable
variable graph.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from _graph_draw import (
    draw_cdag,
    draw_var_graph,
    layered_positions,
    member_positions,
    scc_coloring,
)
from repare_cycle.tvb_bridge import condensation_from_adjacency

sns.set_context("paper", font_scale=1.8)
sns.set_style("white")

try:
    data_path = snakemake.input.data       # type: ignore[name-defined]
    model_path = snakemake.input.model     # type: ignore[name-defined]
    pdf_path = snakemake.output[0]         # type: ignore[name-defined]
    samp_size = int(snakemake.params.samp_size)  # type: ignore[name-defined]
except NameError:
    data_path = "results/tvb/seed=0/dataset.npz"
    model_path = "results/tvb/seed=0/samp_size=5000/method=lacerda/model.pkl"
    pdf_path = "results/tvb_topology.pdf"
    samp_size = 5000

data = np.load(data_path, allow_pickle=True)
weights = data["weights"]
true_adj = (weights != 0).astype(int)
np.fill_diagonal(true_adj, 0)

model = pickle.load(open(model_path, "rb"))

true_cdag = condensation_from_adjacency(true_adj)
recovered_cdag = model.dag

true_cluster_pos = layered_positions(true_cdag)
var_pos = member_positions(true_cluster_pos)
recovered_pos = layered_positions(recovered_cdag)
cluster_face, scc_of_var = scc_coloring(true_cdag)

fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8))
draw_var_graph(axes[0], true_adj, var_pos, scc_of_var, cluster_face,
               "true variable graph (cyclic)")
draw_cdag(axes[1], true_cdag, true_cluster_pos, cluster_face, "true C-DAG")
draw_cdag(axes[2], recovered_cdag, recovered_pos, cluster_face,
          f"recovered C-DAG (n={samp_size})")

fig.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(f"Saved {pdf_path}")
