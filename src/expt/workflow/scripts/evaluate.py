"""Evaluate a fitted PartitionDgModelOICA against the SCC ground truth.

Ground truth: the SCC partition (Tarjan's algorithm on the true DG), which is
the finest valid DAG-coarsening identifiable from observational data alone.

Metrics saved:
  ari_scc     — ARI vs SCC partition (partition quality)
  precision   — edge precision vs true condensation edges
  recall      — edge recall
  fscore      — edge F-score
  runtime_sec — fitting time

SCC structure info (for analysis plots):
  num_sccs         — number of SCCs in the true DG
  max_scc_size     — size of the largest SCC
  num_nontrivial   — number of SCCs with size > 1
  frac_nontrivial  — fraction of nodes in non-trivial SCCs (size > 1)
  has_cycles       — 1 if any non-trivial SCC exists, else 0
"""

import pickle

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
seed = int(snakemake.wildcards.seed)
num_nodes = int(snakemake.wildcards["num_nodes"])
graph_family = snakemake.wildcards["graph"]

model = pickle.load(open(snakemake.input.model, "rb"))
data = np.load(snakemake.input.data, allow_pickle=True)

weights = data["weights"]
scc_labels_true = data["scc_labels"]

# True directed graph (may contain cycles)
true_dg = nx.DiGraph(weights.astype(bool))

# -------------------------------------------------------------------------
# SCC structure of the true graph
# -------------------------------------------------------------------------
true_sccs = list(nx.strongly_connected_components(true_dg))
num_sccs = len(true_sccs)
max_scc_size = max(len(s) for s in true_sccs)
num_nontrivial = sum(1 for s in true_sccs if len(s) > 1)
frac_nontrivial = sum(len(s) for s in true_sccs if len(s) > 1) / num_nodes
has_cycles = int(num_nontrivial > 0)

# -------------------------------------------------------------------------
# Estimated partition labels
# -------------------------------------------------------------------------
est_labels = np.zeros(num_nodes, dtype=int)
for label, part in enumerate(model.dag.nodes):
    est_labels[list(part)] = label

# ARI vs SCC ground truth.
# sklearn.adjusted_rand_score returns 0.0 (undefined 0/0) when both labelings
# have only one cluster.  That case means the algorithm correctly found the
# trivial partition (all-in-one block), so true ARI should be 1.0.
def _ari_robust(a, b):
    if len(set(a)) == 1 and len(set(b)) == 1:
        return 1.0
    return adjusted_rand_score(a, b)

ari_scc = _ari_robust(scc_labels_true, est_labels)

# -------------------------------------------------------------------------
# Edge F-score vs true condensation
# -------------------------------------------------------------------------
def _is_adj(pa, ch):
    return any(true_dg.has_edge(u, v) for u in pa for v in ch)


true_edges_on_est = nx.create_empty_copy(model.dag)
node_list = list(model.dag.nodes)
for i, pa in enumerate(node_list[:-1]):
    for ch in node_list[i + 1:]:
        if _is_adj(pa, ch):
            true_edges_on_est.add_edge(pa, ch)

tp = sum(1 for e in model.dag.edges if e in true_edges_on_est.edges)
precision = tp / len(model.dag.edges) if model.dag.edges else 1.0
recall = tp / len(true_edges_on_est.edges) if true_edges_on_est.edges else 1.0
fscore = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

results = {
    "density": density,
    "samp_size": samp_size,
    "seed": seed,
    "num_nodes": num_nodes,
    "graph_family": graph_family,
    "ari_scc": ari_scc,
    "precision": precision,
    "recall": recall,
    "fscore": fscore,
    "runtime_sec": float(getattr(model, "fit_runtime_sec", float("nan"))),
    # SCC structure
    "num_sccs": num_sccs,
    "max_scc_size": max_scc_size,
    "num_nontrivial": num_nontrivial,
    "frac_nontrivial": frac_nontrivial,
    "has_cycles": has_cycles,
    "num_parts_est": model.dag.number_of_nodes(),
}
pd.DataFrame([results]).to_csv(snakemake.output[0], index=False)
