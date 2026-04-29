"""Evaluate a fitted PartitionDgModelOICA on the main synthetic experiment.

Ground truth: the SCC partition (Tarjan's algorithm on the true DG).

Metrics saved:
  ari_scc          — ARI vs SCC partition (partition quality)
  precision        — edge precision vs true condensation edges
  recall           — edge recall
  fscore           — edge F-score
  runtime_sec      — fitting time
  num_sccs         — number of SCCs in the true DG
  max_scc_size     — size of the largest SCC
  num_nontrivial   — number of SCCs with size > 1
  frac_nontrivial  — fraction of nodes in non-trivial SCCs
  has_cycles       — 1 if any non-trivial SCC exists
"""

import pickle

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

num_cycles = int(snakemake.wildcards.num_cycles)
density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
seed = int(snakemake.wildcards.seed)
d = int(snakemake.wildcards.d)
regime = getattr(snakemake.wildcards, "regime", "easy")

model = pickle.load(open(snakemake.input.model, "rb"))
data = np.load(snakemake.input.data, allow_pickle=True)

weights = data["weights"]
scc_labels_true = data["scc_labels"]
num_nodes = weights.shape[0]

# True directed graph
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

# ARI vs SCC ground truth
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
# Check all ordered pairs — the partition DAG node iteration order is not
# guaranteed to be topological (e.g. nx.strongly_connected_components yields
# SCCs in reverse-topological order). Iterating only forward in `node_list`
# would miss every true edge when the order is reversed.
for pa in node_list:
    for ch in node_list:
        if pa is ch:
            continue
        if _is_adj(pa, ch):
            true_edges_on_est.add_edge(pa, ch)

tp = sum(1 for e in model.dag.edges if e in true_edges_on_est.edges)
n_pred = len(model.dag.edges)
n_true = len(true_edges_on_est.edges)

# Edge case handling: both empty is perfect match; one empty is total failure
if n_pred == 0 and n_true == 0:
    # Both empty: no inter-SCC edges predicted and none exist (perfect match)
    precision = recall = 1.0
elif n_pred == 0 or n_true == 0:
    # One empty, one not: complete failure to recover inter-SCC structure
    precision = recall = 0.0
else:
    # Normal case: compute standard precision/recall
    precision = tp / n_pred
    recall = tp / n_true

fscore = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

# -------------------------------------------------------------------------
# Variable-level (full DAG) metrics — only computed when the method exposes
# a full variable-level adjacency. `lacerda` saves `full_adj_ij` in i→j
# convention.
# -------------------------------------------------------------------------
true_adj_ij = (weights != 0).astype(int)
np.fill_diagonal(true_adj_ij, 0)

full_adj = getattr(model, "full_adj_ij", None)
if full_adj is None:
    var_precision = float("nan")
    var_recall = float("nan")
    var_fscore = float("nan")
    var_shd = float("nan")
    inter_scc_precision = float("nan")
    inter_scc_recall = float("nan")
    inter_scc_fscore = float("nan")
else:
    pred = (np.asarray(full_adj) > 0).astype(int)
    np.fill_diagonal(pred, 0)
    tp_v = int(np.sum((pred == 1) & (true_adj_ij == 1)))
    fp_v = int(np.sum((pred == 1) & (true_adj_ij == 0)))
    fn_v = int(np.sum((pred == 0) & (true_adj_ij == 1)))
    var_precision = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 1.0
    var_recall = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 1.0
    var_fscore = (
        2 * var_precision * var_recall / (var_precision + var_recall)
        if (var_precision + var_recall) > 0 else 0.0
    )
    # Structural Hamming Distance (directed): missing + extra + reversed.
    # Counted as: edges in exactly one of {pred, true} plus reversed pairs.
    diff = pred - true_adj_ij
    var_shd = int(np.sum(np.abs(diff)))

    # Inter-(true-SCC) F-score — the identifiable target from observational
    # data. Mask out intra-true-SCC pairs (i, j) since within-SCC structure
    # is not identifiable. This is the metric where the paper's claim lives:
    # methods that recover the SCC partition + inter-block edges should hit
    # F = 1; Lacerda (which misidentifies SCCs from noisy B̂) should not.
    var_to_true_scc = np.empty(num_nodes, dtype=int)
    for label, scc in enumerate(true_sccs):
        for v in scc:
            var_to_true_scc[v] = label
    inter_mask = var_to_true_scc[:, None] != var_to_true_scc[None, :]
    pred_inter = pred & inter_mask
    true_inter = true_adj_ij & inter_mask
    tp_i = int(np.sum((pred_inter == 1) & (true_inter == 1)))
    fp_i = int(np.sum((pred_inter == 1) & (true_inter == 0)))
    fn_i = int(np.sum((pred_inter == 0) & (true_inter == 1)))
    inter_scc_precision = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 1.0
    inter_scc_recall = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 1.0
    inter_scc_fscore = (
        2 * inter_scc_precision * inter_scc_recall
        / (inter_scc_precision + inter_scc_recall)
        if (inter_scc_precision + inter_scc_recall) > 0 else 0.0
    )

results = {
    "method": snakemake.wildcards.method,
    "regime": regime,
    "num_cycles": num_cycles,
    "density": density,
    "samp_size": samp_size,
    "seed": seed,
    "d": d,
    "num_nodes": num_nodes,
    "ari_scc": ari_scc,
    "precision": precision,
    "recall": recall,
    "fscore": fscore,
    "var_precision": var_precision,
    "var_recall": var_recall,
    "var_fscore": var_fscore,
    "var_shd": var_shd,
    "inter_scc_precision": inter_scc_precision,
    "inter_scc_recall": inter_scc_recall,
    "inter_scc_fscore": inter_scc_fscore,
    # Hyperparameter-tuner outputs (NaN for non-tuned methods).
    "chosen_threshold": (model.tuner["chosen_threshold"]
                         if hasattr(model, "tuner") else float("nan")),
    "chosen_beta": (model.tuner["chosen_beta"]
                    if hasattr(model, "tuner") else float("nan")),
    "bic_score": (model.tuner.get("bic_score", model.tuner.get("oracle_score", float("nan")))
                  if hasattr(model, "tuner") else float("nan")),
    "runtime_sec": float(getattr(model, "fit_runtime_sec", float("nan"))),
    # SCC structure (from Tarjan's on the true graph)
    "num_sccs": num_sccs,
    "max_scc_size": max_scc_size,
    "num_nontrivial": num_nontrivial,
    "frac_nontrivial": frac_nontrivial,
    "has_cycles": has_cycles,
    "num_parts_est": model.dag.number_of_nodes(),
}
pd.DataFrame([results]).to_csv(snakemake.output[0], index=False)
