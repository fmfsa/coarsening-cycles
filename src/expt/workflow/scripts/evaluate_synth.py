"""Evaluate a fitted Tetrad-LingD model on the main synthetic experiment.

Ground truth: the SCC partition (Tarjan's algorithm on the true DG).

Metrics saved:
  ari_scc            — ARI vs SCC partition (partition quality).
  fscore             — Cluster-DAG F1. Predicted variable edges are
                       projected onto the TRUE SCC partition; we compare
                       the resulting set of (src_scc, dst_scc) pairs
                       to the true cluster-DAG edge set. This is the
                       *identifiable target* in the paper's claim
                       (Lacerda Theorem 4): every member of the
                       distributional equivalence class shares the
                       same condensation, so this metric should → 1
                       across pick strategies.
  var_*              — Variable-level (full DAG) F1, precision, recall,
                       SHD. Not identifiable in the paper's claim —
                       saturates < 1 for `random_admissible`.
  inter_scc_*        — Variable-pair-level inter-SCC F1 (diagnostic).
                       Penalises x1→x2 vs x1→x4 even when both are
                       valid recoveries of the same cluster-DAG edge
                       {x1}→{x2,x3,x4}. Kept for diagnostic value.
  runtime_sec        — fitting time.
  num_sccs / max_scc_size / num_nontrivial / frac_nontrivial /
  has_cycles         — true-graph SCC structure summary.
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
# `threshold` wildcard exists only in the threshold-sensitivity grid (App. C.2);
# default to NaN elsewhere so the row is still well-formed in the merged CSV.
_thr_wc = getattr(snakemake.wildcards, "threshold", None)
threshold_used = float(_thr_wc) if _thr_wc is not None else float("nan")

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
# Variable → true-SCC label map. Used by both the cluster-DAG F1 and the
# inter-SCC variable-pair F1 below.
# -------------------------------------------------------------------------
var_to_true_scc = np.empty(num_nodes, dtype=int)
for label, scc in enumerate(true_sccs):
    for v in scc:
        var_to_true_scc[v] = label

# True variable-level adjacency, i→j convention.
true_adj_ij = (weights != 0).astype(int)
np.fill_diagonal(true_adj_ij, 0)

# -------------------------------------------------------------------------
# Cluster-DAG F1 — projects predicted variable edges to the TRUE SCC
# partition, then computes F1 against the true cluster-DAG edge set.
#
# This is the metric that directly tests Lacerda Theorem 4: any member of
# the distributional equivalence class has the same condensation, so this
# metric should → 1 regardless of which member the algorithm picks.
#
# Projecting onto the TRUE partition (rather than the estimated one) keeps
# the metric well-defined when the estimate collapses to a single SCC at
# small n: in that case `pred_cluster_edges` is empty, the truth has
# multiple cluster edges, and recall correctly drops to 0 (instead of the
# old vacuous 1.0 from "both empty = perfect match").
# -------------------------------------------------------------------------
true_cluster_edges = set()
for u, v in zip(*np.nonzero(true_adj_ij)):
    su, sv = int(var_to_true_scc[u]), int(var_to_true_scc[v])
    if su != sv:
        true_cluster_edges.add((su, sv))

full_adj = getattr(model, "full_adj_ij", None)
if full_adj is not None:
    pred_var = (np.asarray(full_adj) > 0).astype(int)
    np.fill_diagonal(pred_var, 0)
    pred_cluster_edges = set()
    for u, v in zip(*np.nonzero(pred_var)):
        su, sv = int(var_to_true_scc[u]), int(var_to_true_scc[v])
        if su != sv:
            pred_cluster_edges.add((su, sv))
else:
    pred_cluster_edges = set()

tp = len(pred_cluster_edges & true_cluster_edges)
n_pred = len(pred_cluster_edges)
n_true = len(true_cluster_edges)

if n_pred == 0 and n_true == 0:
    # Truth is one big SCC and prediction agrees — vacuously perfect.
    fscore = 1.0
elif n_pred == 0 or n_true == 0:
    fscore = 0.0
else:
    precision_c = tp / n_pred
    recall_c = tp / n_true
    fscore = (2 * precision_c * recall_c / (precision_c + recall_c)
              if (precision_c + recall_c) > 0 else 0.0)

# -------------------------------------------------------------------------
# Variable-level (full DAG) metrics — only when `full_adj_ij` exists.
# `lacerda` and `lacerda_rnd` save it; other methods would not.
#
# Two flavours:
#   * `var_*`: F1 over ALL variable edges (i, j). Penalises both
#     intra-SCC structural errors and inter-SCC variable shifts —
#     the strongest "everything must be exactly right" metric.
#   * `inter_scc_*`: F1 restricted to variable pairs (i, j) that
#     cross true-SCC boundaries. Penalises x1→x2 vs x1→x4 even when
#     they project to the same cluster-DAG edge — kept as a
#     diagnostic showing how variable-level inter-SCC edges *shift*
#     across equivalence-class members. The paper's identifiability
#     target is `fscore` (cluster-DAG F1) above, not this.
# -------------------------------------------------------------------------
if full_adj is None:
    var_precision = float("nan")
    var_recall = float("nan")
    var_fscore = float("nan")
    var_shd = float("nan")
    inter_scc_precision = float("nan")
    inter_scc_recall = float("nan")
    inter_scc_fscore = float("nan")
else:
    tp_v = int(np.sum((pred_var == 1) & (true_adj_ij == 1)))
    fp_v = int(np.sum((pred_var == 1) & (true_adj_ij == 0)))
    fn_v = int(np.sum((pred_var == 0) & (true_adj_ij == 1)))
    var_precision = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 1.0
    var_recall = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 1.0
    var_fscore = (
        2 * var_precision * var_recall / (var_precision + var_recall)
        if (var_precision + var_recall) > 0 else 0.0
    )
    # Structural Hamming Distance (directed): missing + extra + reversed.
    diff = pred_var - true_adj_ij
    var_shd = int(np.sum(np.abs(diff)))

    inter_mask = var_to_true_scc[:, None] != var_to_true_scc[None, :]
    pred_inter = pred_var & inter_mask
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
    "threshold": threshold_used,
    "ari_scc": ari_scc,
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
