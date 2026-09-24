"""Evaluate a fitted model on the TVB linear-equilibrium pilot grid.

Fork of evaluate_synth.py for the TVB grid (wildcards: seed, samp_size,
method; the grid constants d/num_cycles/density arrive via params). The
metric blocks — ARI, cluster-DAG F1 projected onto the TRUE SCC partition,
variable-level and inter-SCC F1, SCC summaries — are copied verbatim so the
pilot's numbers are directly comparable to the synthetic experiments.

Added for the pilot:
  desc_precision / desc_recall / desc_fscore — cross-SCC descendant
      reachability over ordered variable pairs (transitive closures of the
      true and estimated condensations expanded to variable pairs, scored
      only on pairs from different TRUE SCCs — no cluster-label alignment).
  n_desc_pairs / n_desc_true / n_desc_pred   — pair counts behind the score.
  cdag_precision / cdag_recall / cdag_fscore — DIRECT comparison of the
      recovered C-DAG (`model.dag`) against the true condensation: an edge
      counts only when both endpoint clusters match a true SCC exactly.
      Unlike the projected `fscore`, this is 0 when the estimate collapses
      to a single cluster (no C-DAG edges), so it evaluates the object the
      pipeline actually outputs.
  partition_exact                            — 1 iff the estimated cluster
      set equals the true SCC partition exactly.
  n_cdag_edges_true / n_cdag_edges_est       — edge counts behind cdag_*.
  spectral_radius, n_obs_total               — dataset provenance.

Needs numpy/networkx/pandas/sklearn only — TVB is not imported.
"""

import pickle

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from repare_cycle.tvb_bridge import (
    cdag_direct_scores,
    condensation_from_adjacency,
    descendant_reachability_scores,
)

num_cycles = int(snakemake.params.num_cycles)
density = float(snakemake.params.density)
d = int(snakemake.params.d)
samp_size = int(snakemake.wildcards.samp_size)
seed = int(snakemake.wildcards.seed)
regime = str(getattr(snakemake.params, "regime", "tvb_linear"))
# `amp` wildcard exists only in the Wong-Wang grid (Experiment 2).
amp = getattr(snakemake.wildcards, "amp", None)
threshold_used = float(getattr(snakemake.params, "threshold", 0.1))

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
# (Copied from evaluate_synth.py; see the discussion there.)
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
    fscore = 1.0
elif n_pred == 0 or n_true == 0:
    fscore = 0.0
else:
    precision_c = tp / n_pred
    recall_c = tp / n_true
    fscore = (2 * precision_c * recall_c / (precision_c + recall_c)
              if (precision_c + recall_c) > 0 else 0.0)

# -------------------------------------------------------------------------
# Variable-level metrics + cross-SCC descendant reachability — only when
# `full_adj_ij` exists (lacerda/lacerda_rnd save it).
# -------------------------------------------------------------------------
if full_adj is None:
    var_precision = float("nan")
    var_recall = float("nan")
    var_fscore = float("nan")
    var_shd = float("nan")
    inter_scc_precision = float("nan")
    inter_scc_recall = float("nan")
    inter_scc_fscore = float("nan")
    desc = {k: float("nan") for k in (
        "desc_precision", "desc_recall", "desc_fscore",
        "n_desc_pairs", "n_desc_true", "n_desc_pred",
    )}
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

    desc = descendant_reachability_scores(true_adj_ij, pred_var, scc_labels_true)

# Direct C-DAG comparison — evaluates the recovered `model.dag` itself.
cdag = cdag_direct_scores(condensation_from_adjacency(true_adj_ij), model.dag)

results = {
    "method": snakemake.wildcards.method,
    "regime": regime,
    "amp": amp if amp is not None else float("nan"),
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
    "desc_precision": desc["desc_precision"],
    "desc_recall": desc["desc_recall"],
    "desc_fscore": desc["desc_fscore"],
    "n_desc_pairs": desc["n_desc_pairs"],
    "n_desc_true": desc["n_desc_true"],
    "n_desc_pred": desc["n_desc_pred"],
    "cdag_precision": cdag["cdag_precision"],
    "cdag_recall": cdag["cdag_recall"],
    "cdag_fscore": cdag["cdag_fscore"],
    "partition_exact": cdag["partition_exact"],
    "n_cdag_edges_true": cdag["n_cdag_edges_true"],
    "n_cdag_edges_est": cdag["n_cdag_edges_est"],
    "runtime_sec": float(getattr(model, "fit_runtime_sec", float("nan"))),
    # SCC structure (from the true graph)
    "num_sccs": num_sccs,
    "max_scc_size": max_scc_size,
    "num_nontrivial": num_nontrivial,
    "frac_nontrivial": frac_nontrivial,
    "has_cycles": has_cycles,
    "num_parts_est": model.dag.number_of_nodes(),
    "spectral_radius": float(data["spectral_radius"]),
    "n_obs_total": int(data["obs"].shape[0]),
}
pd.DataFrame([results]).to_csv(snakemake.output[0], index=False)
