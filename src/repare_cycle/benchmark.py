"""Adapters and metrics for the paired GroupLiNGAM comparison.

All adjacency arguments here use row=source, column=effect. GroupLiNGAM
groups are its native output; NaNs within a group remain unknown coefficients.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score


def validate_groups(groups, d):
    parts = [frozenset(int(v) for v in group) for group in groups]
    flat = [v for group in parts for v in group]
    if any(not p for p in parts) or len(flat) != d or set(flat) != set(range(d)):
        raise ValueError("Estimated groups must partition all observed variables exactly once")
    return parts


def group_lingam_adapter(groups, coefficients):
    """Return native groups and *known* directed edges, without filling NaNs."""
    b = np.asarray(coefficients, dtype=float)
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError("Expected a square coefficient matrix")
    parts = validate_groups(groups, len(b))
    if np.isinf(b).any():
        raise ValueError("Infinite coefficient")
    membership = {v: i for i, part in enumerate(parts) for v in part}
    for i, j in zip(*np.where(np.isnan(b))):
        if membership[i] != membership[j]:
            raise ValueError("Unknown coefficient across estimated groups")
    adj = np.isfinite(b.T) & (np.abs(b.T) > 0)
    np.fill_diagonal(adj, False)
    return parts, adj


def groups_from_adjacency(adj):
    return [frozenset(s) for s in nx.strongly_connected_components(nx.DiGraph(adj))]


def _edges(parts, adj):
    membership = {v: part for part in parts for v in part}
    return {(membership[u], membership[v]) for u, v in zip(*np.nonzero(adj))
            if membership[u] != membership[v]}


def _f1(predicted, truth):
    return 2 * len(predicted & truth) / (len(predicted) + len(truth)) if predicted or truth else 1.0


def condensation_metrics(weights, groups, adj):
    """Exact equality compares original variable memberships, not graph isomorphism.

Oracle F1 projects only known predicted directed edges onto the true groups;
unknown within-estimated-group edges are absent from this diagnostic. Always
report it alongside partition accuracy and end-to-end exact recovery.
"""
    truth = np.asarray(weights) != 0
    adj = np.asarray(adj, dtype=bool)
    if truth.shape != adj.shape:
        raise ValueError("Truth and prediction shapes differ")
    true_parts = groups_from_adjacency(truth)
    parts = validate_groups(groups, len(truth))
    true_edges = _edges(true_parts, truth)
    predicted_edges = _edges(parts, adj)
    def labels(pp):
        out = np.empty(len(truth), dtype=int)
        for k, part in enumerate(pp):
            out[list(part)] = k
        return out
    return {
        "ari_scc": float(adjusted_rand_score(labels(true_parts), labels(parts))),
        "oracle_cluster_f1": float(_f1(_edges(true_parts, adj), true_edges)),
        "exact_condensation": int(set(parts) == set(true_parts) and predicted_edges == true_edges),
        "n_groups": len(parts),
    }
