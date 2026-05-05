"""Fit a method on a synthetic cyclic dataset.

Methods:
    lacerda / lacerda_rnd
        Tetrad ICA-LiNG-D (FastICA + N-rooks permutation enumeration)
        → choose first stable B̂ (any member is fine: §3.2 condensation
                                  identifiability — every member has the
                                  same condensation)
        → threshold (already applied inside Tetrad)
        → Tarjan on supp(B̂)
        → condensation.

    disjointcycles
        Drton-et-al. higher-order moment procedure (R package). Produces
        a layered topological ordering whose layers are SCCs (cycles or
        singletons) plus an estimated variable-level adjacency matrix.
"""

import json
import os
import pickle
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

from repare_cycle.lingd import run_lingd

threshold = float(getattr(snakemake.params, "threshold", 0.1))
max_iter = int(getattr(snakemake.params, "max_iter", 10_000))
random_state = int(getattr(snakemake.params, "random_state", 0))
pick_strategy = str(getattr(snakemake.params, "pick_strategy", "first_stable"))

method = snakemake.wildcards.method
SUPPORTED_METHODS = ("lacerda", "lacerda_rnd", "disjointcycles")
if method not in SUPPORTED_METHODS:
    raise ValueError(f"Unknown method: {method!r}. Supported: {SUPPORTED_METHODS}.")

data = np.load(snakemake.input.data, allow_pickle=True)
obs = data["obs"]


def _adj_to_digraph(full_adj_ij):
    """Convert (n,n) binary matrix in i→j convention to nx.DiGraph."""
    g = nx.DiGraph()
    g.add_nodes_from(range(full_adj_ij.shape[0]))
    rows, cols = np.where(full_adj_ij > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        g.add_edge(int(i), int(j))
    return g


def _build_cluster_dag(clusters, full_adj_ij):
    """Construct the cluster DAG from a list of clusters (frozensets) and a
    variable-level adjacency matrix in i→j convention.

    Mirrors the lacerda branch: nodes are frozensets, edges are inter-cluster
    edges induced by the variable-level adjacency.
    """
    dag = nx.DiGraph()
    for c in clusters:
        dag.add_node(c)

    var_to_cluster = {}
    for c in clusters:
        for v in c:
            var_to_cluster[int(v)] = c

    for i, j in zip(*np.where(full_adj_ij > 0)):
        ci, cj = var_to_cluster[int(i)], var_to_cluster[int(j)]
        if ci != cj:
            dag.add_edge(ci, cj)
    return dag


if method in ("lacerda", "lacerda_rnd"):
    start = time.perf_counter()
    result = run_lingd(
        obs,
        threshold_b=threshold,
        threshold_w=threshold,
        ica_max_iter=max_iter,
        ica_tolerance=1e-6,
        pick_strategy=pick_strategy,
    )
    fit_runtime_sec = time.perf_counter() - start

    # Tetrad returns B in column convention (B[i,j] != 0  ⇔  j → i), same as ours.
    B = result["B_chosen"]
    n_nodes = B.shape[0]

    # Build the variable-level digraph from supp(B̂)
    dg = nx.DiGraph()
    dg.add_nodes_from(range(n_nodes))
    rows, cols = np.where(np.abs(B) > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i == j:
            continue
        dg.add_edge(int(j), int(i))  # column convention: B[i,j] -> edge j→i

    # Tarjan SCCs + condensation DAG
    sccs = [frozenset(scc) for scc in nx.strongly_connected_components(dg)]
    condensation = nx.condensation(dg)
    dag = nx.DiGraph()
    for scc in sccs:
        dag.add_node(scc)
    scc_by_cond_node = {
        k: frozenset(condensation.nodes[k]["members"])
        for k in condensation.nodes
    }
    for u, v in condensation.edges:
        dag.add_edge(scc_by_cond_node[u], scc_by_cond_node[v])

    # Full variable-level B estimate in truth (i→j) convention.
    # B is in column convention (B[i,j] = j→i), so transpose for the i→j layout.
    full_adj_ij = (np.abs(B.T) > 0).astype(int)
    np.fill_diagonal(full_adj_ij, 0)

    model = SimpleNamespace(
        dag=dag,
        full_adj_ij=full_adj_ij,
        full_dag=_adj_to_digraph(full_adj_ij),
        fit_runtime_sec=fit_runtime_sec,
        # Equivalence-class metadata for diagnostic / paper figures.
        n_stable_candidates=len(result["stable"]),
        n_unstable_candidates=len(result["unstable"]),
        chosen_is_stable=result["is_stable"],
    )

elif method == "disjointcycles":
    # Baseline: Drton et al. disjointCycles (R package) via Rscript subprocess.
    # The R wrapper writes (i) a (d, d) adjacency in i→j convention and
    # (ii) a JSON list of clusters (each one an SCC, 0-indexed).
    alpha = float(getattr(snakemake.params, "alpha", 0.01))
    r_script = Path(__file__).resolve().with_name("disjointcycles_fit.R")

    with tempfile.TemporaryDirectory() as td:
        obs_csv = os.path.join(td, "obs.csv")
        adj_csv = os.path.join(td, "adj.csv")
        ord_json = os.path.join(td, "ordering.json")
        np.savetxt(obs_csv, obs, delimiter=",")

        start = time.perf_counter()
        subprocess.run(
            ["Rscript", "--vanilla", str(r_script),
             obs_csv, str(alpha), adj_csv, ord_json],
            check=True,
        )
        fit_runtime_sec = time.perf_counter() - start

        full_adj_ij = np.loadtxt(adj_csv, delimiter=",", dtype=int)
        with open(ord_json) as f:
            ordering = json.load(f)

    full_adj_ij = np.atleast_2d(full_adj_ij).astype(int)
    np.fill_diagonal(full_adj_ij, 0)

    clusters = [frozenset(int(v) for v in c) for c in ordering]

    # Sanity: every variable should appear in exactly one cluster.
    seen = set().union(*clusters) if clusters else set()
    missing = set(range(obs.shape[1])) - seen
    for v in sorted(missing):
        clusters.append(frozenset({int(v)}))

    dag = _build_cluster_dag(clusters, full_adj_ij)

    model = SimpleNamespace(
        dag=dag,
        full_adj_ij=full_adj_ij,
        full_dag=_adj_to_digraph(full_adj_ij),
        fit_runtime_sec=fit_runtime_sec,
        disjointcycles_alpha=alpha,
    )

else:  # pragma: no cover — guarded by SUPPORTED_METHODS check above
    raise AssertionError(method)

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
