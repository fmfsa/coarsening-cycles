"""Fit the Lacerda observational recipe on a synthetic cyclic dataset.

Pipeline:
    FastICA → Â → B̂ = I − Â⁻¹ → threshold → Tarjan on supp(B̂) → condensation.
The resulting cluster DAG is the identification target of Theorem 2 in the
paper (condensation identifiability).
"""

import pickle
import time
from types import SimpleNamespace

import networkx as nx
import numpy as np

from repare_cycle.oica import estimate_mixing_matrix, mixing_to_adjacency

threshold = float(getattr(snakemake.params, "threshold", 0.1))
max_iter = int(getattr(snakemake.params, "max_iter", 10_000))
random_state = int(getattr(snakemake.params, "random_state", 0))

method = snakemake.wildcards.method
if method != "lacerda":
    raise ValueError(f"Unknown method: {method}. Only 'lacerda' is supported.")

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


start = time.perf_counter()
mixing = estimate_mixing_matrix(
    obs, max_iter=max_iter, random_state=random_state, tol=1e-6,
)
B = mixing_to_adjacency(mixing, threshold=threshold)
fit_runtime_sec = time.perf_counter() - start

# B[i, j] = 1 means edge j -> i (column convention from mixing_to_adjacency).
n_nodes = B.shape[0]
dg = nx.DiGraph()
dg.add_nodes_from(range(n_nodes))
rows, cols = np.where(B > 0)
for i, j in zip(rows, cols):
    dg.add_edge(j, i)

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

# Full variable-level B estimate in truth convention (i→j).
full_adj_ij = B.T
model = SimpleNamespace(
    dag=dag,
    full_adj_ij=full_adj_ij,
    full_dag=_adj_to_digraph(full_adj_ij),
    fit_runtime_sec=fit_runtime_sec,
)

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
