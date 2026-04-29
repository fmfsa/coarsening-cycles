"""Generate synthetic datasets with controlled multi-SCC structure.

Uses random_multi_scc_graph to create directed graphs with a specified number
of non-trivial SCCs (cycles), optional singletons, and DAG-respecting
inter-SCC edges.  This allows proper evaluation of whether the OICA pipeline
can distinguish between multiple distinct cycles.

Ground truth in the .npz file:
  weights    — (n, n) weight matrix
  scc_labels — (n,) integer labels (nodes sharing a label are in the same SCC)
  scc_sizes  — sorted SCC sizes (descending)
  obs        — (samp_size, n) observational samples
"""

import networkx as nx
import numpy as np
from repare_cycle.graph import random_multi_scc_graph, CyclicLinearSEM

seed = int(snakemake.wildcards.seed)
n_sccs = int(snakemake.wildcards.n_sccs)
scc_size = int(snakemake.wildcards.scc_size)
samp_size = int(snakemake.wildcards.samp_size)
inter_density = float(snakemake.wildcards.inter_density)
n_singletons = int(getattr(snakemake.params, "n_singletons", 2))
noise_dist = getattr(snakemake.params, "noise_dist", "laplace")
intra_extra_edges = int(getattr(snakemake.params, "intra_extra_edges", 1))

graph, weights = random_multi_scc_graph(
    n_sccs=n_sccs,
    scc_sizes=scc_size,
    n_singletons=n_singletons,
    inter_density=inter_density,
    intra_extra_edges=intra_extra_edges,
    seed=seed,
)

num_nodes = weights.shape[0]

# Ground-truth SCC partition (Tarjan's)
sccs = list(nx.strongly_connected_components(graph))
scc_labels = np.zeros(num_nodes, dtype=int)
for label, scc in enumerate(sccs):
    for node in scc:
        scc_labels[node] = label

scc_sizes_arr = np.array(sorted([len(s) for s in sccs], reverse=True))

# Cyclic linear SEM — Laplace noise required for ICA identifiability
model = CyclicLinearSEM(
    weights,
    means=(-2.0, 2.0),
    variances=(0.5, 2.0),
    rng=np.random.default_rng(seed),
    noise_dist=noise_dist,
)

obs_dataset = model.sample(samp_size)

np.savez(
    snakemake.output[0],
    weights=weights,
    scc_labels=scc_labels,
    scc_sizes=scc_sizes_arr,
    obs=obs_dataset,
)
