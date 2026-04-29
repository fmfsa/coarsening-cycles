"""Generate synthetic datasets for OICA experiments.

Generates a random directed graph (ER or scale-free) with cycles, samples from
a linear non-Gaussian cyclic SEM, and saves the observational data together
with the ground-truth SCC partition (for evaluation against Tarjan's coarsening).

Graphs are guaranteed to contain at least one non-trivial SCC (i.e., at least
two nodes that are mutually reachable). If the first random seed does not
produce cycles, subsequent seeds are tried.

Ground truth in the .npz file:
  weights    — (n, n) weight matrix
  scc_labels — (n,) integer labels (nodes sharing a label are in the same SCC)
  obs        — (samp_size, n) observational samples
"""

import networkx as nx
import numpy as np
from repare_cycle.graph import random_directed_graph, CyclicLinearSEM

seed = int(snakemake.wildcards.seed)
density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
num_nodes = int(snakemake.wildcards.num_nodes)
graph_type = snakemake.wildcards["graph"]
noise_dist = getattr(snakemake.params, "noise_dist", "laplace")

# Generate a DG that contains at least one non-trivial SCC (a cycle).
# Retry with successive seeds if necessary.
for attempt in range(20):
    graph, weights = random_directed_graph(
        num_nodes, density, graph_type=graph_type, seed=seed + attempt * 1000
    )
    sccs = list(nx.strongly_connected_components(graph))
    if any(len(scc) > 1 for scc in sccs):
        break
else:
    # Fall back: use the last attempt even if no cycles (evaluation will reflect this)
    pass

# Ground-truth SCC partition (Tarjan's, via nx.strongly_connected_components)
scc_labels = np.zeros(num_nodes, dtype=int)
for label, scc in enumerate(sccs):
    for node in scc:
        scc_labels[node] = label

# SCC structure summary (stored for evaluation metadata)
num_sccs = len(sccs)
scc_sizes = np.array(sorted([len(s) for s in sccs], reverse=True))

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
    scc_sizes=scc_sizes,
    obs=obs_dataset,
)
