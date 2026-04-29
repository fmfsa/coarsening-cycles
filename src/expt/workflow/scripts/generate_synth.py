"""Generate synthetic datasets for the main experiment.

Mirrors the original RePaRe paper's setup (d=10, varying interventions)
but instead varies the number of non-trivial SCCs (cycles) κ ∈ {0,1,2,3}.

Uses random_cyclic_graph which:
  - Randomly partitions d nodes into κ groups (≥2 each) + singletons
  - Builds directed cycles within each group + density-controlled extras
  - Adds DAG-respecting inter-block edges

Ground truth in the .npz file:
  weights    — (d, d) weight matrix
  scc_labels — (d,) integer labels (nodes sharing a label are in the same SCC)
  scc_sizes  — sorted SCC sizes (descending)
  obs        — (samp_size, d) observational samples
"""

import networkx as nx
import numpy as np
from repare_cycle.graph import random_cyclic_graph, CyclicLinearSEM

seed = int(snakemake.wildcards.seed)
num_cycles = int(snakemake.wildcards.num_cycles)
density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
d = int(snakemake.wildcards.d)
regime = getattr(snakemake.wildcards, "regime", "easy")
noise_dist = getattr(snakemake.params, "noise_dist", "laplace")

# Weight regimes: drives ρ(B) and the diagonal-dominance margin of A=(I-B)^{-1}.
# - "easy"   ρ(B) ~ 0.6, diag-margin ~ 0.7 → Lacerda heuristic essentially always works
# - "hard"   ρ(B) ~ 0.9, diag-margin ~ 0.2 → heuristic fragile
# - "harder" ρ(B) ~ 0.9, diag-margin can drop near 0 → unidentifiable seeds
WEIGHT_REGIMES = {
    "easy": (0.25, 0.75),
    "hard": (0.5, 0.95),
    "harder": (0.7, 0.95),
}
weight_range = WEIGHT_REGIMES[regime]

graph, weights = random_cyclic_graph(
    d=d,
    num_cycles=num_cycles,
    density=density,
    seed=seed,
    weight_range=weight_range,
)

# Ground-truth SCC partition (Tarjan's)
sccs = list(nx.strongly_connected_components(graph))
scc_labels = np.zeros(d, dtype=int)
for label, scc in enumerate(sccs):
    for node in scc:
        scc_labels[node] = label

scc_sizes_arr = np.array(sorted([len(s) for s in sccs], reverse=True))

# Cyclic linear SEM
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
