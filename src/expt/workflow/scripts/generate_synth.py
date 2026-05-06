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
intra_scc_density = getattr(snakemake.params, "intra_scc_density", None)

# Regime config: weight range and (optionally) target spectral radius.
# - "hard"     ρ(B) ~ 0.9, weights uniform in [0.5, 0.95]; rescaled to ρ < 1.
# - "unstable" target ρ(B) = 1.5: same weight pool as "hard" but rescaled
#              upward instead of downward. Sampling from X = E (I-B)^{-1}
#              still works as long as (I-B) is invertible. Used to expose
#              the variable-level identifiability gap in §6.4 — the
#              stability filter cannot pick the true (unstable) B and
#              instead returns its stable equivalence-class twin, which
#              has the same condensation but a reversed cycle.
REGIME_CONFIG = {
    "hard":     {"weight_range": (0.5, 0.95), "target_rho": None},
    "unstable": {"weight_range": (0.5, 0.95), "target_rho": 1.5},
}
cfg = REGIME_CONFIG[regime]

graph, weights = random_cyclic_graph(
    d=d,
    num_cycles=num_cycles,
    density=density,
    seed=seed,
    weight_range=cfg["weight_range"],
    target_rho=cfg["target_rho"],
    intra_scc_density=intra_scc_density,
)

# Ground-truth SCC partition (Tarjan's)
sccs = list(nx.strongly_connected_components(graph))
scc_labels = np.zeros(d, dtype=int)
for label, scc in enumerate(sccs):
    for node in scc:
        scc_labels[node] = label

scc_sizes_arr = np.array(sorted([len(s) for s in sccs], reverse=True))

# Cyclic linear SEM. allow_unstable=True for the "unstable" regime (ρ ≥ 1).
model = CyclicLinearSEM(
    weights,
    means=(-2.0, 2.0),
    variances=(0.5, 2.0),
    rng=np.random.default_rng(seed),
    noise_dist=noise_dist,
    allow_unstable=(regime == "unstable"),
)

obs_dataset = model.sample(samp_size)

np.savez(
    snakemake.output[0],
    weights=weights,
    scc_labels=scc_labels,
    scc_sizes=scc_sizes_arr,
    obs=obs_dataset,
)
