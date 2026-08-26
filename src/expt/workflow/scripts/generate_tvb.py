"""Generate seed-level datasets for the TVB linear-equilibrium pilot.

Controlled benchmark: d=20 nodes partitioned into 5 non-trivial SCCs
(random_cyclic_graph), spectral radius rescaled to 0.9, weight magnitudes
drawn from [0.5, 0.95] before rescaling. Inputs are independent
standardized Laplace draws (mean 0, variance 1), and observations are the
exact equilibria X = E (I - W)^{-1} — the same quantity TVB's linear model
converges to under a constant regional stimulus E (see
repare_cycle.tvb_bridge). This script needs numpy/networkx only; TVB is
required nowhere in the fit path.

Ground truth in the .npz file (the first four keys match generate_synth.py
so fit.py works unchanged):
  weights                 — (d, d) weight matrix, repository orientation
                            weights[src, dst]
  scc_labels              — (d,) integer SCC labels
  scc_sizes               — sorted SCC sizes (descending)
  obs                     — (n_obs, d) exact equilibrium observations
  weights_tvb             — (d, d) TVB orientation weights[target, source]
  inputs                  — (n_obs, d) standardized Laplace inputs E
  spectral_radius         — ρ(W) after rescaling
  orientation             — orientation metadata string
  intervention_scc_labels — (K,) SCC label of each soft intervention
  intervention_scc_sizes  — (K,) size of the shifted SCC
  intervention_inputs     — (K, d) delta_e rows: 1/sqrt(size) on members
  intervention_responses  — (K, d) exact steady-state responses
                            delta_x = delta_e (I - W)^{-1}
"""

import networkx as nx
import numpy as np
from repare_cycle.graph import random_cyclic_graph
from repare_cycle.tvb_bridge import exact_equilibrium, to_tvb_orientation

seed = int(snakemake.wildcards.seed)
d = int(snakemake.params.d)
num_cycles = int(snakemake.params.num_cycles)
density = float(snakemake.params.density)
intra_scc_density = float(snakemake.params.intra_scc_density)
weight_range = (float(snakemake.params.weight_lo), float(snakemake.params.weight_hi))
target_rho = float(snakemake.params.target_rho)
n_obs = int(snakemake.params.n_obs)

graph, weights = random_cyclic_graph(
    d=d,
    num_cycles=num_cycles,
    density=density,
    seed=seed,
    weight_range=weight_range,
    target_rho=target_rho,
    intra_scc_density=intra_scc_density,
)

# Ground-truth SCC partition
sccs = list(nx.strongly_connected_components(graph))
scc_labels = np.zeros(d, dtype=int)
for label, scc in enumerate(sccs):
    for node in scc:
        scc_labels[node] = label

scc_sizes_arr = np.array(sorted([len(s) for s in sccs], reverse=True))

# Standardized Laplace inputs (mean 0, variance 1 = 2·scale²). A fresh
# spawn key: random_cyclic_graph already consumed default_rng(seed), and
# reusing the bare seed would replay the identical stream.
rng = np.random.default_rng([seed, 314159])
inputs = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=(n_obs, d))
obs = exact_equilibrium(weights, inputs)

# One soft intervention per true SCC (singletons included): shift each
# member's input by 1/sqrt(|SCC|); the exact steady-state response is
# delta_x = delta_e (I - W)^{-1}.
intervention_inputs = np.zeros((len(sccs), d))
for label, scc in enumerate(sccs):
    intervention_inputs[label, list(scc)] = 1.0 / np.sqrt(len(scc))
intervention_responses = exact_equilibrium(weights, intervention_inputs)

np.savez(
    snakemake.output[0],
    weights=weights,
    scc_labels=scc_labels,
    scc_sizes=scc_sizes_arr,
    obs=obs,
    weights_tvb=to_tvb_orientation(weights),
    inputs=inputs,
    spectral_radius=float(np.max(np.abs(np.linalg.eigvals(weights)))),
    orientation="weights[src,dst]; weights_tvb = weights.T = [target,source]",
    intervention_scc_labels=np.arange(len(sccs)),
    intervention_scc_sizes=np.array([len(s) for s in sccs]),
    intervention_inputs=intervention_inputs,
    intervention_responses=intervention_responses,
)
