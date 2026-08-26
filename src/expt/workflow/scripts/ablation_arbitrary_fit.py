"""Arbitrary-row-permutation negative control for the selection ablation.

For each dataset, estimate FastICA with the same fixed settings as the other
selection procedures.  For every replicate, uniformly permute the rows of W,
normalise that permutation into B̂ regardless of its diagonal, and evaluate
the resulting SCC condensation.  This procedure deliberately ignores
LiNG-D's admissibility constraint; ``permutation_admissible`` records whether
the sampled permutation happened to satisfy it.
"""

import hashlib
import pickle
import time
from types import SimpleNamespace

import networkx as nx
import numpy as np

from repare_cycle.lingd import (
    arbitrary_permutation_candidate,
    fit_ica_unmixing,
)

threshold = float(getattr(snakemake.params, "threshold", 0.1))
max_iter = int(getattr(snakemake.params, "max_iter", 10_000))
n_random_draws = int(getattr(snakemake.params, "n_random_draws", 5))
seed = int(snakemake.wildcards.seed)

data = np.load(snakemake.input.data, allow_pickle=True)
obs = data["obs"]
n_nodes = obs.shape[1]


def _adj_to_digraph(full_adj_ij):
    g = nx.DiGraph()
    g.add_nodes_from(range(full_adj_ij.shape[0]))
    rows, cols = np.where(full_adj_ij > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        g.add_edge(int(i), int(j))
    return g


def _condensation_model(B_weighted, meta):
    dg = nx.DiGraph()
    dg.add_nodes_from(range(n_nodes))
    rows, cols = np.where(np.abs(B_weighted) > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i != j:
            dg.add_edge(int(j), int(i))  # column convention: B[i,j] -> j→i

    sccs = [frozenset(scc) for scc in nx.strongly_connected_components(dg)]
    condensation = nx.condensation(dg)
    dag = nx.DiGraph()
    dag.add_nodes_from(sccs)
    scc_by_cond_node = {
        k: frozenset(condensation.nodes[k]["members"])
        for k in condensation.nodes
    }
    for u, v in condensation.edges:
        dag.add_edge(scc_by_cond_node[u], scc_by_cond_node[v])

    full_adj_ij = (np.abs(B_weighted.T) > 0).astype(int)
    np.fill_diagonal(full_adj_ij, 0)
    return SimpleNamespace(
        dag=dag,
        full_adj_ij=full_adj_ij,
        full_dag=_adj_to_digraph(full_adj_ij),
        B_weighted=B_weighted,
        **meta,
    )


t0 = time.perf_counter()
W, ica_err = fit_ica_unmixing(
    obs, ica_max_iter=max_iter, ica_tolerance=1e-6, random_state=0
)
ica_runtime_sec = time.perf_counter() - t0
W_hash = hashlib.sha256(W.tobytes()).hexdigest() if W is not None else None
B_zero = np.zeros((n_nodes, n_nodes))

for r in range(n_random_draws):
    t1 = time.perf_counter()
    if W is None:
        B_rnd = B_zero
        admissible = False
        min_diag = float("nan")
    else:
        B_rnd, admissible, min_diag = arbitrary_permutation_candidate(
            W,
            threshold_b=threshold,
            threshold_w=threshold,
            random_state=seed * 100 + r,
        )
    selection_runtime_sec = time.perf_counter() - t1

    model = _condensation_model(
        B_rnd,
        dict(
            W_hash=W_hash,
            ica_failed=W is None,
            ica_last_error=ica_err,
            ica_runtime_sec=ica_runtime_sec,
            pick_strategy="arbitrary_permutation",
            permutation_admissible=admissible,
            min_abs_selected_diagonal=min_diag,
            n_candidates_enumerated=0,
            n_candidates_returned=1 if W is not None else 0,
            enumeration_cap_hit=False,
            enumeration_timed_out=False,
            n_stable_candidates=float("nan"),
            n_unstable_candidates=float("nan"),
            chosen_is_stable=None,
            selection_runtime_sec=selection_runtime_sec,
            fit_runtime_sec=ica_runtime_sec + selection_runtime_sec,
            random_draw_index=r,
        ),
    )
    with open(getattr(snakemake.output, f"arbitrary_r{r}"), "wb") as f:
        pickle.dump(model, f)
