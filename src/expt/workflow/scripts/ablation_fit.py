"""Candidate-selection ablation: one FastICA estimate, three selection branches.

Fits FastICA ONCE per dataset cell and derives every selection branch from
that single unmixing matrix ``W``, so differences between branches isolate
candidate *selection*, not ICA randomness:

  abl_first_stable   — N-rooks enumeration, first candidate with ρ(B̂) < 1.
  abl_random_r{r}    — N-rooks enumeration, uniform draw from all admissible
                       candidates; ``r`` indexes independent draws from the
                       SAME enumerated set (draws share one dataset/W and
                       must be averaged within-cell before across-seed CIs).
  abl_hungarian      — single admissible representative via linear
                       assignment; performs ZERO equivalence-class
                       enumeration (n_candidates_enumerated = 0).

The N-rooks enumeration itself runs once and is shared by the first-stable
and random branches (they differ only in the choice rule); its wall time is
charged to both branches' ``selection_runtime_sec`` because each strategy
would have to pay it standalone.

Every branch's pickle stores ``W_hash`` (SHA-256 of the shared ``W``) so a
post-hoc check can verify all branches of a cell consumed the identical
ICA estimate, plus the weighted ``B_chosen`` (column convention) as
``B_weighted`` for downstream reuse.
"""

import hashlib
import pickle
import time
from types import SimpleNamespace

import networkx as nx
import numpy as np

from repare_cycle.lingd import (
    choose_enumerated_candidate,
    enumerate_admissible_candidates,
    fit_ica_unmixing,
    hungarian_candidate,
)

threshold = float(getattr(snakemake.params, "threshold", 0.1))
max_iter = int(getattr(snakemake.params, "max_iter", 10_000))
max_perms = int(getattr(snakemake.params, "max_perms", 10_000))
# Wall-clock budget per enumeration round: the N-rooks backtracking is
# worst-case exponential; pruned rounds are flagged enumeration_timed_out
# and treated as truncated candidate lists downstream.
enum_budget = float(getattr(snakemake.params, "enum_time_budget_sec", 60.0))
n_random_draws = int(getattr(snakemake.params, "n_random_draws", 5))
seed = int(snakemake.wildcards.seed)

data = np.load(snakemake.input.data, allow_pickle=True)
obs = data["obs"]
n_nodes = obs.shape[1]

MIN_THRESHOLD_W = 0.01  # same halving floor as run_lingd


def _adj_to_digraph(full_adj_ij):
    g = nx.DiGraph()
    g.add_nodes_from(range(full_adj_ij.shape[0]))
    rows, cols = np.where(full_adj_ij > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        g.add_edge(int(i), int(j))
    return g


def _condensation_model(B_weighted, meta):
    """Package a column-convention B̂ into the same SimpleNamespace layout
    as fit.py (dag / full_adj_ij / full_dag), plus ablation metadata."""
    dg = nx.DiGraph()
    dg.add_nodes_from(range(n_nodes))
    rows, cols = np.where(np.abs(B_weighted) > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i == j:
            continue
        dg.add_edge(int(j), int(i))  # column convention: B[i,j] -> edge j→i

    sccs = [frozenset(scc) for scc in nx.strongly_connected_components(dg)]
    condensation = nx.condensation(dg)
    dag = nx.DiGraph()
    for scc in sccs:
        dag.add_node(scc)
    scc_by_cond_node = {
        k: frozenset(condensation.nodes[k]["members"]) for k in condensation.nodes
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


# ── Stage 1: one shared FastICA estimate ─────────────────────────────────
t0 = time.perf_counter()
W, ica_err = fit_ica_unmixing(
    obs, ica_max_iter=max_iter, ica_tolerance=1e-6, random_state=0
)
ica_runtime_sec = time.perf_counter() - t0
W_hash = hashlib.sha256(W.tobytes()).hexdigest() if W is not None else None

# ── Stage 2 (shared): N-rooks enumeration with threshold-w halving ───────
if W is not None:
    t1 = time.perf_counter()
    thr_w = threshold
    while True:
        candidates = enumerate_admissible_candidates(
            W, threshold_b=threshold, threshold_w=thr_w, max_perms=max_perms,
            time_budget_sec=enum_budget,
        )
        n_found = len(candidates["stable"]) + len(candidates["unstable"])
        if n_found > 0 or thr_w <= MIN_THRESHOLD_W:
            break
        thr_w /= 2
    enumeration_runtime_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    thr_w_hun = threshold
    while True:
        B_hun = hungarian_candidate(
            W, threshold_b=threshold, threshold_w=thr_w_hun
        )
        if B_hun is not None or thr_w_hun <= MIN_THRESHOLD_W:
            break
        thr_w_hun /= 2
    hungarian_runtime_sec = time.perf_counter() - t2
else:
    candidates = {
        "stable": [],
        "unstable": [],
        "n_candidates_enumerated": 0,
        "enumeration_cap_hit": False,
        "enumeration_timed_out": False,
    }
    enumeration_runtime_sec = 0.0
    B_hun = None
    hungarian_runtime_sec = 0.0

common_meta = dict(
    W_hash=W_hash,
    ica_failed=W is None,
    ica_last_error=ica_err,
    ica_runtime_sec=ica_runtime_sec,
)
enum_meta = dict(
    n_candidates_enumerated=candidates["n_candidates_enumerated"],
    n_candidates_returned=(
        len(candidates["stable"]) + len(candidates["unstable"])
    ),
    enumeration_cap_hit=candidates["enumeration_cap_hit"],
    enumeration_timed_out=candidates["enumeration_timed_out"],
    n_stable_candidates=len(candidates["stable"]),
    n_unstable_candidates=len(candidates["unstable"]),
)

# ── Stage 3: selection branches ──────────────────────────────────────────
B_zero = np.zeros((n_nodes, n_nodes))
outputs = {}  # snakemake output name -> model namespace

# (a) first_stable — enumeration cost is part of the strategy's selection.
t3 = time.perf_counter()
B_fs, fs_is_stable = choose_enumerated_candidate(
    candidates, pick_strategy="first_stable"
)
fs_select = enumeration_runtime_sec + (time.perf_counter() - t3)
outputs["first_stable"] = _condensation_model(
    B_fs if B_fs is not None else B_zero,
    dict(
        common_meta,
        **enum_meta,
        pick_strategy="first_stable",
        chosen_is_stable=fs_is_stable,
        selection_runtime_sec=fs_select,
        fit_runtime_sec=ica_runtime_sec + fs_select,
        random_draw_index=float("nan"),
    ),
)

# (b) independent uniform draws from the SAME enumerated candidate set.
for r in range(n_random_draws):
    t4 = time.perf_counter()
    B_rnd, rnd_is_stable = choose_enumerated_candidate(
        candidates,
        pick_strategy="random_admissible",
        random_state=seed * 100 + r,
    )
    rnd_select = enumeration_runtime_sec + (time.perf_counter() - t4)
    outputs[f"random_r{r}"] = _condensation_model(
        B_rnd if B_rnd is not None else B_zero,
        dict(
            common_meta,
            **enum_meta,
            pick_strategy="random_admissible",
            chosen_is_stable=rnd_is_stable,
            selection_runtime_sec=rnd_select,
            fit_runtime_sec=ica_runtime_sec + rnd_select,
            random_draw_index=r,
        ),
    )

# (c) Hungarian — zero enumeration; honest metadata.
outputs["hungarian"] = _condensation_model(
    B_hun if B_hun is not None else B_zero,
    dict(
        common_meta,
        pick_strategy="hungarian_any",
        n_candidates_enumerated=0,
        n_candidates_returned=1 if B_hun is not None else 0,
        enumeration_cap_hit=False,
        enumeration_timed_out=False,
        n_stable_candidates=float("nan"),
        n_unstable_candidates=float("nan"),
        chosen_is_stable=None,
        selection_runtime_sec=hungarian_runtime_sec,
        fit_runtime_sec=ica_runtime_sec + hungarian_runtime_sec,
        random_draw_index=float("nan"),
    ),
)

for name, model in outputs.items():
    with open(getattr(snakemake.output, name), "wb") as f:
        pickle.dump(model, f)
