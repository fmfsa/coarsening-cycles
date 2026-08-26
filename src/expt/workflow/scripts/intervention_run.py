"""Whole-SCC hard-intervention experiment (one dataset cell → long CSV).

Intervention: ``do(X_S = c)`` on an entire SCC ``S`` (the intervention
discussed in §4.3 — clamping a whole SCC removes every feedback loop
through it), with standardised clamp values ``c_j = μ̂_j + σ̂_j`` from the
observational sample. The compared quantity is the causal effect
``Δμ = μ^{do} − μ^{obs}`` on each true downstream SCC's block average.

Arms (block-effect rows, one per downstream true SCC):
  oracle       — true partition + true condensation edges, block mechanisms
                 estimated by OLS from the observational sample. Isolates
                 causal-effect estimation from structure discovery.
  hungarian    — partition + condensation recovered by the single Hungarian
                 representative (Algorithm 1's polynomial-time path):
                 end-to-end performance of the proposed method.
  first_stable — same, with the enumeration + first-stable representative
                 (secondary comparison).

Structural validity (strict, dataset-level, recovered arms): a run is
structurally valid iff (i) the recovered SCC partition equals the true
partition exactly AND (ii) the recovered condensation edge set equals the
true one IN FULL. Restricting (ii) to target→outcome ancestral edges was
rejected: a spurious estimated path through an off-path block changes the
propagation and would go undetected. Under the full-DAG criterion a valid
recovered arm coincides with the oracle computation by construction.
Invalid rows keep NaN effects when the partition is wrong (block averages
are not comparable — no silent projection onto the true partition); when
only edges are wrong the effect is still computed but flagged invalid, so
conditional effect error and the structural-recovery rate can be reported
separately. Validity is constant across a dataset's rows — report recovery
per dataset and aggregate effect errors within a dataset before any
cross-dataset statistics.

Micro-level diagnostic (per cell, replicated on each row): variable-level
Δμ computed from two observationally equivalent representatives on the
SAME FastICA estimate (first-stable vs one random admissible draw) —
quantifies how far variable-level intervention conclusions drift across
the equivalence class while the block-level computation needs no
representative at all.
"""

import networkx as nx
import numpy as np
import pandas as pd

from repare_cycle.intervention import (
    hard_intervention_means,
    observational_means,
    propagate_block_intervention,
)
from repare_cycle.lingd import (
    choose_enumerated_candidate,
    enumerate_admissible_candidates,
    fit_ica_unmixing,
    hungarian_candidate,
)

regime = snakemake.wildcards.regime
samp_size = int(snakemake.wildcards.samp_size)
seed = int(snakemake.wildcards.seed)
threshold = float(getattr(snakemake.params, "threshold", 0.1))
max_iter = int(getattr(snakemake.params, "max_iter", 10_000))
MIN_THRESHOLD_W = 0.01

data = np.load(snakemake.input.data, allow_pickle=True)
weights = data["weights"]          # row convention: A[i, j] means i → j
noise_means = data["noise_means"]
obs = data["obs"]
d = weights.shape[0]

base_row = {
    "regime": regime,
    "samp_size": samp_size,
    "seed": seed,
    "d": d,
}


def _partition_and_edges(adj_ij):
    """SCC partition (sorted for a stable indexing) + block edges of a
    variable-level adjacency in i→j convention."""
    g = nx.DiGraph()
    g.add_nodes_from(range(adj_ij.shape[0]))
    g.add_edges_from(zip(*np.nonzero(adj_ij)))
    partition = sorted(
        (frozenset(s) for s in nx.strongly_connected_components(g)),
        key=lambda s: sorted(s),
    )
    var_to_block = {v: b for b, blk in enumerate(partition) for v in blk}
    edges = {
        (var_to_block[i], var_to_block[j])
        for i, j in zip(*np.nonzero(adj_ij))
        if var_to_block[i] != var_to_block[j]
    }
    return partition, edges


true_adj = (weights != 0).astype(int)
np.fill_diagonal(true_adj, 0)
true_partition, true_edges = _partition_and_edges(true_adj)
g_true = nx.DiGraph()
g_true.add_nodes_from(range(len(true_partition)))
g_true.add_edges_from(true_edges)

# ── Target SCC: largest non-trivial true SCC with ≥1 downstream block ────
candidates_blocks = [
    b for b in range(len(true_partition))
    if len(true_partition[b]) > 1 and nx.descendants(g_true, b)
]
if not candidates_blocks:
    pd.DataFrame([dict(base_row, skipped=1)]).to_csv(
        snakemake.output[0], index=False
    )
    raise SystemExit(0)

target_block = max(
    candidates_blocks, key=lambda b: (len(true_partition[b]), -b)
)
target_vars = np.array(sorted(true_partition[target_block]), dtype=int)
downstream_blocks = sorted(nx.descendants(g_true, target_block))
downstream_vars = np.array(
    sorted(v for b in downstream_blocks for v in true_partition[b]), dtype=int
)

# ── Standardised clamp + analytic ground truth ───────────────────────────
c = obs[:, target_vars].mean(axis=0) + obs[:, target_vars].std(axis=0)
mu_do_true = hard_intervention_means(weights, noise_means, target_vars, c)
delta_true = mu_do_true - observational_means(weights, noise_means)

obs_mean = obs.mean(axis=0)


def _block_effects(mu_do_est):
    """Estimated Δμ block averages for each downstream true block."""
    delta_est = mu_do_est - obs_mean
    return {
        b: float(delta_est[np.array(sorted(true_partition[b]))].mean())
        for b in downstream_blocks
    }


def _effect_rows(arm, block_eff, valid_by_block, partition_exact):
    rows = []
    for b in downstream_blocks:
        bvars = np.array(sorted(true_partition[b]))
        t_b = float(delta_true[bvars].mean())
        sigma_b = float(obs[:, bvars].mean(axis=1).std())
        e_b = block_eff.get(b, float("nan")) if block_eff else float("nan")
        abs_err = abs(e_b - t_b) if np.isfinite(e_b) else float("nan")
        # Pre-registered normalisation: block effects can nearly cancel, so
        # the denominator is floored at 0.1 × the block-average sd.
        norm_err = (
            abs_err / max(abs(t_b), 0.1 * sigma_b)
            if np.isfinite(abs_err) else float("nan")
        )
        rows.append(dict(
            base_row,
            skipped=0,
            arm=arm,
            target_block_size=len(target_vars),
            n_downstream_blocks=len(downstream_blocks),
            outcome_block=b,
            outcome_block_size=len(true_partition[b]),
            effect_true=t_b,
            effect_est=e_b,
            abs_err=abs_err,
            norm_err=norm_err,
            partition_exact=partition_exact,
            structurally_valid=valid_by_block[b],
        ))
    return rows


rows = []

# ── Arm: oracle condensation ─────────────────────────────────────────────
mu_do_oracle = propagate_block_intervention(
    obs, true_partition, true_edges, target_block, c
)
rows += _effect_rows(
    "oracle",
    _block_effects(mu_do_oracle),
    {b: True for b in downstream_blocks},
    partition_exact=True,
)

# ── Shared FastICA estimate for the recovered arms + micro diagnostic ────
W, _ = fit_ica_unmixing(
    obs, ica_max_iter=max_iter, ica_tolerance=1e-6, random_state=0
)


def _b_recovered(strategy):
    """Weighted column-convention B̂ for a selection strategy, or None."""
    if W is None:
        return None
    if strategy == "hungarian":
        thr = threshold
        while True:
            B = hungarian_candidate(W, threshold_b=threshold, threshold_w=thr)
            if B is not None or thr <= MIN_THRESHOLD_W:
                return B
            thr /= 2
    thr = threshold
    while True:
        cands = enumerate_admissible_candidates(
            W, threshold_b=threshold, threshold_w=thr
        )
        B, _stable = choose_enumerated_candidate(
            cands,
            pick_strategy=(
                "random_admissible" if strategy == "random" else "first_stable"
            ),
            random_state=seed,
        )
        if B is not None or thr <= MIN_THRESHOLD_W:
            return B
        thr /= 2


def _recovered_arm(arm_name, B_col):
    """Rows for a recovered-structure arm from a column-convention B̂."""
    if B_col is None:
        rows_arm = _effect_rows(
            arm_name, {}, {b: False for b in downstream_blocks},
            partition_exact=False,
        )
        return rows_arm
    est_adj = (np.abs(B_col.T) > 0).astype(int)   # transpose → i→j convention
    np.fill_diagonal(est_adj, 0)
    est_partition, est_edges = _partition_and_edges(est_adj)

    partition_exact = set(est_partition) == set(true_partition)
    if not partition_exact:
        # Split/merged target or outcome blocks: block averages are not
        # comparable — structural failure, no effect is reported.
        return _effect_rows(
            arm_name, {}, {b: False for b in downstream_blocks},
            partition_exact=False,
        )

    # Partitions match as sets; re-index estimated edges onto the true
    # block indexing (frozensets are identical objects across the two).
    est_block_of = {blk: i for i, blk in enumerate(est_partition)}
    remap = {
        est_block_of[blk]: i for i, blk in enumerate(true_partition)
    }
    est_edges_true_idx = {(remap[u], remap[v]) for u, v in est_edges}

    # Strict dataset-level validity: the ENTIRE condensation edge set must
    # match (an ancestral-subgraph-only check would miss spurious estimated
    # paths through off-path blocks that alter the propagation).
    dag_exact = est_edges_true_idx == true_edges
    valid_by_block = {b: dag_exact for b in downstream_blocks}

    mu_do_est = propagate_block_intervention(
        obs, true_partition, est_edges_true_idx, target_block, c
    )
    return _effect_rows(
        arm_name, _block_effects(mu_do_est), valid_by_block,
        partition_exact=True,
    )


rows += _recovered_arm("hungarian", _b_recovered("hungarian"))
rows += _recovered_arm("first_stable", _b_recovered("first_stable"))

# ── Micro-level representative-disagreement diagnostic ───────────────────
def _micro_delta(B_col):
    """Variable-level Δμ̂ under do(X_S=c) from one representative:
    Â = B̂ᵀ (transpose column → row convention), μ̂_ε = μ̂_X (I − Â)."""
    if B_col is None:
        return None
    A_hat = B_col.T
    try:
        mu_eps_hat = obs_mean @ (np.eye(d) - A_hat)
        mu_do_hat = hard_intervention_means(
            A_hat, mu_eps_hat, target_vars, c
        )
    except np.linalg.LinAlgError:
        return None
    return mu_do_hat - obs_mean


delta_fs = _micro_delta(_b_recovered("first_stable"))
delta_rnd = _micro_delta(_b_recovered("random"))
truth_norm = float(np.linalg.norm(delta_true[downstream_vars]))
denom = max(truth_norm, 1e-9)

micro = {
    "micro_disagreement_l2": (
        float(np.linalg.norm(
            (delta_fs - delta_rnd)[downstream_vars]
        )) / denom
        if delta_fs is not None and delta_rnd is not None else float("nan")
    ),
    "micro_err_first_stable": (
        float(np.linalg.norm(
            (delta_fs - delta_true)[downstream_vars]
        )) / denom
        if delta_fs is not None else float("nan")
    ),
    "micro_err_random": (
        float(np.linalg.norm(
            (delta_rnd - delta_true)[downstream_vars]
        )) / denom
        if delta_rnd is not None else float("nan")
    ),
}
for row in rows:
    row.update(micro)

pd.DataFrame(rows).to_csv(snakemake.output[0], index=False)
