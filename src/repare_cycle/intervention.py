"""Whole-SCC hard interventions and block-level effect estimation.

Conventions
-----------
Everything here uses the ROW convention of ``CyclicLinearSEM``:
``A[i, j]`` is the weight of edge ``i → j`` and samples are row vectors,
``X = N (I − A)^{-1}``. LiNG-D's ``B_chosen`` is in the COLUMN convention
(``B[i, j] ≠ 0 ⇔ j → i``) — transpose it before calling into this module.

Hard intervention on a whole SCC
--------------------------------
``do(X_S = c)`` clamps every variable in SCC ``S``. The remaining
variables ``R = V ∖ S`` still satisfy their equilibrium equations
``X_R = X_S A[S, R] + X_R A[R, R] + N_R``, so

    μ_R^{do(X_S = c)} = (μ_{ε,R} + c A[S, R]) (I − A[R, R])^{-1},

which is well-defined whenever ``I − A[R, R]`` is invertible (no ρ < 1
stability requirement). The paper studies whole-SCC interventions because
they define the relevant cluster-level estimand. Interventions on strict
subsets of an SCC may also be mathematically solvable when the corresponding
reduced system is invertible, but they require within-SCC mechanisms and are
outside the identification claim evaluated here.

Block-level effect estimation
-----------------------------
Given a partition into blocks and an acyclic block-level DAG, each block
satisfies ``X_C = (X_{pa(C)} A[pa(C), C] + N_C)(I − A[C, C])^{-1}`` — i.e.
``X_C`` is exactly linear in its parent blocks plus an additive noise term
independent of them. OLS of ``X_C`` on ``X_{pa(C)}`` therefore consistently
estimates the block mechanism WITHOUT choosing any variable-level
representative, and expected values propagate exactly through the block
DAG in topological order. This holds in the unstable regime too (only
invertibility of ``I − A[C, C]`` is needed).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

import networkx as nx
import numpy as np


def observational_means(weights: np.ndarray, noise_means: np.ndarray) -> np.ndarray:
    """``μ = μ_ε (I − A)^{-1}`` (row convention)."""
    d = weights.shape[0]
    return np.asarray(noise_means) @ np.linalg.inv(np.eye(d) - weights)


def hard_intervention_means(
    weights: np.ndarray,
    noise_means: np.ndarray,
    target: Collection[int],
    c: np.ndarray,
) -> np.ndarray:
    """Analytic post-intervention mean vector under ``do(X_target = c)``.

    Parameters
    ----------
    weights : ``(d, d)`` row-convention weight matrix (``A[i, j]``: i → j).
    noise_means : ``(d,)`` noise means ``μ_ε``.
    target : indices of the clamped variables (a whole SCC).
    c : ``(len(target),)`` clamp values, ordered like ``sorted(target)``.

    Returns
    -------
    ``(d,)`` vector: ``c`` on the target coordinates and
    ``(μ_{ε,R} + c A[S, R])(I − A[R, R])^{-1}`` on the rest.
    """
    A = np.asarray(weights, dtype=float)
    mu_eps = np.asarray(noise_means, dtype=float)
    d = A.shape[0]
    S = np.array(sorted(target), dtype=int)
    c_arr = np.asarray(c, dtype=float)
    if c_arr.shape != S.shape:
        raise ValueError(f"c has shape {c_arr.shape}, expected {S.shape}.")
    R = np.array([i for i in range(d) if i not in set(S.tolist())], dtype=int)

    mu = np.empty(d)
    mu[S] = c_arr
    if R.size:
        A_SR = A[np.ix_(S, R)]
        A_RR = A[np.ix_(R, R)]
        mu[R] = (mu_eps[R] + c_arr @ A_SR) @ np.linalg.inv(
            np.eye(R.size) - A_RR
        )
    return mu


def simulate_hard_intervention(
    sem,
    target: Collection[int],
    c: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Monte-Carlo check of ``hard_intervention_means``: draw the SEM's
    noise, clamp ``X_target = c`` and solve the reduced ``R``-system per
    sample. Independent of the analytic mean formula's derivation path
    (uses sampled noise, not ``μ_ε``)."""
    A = sem.weights
    d = A.shape[0]
    S = np.array(sorted(target), dtype=int)
    R = np.array([i for i in range(d) if i not in set(S.tolist())], dtype=int)
    noise = sem._sample_noise(n_samples)

    X = np.empty((n_samples, d))
    X[:, S] = np.asarray(c, dtype=float)[None, :]
    if R.size:
        A_SR = A[np.ix_(S, R)]
        A_RR = A[np.ix_(R, R)]
        X[:, R] = (noise[:, R] + np.asarray(c) @ A_SR) @ np.linalg.inv(
            np.eye(R.size) - A_RR
        )
    return X


def block_dag(
    partition: list[frozenset],
    edges: Iterable[tuple[int, int]],
) -> nx.DiGraph:
    """Block-level DAG over partition indices; raises if cyclic."""
    g = nx.DiGraph()
    g.add_nodes_from(range(len(partition)))
    g.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Block-level graph must be acyclic.")
    return g


def propagate_block_intervention(
    obs: np.ndarray,
    partition: list[frozenset],
    edges: Iterable[tuple[int, int]],
    target_block: int,
    c: np.ndarray,
) -> np.ndarray:
    """Predicted post-intervention means via block-wise OLS + propagation.

    Fits ``X_C = α_C + X_{pa(C)} Γ_C + η_C`` for every block from
    observational data, clamps the target block at ``c`` and propagates
    expected values through the block DAG in topological order. No
    variable-level weight matrix is ever chosen.

    Parameters
    ----------
    obs : ``(n, d)`` observational sample.
    partition : list of frozensets of variable indices (the blocks).
    edges : block-level edges as pairs of partition indices.
    target_block : partition index of the clamped block.
    c : clamp values ordered like ``sorted(partition[target_block])``.

    Returns
    -------
    ``(d,)`` vector of predicted post-intervention means. Blocks upstream
    of (or disconnected from) the target keep their observational means
    (OLS fitted values at parent sample means reproduce sample means).
    """
    obs = np.asarray(obs, dtype=float)
    g = block_dag(partition, edges)

    mu_hat = np.empty(obs.shape[1])
    for b in nx.topological_sort(g):
        block_vars = np.array(sorted(partition[b]), dtype=int)
        if b == target_block:
            mu_hat[block_vars] = np.asarray(c, dtype=float)
            continue
        parents = sorted(g.predecessors(b))
        if not parents:
            mu_hat[block_vars] = obs[:, block_vars].mean(axis=0)
            continue
        parent_vars = np.array(
            sorted(v for p in parents for v in partition[p]), dtype=int
        )
        X = np.column_stack(
            [np.ones(obs.shape[0]), obs[:, parent_vars]]
        )
        Y = obs[:, block_vars]
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        mu_hat[block_vars] = (
            np.concatenate([[1.0], mu_hat[parent_vars]]) @ coef
        )
    return mu_hat


def ancestral_subgraph_nodes(
    g: nx.DiGraph, source: int, outcome: int
) -> set[int]:
    """Nodes on directed paths from ``source`` to ``outcome`` (inclusive):
    descendants of ``source`` ∩ ancestors of ``outcome``, plus endpoints."""
    desc = nx.descendants(g, source) | {source}
    anc = nx.ancestors(g, outcome) | {outcome}
    return desc & anc


def partitions_equal(
    p1: Iterable[Collection[int]], p2: Iterable[Collection[int]]
) -> bool:
    return {frozenset(b) for b in p1} == {frozenset(b) for b in p2}
