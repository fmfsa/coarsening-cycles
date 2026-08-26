"""Internal bridge between repare_cycle and The Virtual Brain (TVB).

NOT exported from ``repare_cycle.__init__`` — importing this module never
imports TVB. All TVB-touching functions go through :func:`_require_tvb`,
which raises an actionable error when the optional dependency is missing
(``pip install -e ".[tvb]"``). The pure helpers in the top half of the
module (orientation conversion, exact equilibria, descendant-reachability
scores, connectome threshold statistics, and the Reduced Wong-Wang
equilibrium math used by Experiment 2) only need numpy/networkx and are
unit-tested without TVB.

Orientation contract
--------------------
Repository truth (graph.py, all generators, CyclicLinearSEM):

    weights[i, j] != 0  ⇔  edge i → j        ("row = source")
    X (n × d) = E @ (I - W)^{-1}             (rows are samples)

TVB connectivity (tvb.datatypes.connectivity.Connectivity, verified in
tvb-library 2.10.0 ``coupling.SparseCoupling``): coupling into region k is
``Σ_j weights[k, j] · x_j``, i.e.

    weights_tvb[target, source]              ("row = target, 'to-from'")

Conversion between the two is therefore a plain transpose, in both
directions.

Equilibrium derivation
----------------------
The bridge runs ``models.Linear(gamma=-1)`` with linear coupling ``a=1``,
zero conduction delays, and a constant regional stimulus ``e``:

    dx/dt = -x + W_tvb x + e   ⇒   x* = (I - W_tvb)^{-1} e,

which is the column form of the repository equilibrium ``x = e (I - W)^{-1}``
exactly when ``W_tvb = W.T``. With deterministic Heun and zero delays the
equilibrium is an exact fixed point of TVB's discrete update (at ``x*`` the
drift equals ``-e``, cancelling the stimulus term), so the 1e-7 residual
target below is reachable, not merely approachable.

Integration-step choice: the Jacobian ``-I + W_tvb`` has eigenvalues in the
disk of radius ρ(W) ≤ 0.9 centred at -1, so Re λ ∈ [-1.9, -0.1]. At
``dt = 0.0625`` ms, |λ·dt| ≤ 0.12 sits deep inside Heun's stability region,
and the slowest mode decays like exp(-0.1 t), reaching 1e-7 within the
200 ms budget. ``dt`` and the chunk length are exact binary fractions so
each chunk is an integer number of steps.

Float32 history caveat (pilot finding)
--------------------------------------
Stock TVB 2.10 stores the simulator's history buffer, delayed states, and
per-connection weights in float32 (``NDArray(..., 'f')`` descriptors in
``tvb/simulator/history.py``), while every other part of the pipeline runs
in float64. The coupling term is therefore float32-quantized, which shifts
the simulated fixed point by ~1e-6 *relative* — the residual plateaus
there, above this pilot's 1e-7 target, no matter how long one integrates.
:func:`simulate_to_equilibrium` therefore swaps in a float64 subclass of
``SparseHistory`` right after ``Simulator.configure()``; the subclass only
widens the buffer dtypes and changes no logic, after which the equilibrium
residual decays to the float64 floor as predicted.
"""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Pure helpers — importable and testable WITHOUT tvb installed.
# ---------------------------------------------------------------------------


def to_tvb_orientation(weights: np.ndarray) -> np.ndarray:
    """Convert repository ``weights[src, dst]`` to TVB ``weights[to, from]``."""
    return np.asarray(weights, dtype=float).T.copy()


def from_tvb_orientation(weights_tvb: np.ndarray) -> np.ndarray:
    """Convert TVB ``weights[to, from]`` to repository ``weights[src, dst]``."""
    return np.asarray(weights_tvb, dtype=float).T.copy()


def exact_equilibrium(weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """Exact linear-SEM equilibrium ``X = E @ (I - W)^{-1}``.

    Parameters
    ----------
    weights : (d, d) array in repository orientation (weights[src, dst]).
    inputs : (d,) or (n, d) array of constant inputs (noise/stimulus rows).

    Returns
    -------
    Equilibrium state(s), same shape as *inputs*.
    """
    weights = np.asarray(weights, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    d = weights.shape[0]
    lhs = np.eye(d) - weights
    # X (I - W) = E  ⇔  (I - W)^T X^T = E^T
    solution = np.linalg.solve(lhs.T, np.atleast_2d(inputs).T).T
    return solution[0] if inputs.ndim == 1 else solution


def reconstruct_inputs(weights: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Invert :func:`exact_equilibrium`: ``E = X @ (I - W)``.

    Also doubles as the equilibrium residual: for a trajectory sample ``x``,
    ``inputs - reconstruct_inputs(weights, x)`` equals the drift
    ``gamma·x + W_tvb x + e`` of the bridge ODE (in row form), so its
    max-norm is the convergence residual used by
    :func:`simulate_to_equilibrium`.
    """
    weights = np.asarray(weights, dtype=float)
    states = np.asarray(states, dtype=float)
    return states @ (np.eye(weights.shape[0]) - weights)


def descendant_reachability_scores(
    true_adj_ij: np.ndarray,
    pred_adj_ij: np.ndarray,
    true_scc_labels: np.ndarray,
) -> dict:
    """Cross-SCC descendant precision/recall/F1 over ordered variable pairs.

    Both adjacencies are (d, d) binary matrices in i→j convention. The score
    compares reachability (transitive closure) in the true and predicted
    variable digraphs, restricted to ordered pairs (u, v) whose endpoints lie
    in *different TRUE SCCs* — so no alignment between estimated and true
    cluster labels is ever needed.

    This is equivalent to taking the transitive closures of the true and
    estimated condensations and expanding them to ordered variable pairs:
    the estimated clusters are exactly the SCCs of the predicted variable
    graph, and reachability between variables factors through their SCCs.
    A consequence worth noting: when the estimate merges two true SCCs into
    one cluster, both directions between them count as predicted-reachable
    (mutual reachability inside a cluster), so at most one of the two can be
    a true positive — over-merging costs precision, as intended.

    Empty-denominator conventions match the ``var_*`` metrics in
    ``evaluate_synth.py``: precision/recall default to 1.0 when their
    denominator is empty; F1 is 0.0 when precision + recall == 0.
    """
    true_adj_ij = np.asarray(true_adj_ij)
    pred_adj_ij = np.asarray(pred_adj_ij)
    true_scc_labels = np.asarray(true_scc_labels)
    d = true_adj_ij.shape[0]

    def _reachability(adj: np.ndarray) -> np.ndarray:
        graph = nx.DiGraph()
        graph.add_nodes_from(range(d))
        rows, cols = np.where(adj > 0)
        graph.add_edges_from(zip(rows.tolist(), cols.tolist()))
        reach = np.zeros((d, d), dtype=bool)
        for u, descendants in nx.all_pairs_shortest_path_length(graph):
            for v in descendants:
                reach[u, v] = True
        np.fill_diagonal(reach, False)
        return reach

    reach_true = _reachability(true_adj_ij)
    reach_pred = _reachability(pred_adj_ij)

    cross_mask = true_scc_labels[:, None] != true_scc_labels[None, :]
    true_pos_pairs = reach_true & cross_mask
    pred_pos_pairs = reach_pred & cross_mask

    tp = int(np.sum(true_pos_pairs & pred_pos_pairs))
    n_pred = int(np.sum(pred_pos_pairs))
    n_true = int(np.sum(true_pos_pairs))

    precision = tp / n_pred if n_pred > 0 else 1.0
    recall = tp / n_true if n_true > 0 else 1.0
    fscore = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "desc_precision": precision,
        "desc_recall": recall,
        "desc_fscore": fscore,
        "n_desc_pairs": int(np.sum(cross_mask)),
        "n_desc_true": n_true,
        "n_desc_pred": n_pred,
    }


def condensation_from_adjacency(adj_ij: np.ndarray) -> nx.DiGraph:
    """Condensation of a variable digraph, with frozenset-of-members nodes.

    Matches the C-DAG format produced by the fit pipeline (``model.dag``):
    each node is the frozenset of variable indices in one SCC, and edges
    connect SCCs with at least one variable-level edge between them.
    """
    adj_ij = np.asarray(adj_ij)
    d = adj_ij.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(d))
    rows, cols = np.where(adj_ij > 0)
    graph.add_edges_from((u, v) for u, v in zip(rows.tolist(), cols.tolist())
                         if u != v)
    cond = nx.condensation(graph)
    members = {c: frozenset(cond.nodes[c]["members"]) for c in cond.nodes}
    cdag = nx.DiGraph()
    cdag.add_nodes_from(members.values())
    cdag.add_edges_from((members[u], members[v]) for u, v in cond.edges)
    return cdag


def cdag_direct_scores(true_cdag: nx.DiGraph, est_cdag: nx.DiGraph) -> dict:
    """Direct true-vs-estimated C-DAG comparison, no projection involved.

    Both arguments are condensations whose nodes are frozensets of variable
    indices (:func:`condensation_from_adjacency` / ``model.dag``). An
    estimated edge counts as correct only when BOTH endpoint clusters match
    a true SCC exactly — unlike the projected cluster-DAG F1 of
    ``evaluate_synth.py``, which can be nonzero even when the estimated
    C-DAG is a single node with no edges. Empty-denominator conventions as
    elsewhere (precision/recall default 1.0; F1 is 0.0 when p+r == 0).
    """
    true_edges = set(true_cdag.edges)
    est_edges = set(est_cdag.edges)
    tp = len(true_edges & est_edges)
    precision = tp / len(est_edges) if est_edges else 1.0
    recall = tp / len(true_edges) if true_edges else 1.0
    fscore = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "cdag_precision": precision,
        "cdag_recall": recall,
        "cdag_fscore": fscore,
        "partition_exact": int(set(true_cdag.nodes) == set(est_cdag.nodes)),
        "n_cdag_edges_true": len(true_edges),
        "n_cdag_edges_est": len(est_edges),
    }


def predicted_descendant_variables(cdag: nx.DiGraph, stimulated) -> frozenset:
    """Variables the C-DAG predicts to respond to stimulating *stimulated*.

    Prediction: every cluster that intersects the stimulated set responds
    as a whole (mutual influence inside a cluster), and so does every
    cluster reachable from one of them in the C-DAG. Returns the union of
    those clusters' members as a frozenset of variable indices.
    """
    stimulated = frozenset(stimulated)
    responders: set = set()
    for cluster in cdag.nodes:
        if cluster & stimulated:
            responders |= set(cluster)
            for downstream in nx.descendants(cdag, cluster):
                responders |= set(downstream)
    return frozenset(responders)


def connectome_threshold_stats(
    weights: np.ndarray,
    quantiles: tuple = (0.0, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99),
) -> list[dict]:
    """Descriptive SCC statistics of a weighted connectome under thresholding.

    For each quantile *q* of the nonzero |weight| distribution, edges with
    |weight| >= quantile value are retained and the directed graph they form
    is summarised. Purely descriptive — the connectome is never oriented or
    otherwise modified.

    Parameters
    ----------
    weights : (d, d) array in repository orientation (weights[src, dst]);
        the diagonal is ignored.
    quantiles : quantiles of the nonzero |weight| distribution to threshold at.

    Returns
    -------
    One dict per quantile with keys: quantile, threshold_value,
    total_nonzero_edges, retained_edges, retained_fraction, symmetry_error,
    reciprocity, num_sccs, num_nontrivial_sccs, largest_scc_fraction,
    n_condensation_edges.
    """
    weights = np.asarray(weights, dtype=float).copy()
    np.fill_diagonal(weights, 0.0)
    d = weights.shape[0]
    abs_w = np.abs(weights)
    nonzero_vals = abs_w[abs_w > 0]
    total_nonzero = int(nonzero_vals.size)

    rows = []
    for q in quantiles:
        threshold_value = float(np.quantile(nonzero_vals, q)) if total_nonzero else 0.0
        mask = (abs_w >= threshold_value) & (abs_w > 0)
        retained = int(np.sum(mask))

        thresholded = np.where(mask, weights, 0.0)
        norm = float(np.linalg.norm(thresholded))
        symmetry_error = (
            float(np.linalg.norm(thresholded - thresholded.T)) / norm
            if norm > 0 else 0.0
        )
        reciprocity = float(np.sum(mask & mask.T) / retained) if retained else 0.0

        graph = nx.from_numpy_array(mask.astype(int), create_using=nx.DiGraph)
        sccs = list(nx.strongly_connected_components(graph))
        rows.append({
            "quantile": float(q),
            "threshold_value": threshold_value,
            "total_nonzero_edges": total_nonzero,
            "retained_edges": retained,
            "retained_fraction": retained / total_nonzero if total_nonzero else 0.0,
            "symmetry_error": symmetry_error,
            "reciprocity": reciprocity,
            "num_sccs": len(sccs),
            "num_nontrivial_sccs": sum(1 for s in sccs if len(s) > 1),
            "largest_scc_fraction": max(len(s) for s in sccs) / d,
            "n_condensation_edges": nx.condensation(graph, sccs).number_of_edges(),
        })
    return rows


# ---------------------------------------------------------------------------
# Reduced Wong-Wang equilibrium math (pure numpy — Experiment 2).
#
# Mirrors tvb.simulator.models.wong_wang.ReducedWongWang exactly:
#     x  = w * J_N * S + I_o + J_N * c,   c_i = G * sum_j W_tvb[i,j] S_j
#     H  = (a x - b) / (1 - exp(-d (a x - b)))
#     dS = -S / tau_s + (1 - S) * H * gamma
# with repository-oriented weights (W_tvb = weights.T). Verified against
# TVB's own dfun in the optional test suite.
# ---------------------------------------------------------------------------

RWW_DEFAULTS = {
    "a": 0.270,      # nC^-1, input gain
    "b": 0.108,      # kHz, input shift
    "d": 154.0,      # ms
    "gamma": 0.641,  # kinetic parameter
    "tau_s": 100.0,  # ms, NMDA decay
    "w": 0.6,        # local excitatory recurrence
    "j_n": 0.2609,   # nA, synaptic coupling
}


def _rww_transfer(x: np.ndarray, a: float, b: float, d: float):
    """H(x) and dH/dx for the Wong-Wang input-to-rate transfer function.

    Uses the series expansion H = 1/d + u/2 + O(u^2) (with u = a x - b)
    near the removable singularity at u = 0, and clamps the exponent so
    strongly hyperpolarized inputs (u << 0, H -> 0) cannot overflow.
    """
    u = a * np.asarray(x, dtype=float) - b
    du = d * u
    exp_term = np.exp(np.clip(-du, None, 600.0))
    denom = 1.0 - exp_term
    small = np.abs(du) < 1e-6
    safe = np.where(small, 1.0, denom)
    h = np.where(small, 1.0 / d + u / 2.0, u / safe)
    dh_du = np.where(
        small,
        0.5 + du / 6.0,
        (1.0 - exp_term * (1.0 + du)) / (safe * safe),
    )
    return h, a * dh_du


def rww_dfun(
    state: np.ndarray,
    inputs: np.ndarray,
    weights: np.ndarray,
    g: float,
    params: dict | None = None,
) -> np.ndarray:
    """dS/dt of the Reduced Wong-Wang network (repository orientation).

    *state* and *inputs* (per-region external drive I_o) are length-d
    vectors — or (n, d) arrays to evaluate n independent episodes at once;
    *weights* is the repository ``weights[src, dst]`` matrix and *g* the
    global coupling scale (``coupling.Linear(a=g)`` in TVB).
    """
    p = {**RWW_DEFAULTS, **(params or {})}
    state = np.asarray(state, dtype=float)
    # (W_tvb @ s) == s @ W in repository orientation; broadcasts over rows.
    coupling_in = g * (state @ np.asarray(weights, dtype=float))
    x = p["w"] * p["j_n"] * state + np.asarray(inputs, dtype=float) \
        + p["j_n"] * coupling_in
    h, _ = _rww_transfer(x, p["a"], p["b"], p["d"])
    return -state / p["tau_s"] + (1.0 - state) * h * p["gamma"]


def rww_jacobian(
    state: np.ndarray,
    inputs: np.ndarray,
    weights: np.ndarray,
    g: float,
    params: dict | None = None,
) -> np.ndarray:
    """Analytic Jacobian d(dS_i/dt)/dS_j of :func:`rww_dfun` at *state*."""
    p = {**RWW_DEFAULTS, **(params or {})}
    state = np.asarray(state, dtype=float)
    w_tvb = np.asarray(weights, dtype=float).T
    x = p["w"] * p["j_n"] * state + np.asarray(inputs, dtype=float) \
        + p["j_n"] * g * (w_tvb @ state)
    h, dh_dx = _rww_transfer(x, p["a"], p["b"], p["d"])
    gain = (1.0 - state) * p["gamma"] * dh_dx  # d(dS_i)/d(x_i)
    jac = gain[:, None] * (p["j_n"] * g * w_tvb)
    jac[np.diag_indices_from(jac)] += (
        -1.0 / p["tau_s"] - h * p["gamma"] + gain * p["w"] * p["j_n"]
    )
    return jac


def rww_fixed_point(
    weights: np.ndarray,
    inputs: np.ndarray,
    g: float,
    start: np.ndarray | float = 0.3,
    params: dict | None = None,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> SimpleNamespace:
    """Damped-Newton fixed point of the Reduced Wong-Wang network.

    Returns SimpleNamespace(state, converged, residual, jacobian,
    max_real_eig) — *jacobian* and *max_real_eig* evaluated at the
    returned state, giving the local stability margin.
    """
    d_nodes = np.asarray(weights).shape[0]
    state = np.full(d_nodes, float(start)) if np.ndim(start) == 0 \
        else np.asarray(start, dtype=float).copy()
    residual = rww_dfun(state, inputs, weights, g, params)
    converged = False
    for _ in range(max_iter):
        if np.max(np.abs(residual)) < tol:
            converged = True
            break
        jac = rww_jacobian(state, inputs, weights, g, params)
        step = np.linalg.solve(jac, -residual)
        improved = False
        for _ in range(30):  # backtracking line search
            candidate = np.clip(state + step, 0.0, 1.0)
            cand_residual = rww_dfun(candidate, inputs, weights, g, params)
            if np.max(np.abs(cand_residual)) < np.max(np.abs(residual)):
                improved = True
                break
            step *= 0.5
        if not improved:
            break  # stuck at a non-root residual minimum: report unconverged
        state, residual = candidate, cand_residual
    jac = rww_jacobian(state, inputs, weights, g, params)
    return SimpleNamespace(
        state=state,
        converged=converged and bool(np.max(np.abs(residual)) < tol),
        residual=float(np.max(np.abs(residual))),
        jacobian=jac,
        max_real_eig=float(np.max(np.linalg.eigvals(jac).real)),
    )


def rww_equilibria(
    weights: np.ndarray,
    inputs: np.ndarray,
    g: float,
    params: dict | None = None,
    n_uniform: int = 25,
    n_random: int = 20,
    seed: int = 0,
    dedup_tol: float = 1e-6,
) -> SimpleNamespace:
    """Enumerate Reduced Wong-Wang equilibria by multi-start damped Newton.

    Newton converges to saddles as readily as to attractors, so a dense
    start set doubles as a (heuristic) multistability probe: uniform
    levels across [0.02, 0.98] plus random interior starts. Distinct
    solutions are deduplicated at *dedup_tol*.

    Returns SimpleNamespace(states (k, d), margins (k,) [-max Re eig],
    n_equilibria, unique_stable — True iff exactly one equilibrium was
    found and it is stable).
    """
    d = np.asarray(weights).shape[0]
    rng = np.random.default_rng(seed)
    starts = [np.full(d, level) for level in np.linspace(0.02, 0.98, n_uniform)]
    starts += [rng.uniform(0.02, 0.98, size=d) for _ in range(n_random)]

    states: list[np.ndarray] = []
    margins: list[float] = []
    for start in starts:
        fp = rww_fixed_point(weights, inputs, g, start=start, params=params)
        if not fp.converged:
            continue
        if any(np.max(np.abs(fp.state - s)) < dedup_tol for s in states):
            continue
        states.append(fp.state)
        margins.append(-fp.max_real_eig)

    return SimpleNamespace(
        states=np.array(states),
        margins=np.array(margins),
        n_equilibria=len(states),
        unique_stable=(len(states) == 1 and margins[0] > 0.0),
    )


def rww_linearization(
    state: np.ndarray,
    inputs: np.ndarray,
    weights: np.ndarray,
    g: float,
    params: dict | None = None,
) -> SimpleNamespace:
    """Linearize the Wong-Wang equilibrium map around *state*.

    Returns SimpleNamespace(jacobian, gain, mixing, b_col):
      gain    — d(dS_i)/d(I_i) = (1-S_i)*gamma*dH/dx, per node;
      mixing  — M = -J^{-1} diag(gain): dS* ≈ M dI, the local sensitivity
                of the fixed point to input shifts;
      b_col   — the effective linear-SEM matrix in COLUMN convention
                (b_col[i, j]: coefficient of S_j in S_i's equation),
                b_col = C / Lambda with J = -diag(Lambda) + C. Its support
                and SCC structure equal the intended graph's whenever
                gain > 0, but its entry MAGNITUDES are reshaped by the
                dynamics — the quantity that decides whether a fixed
                LiNG-D threshold is meaningful.
    """
    p = {**RWW_DEFAULTS, **(params or {})}
    state = np.asarray(state, dtype=float)
    weights = np.asarray(weights, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    x = p["w"] * p["j_n"] * state + inputs + p["j_n"] * g * (state @ weights)
    _, dh_dx = _rww_transfer(x, p["a"], p["b"], p["d"])
    gain = (1.0 - state) * p["gamma"] * dh_dx
    jac = rww_jacobian(state, inputs, weights, g, params)
    coupling_part = jac.copy()
    lam = -np.diag(jac)
    np.fill_diagonal(coupling_part, 0.0)
    return SimpleNamespace(
        jacobian=jac,
        gain=gain,
        mixing=-np.linalg.solve(jac, np.diag(gain)),
        b_col=coupling_part / lam[:, None],
    )


def rww_stable_fixed_point(
    weights: np.ndarray,
    inputs: np.ndarray,
    g: float,
    params: dict | None = None,
) -> SimpleNamespace:
    """The unique stable Wong-Wang equilibrium, or raise with diagnostics.

    Used by the generation and stimulation stages so they never silently
    warm-start from a saddle when the regime is not single-fixed-point.
    """
    eq = rww_equilibria(weights, inputs, g, params=params)
    if not eq.unique_stable:
        raise RuntimeError(
            "Reduced Wong-Wang regime is not single-stable-fixed-point: "
            f"found {eq.n_equilibria} equilibria with stability margins "
            f"{np.round(eq.margins, 5).tolist()} — recalibrate (G, I_o)."
        )
    return SimpleNamespace(state=eq.states[0], margin=float(eq.margins[0]))


# ---------------------------------------------------------------------------
# TVB-dependent half — every entry point goes through _require_tvb().
# ---------------------------------------------------------------------------

_TVB_INSTALL_HINT = (
    "The Virtual Brain is required for this feature but is not installed. "
    "Install the optional extra:  pip install -e '.[tvb]'"
)


def _require_tvb() -> SimpleNamespace:
    """Import TVB lazily; raise an actionable error when it is missing."""
    try:
        from tvb.datatypes import connectivity, equations, patterns
        from tvb.simulator import coupling, integrators, models, monitors, simulator
    except ImportError as exc:
        raise ImportError(_TVB_INSTALL_HINT) from exc
    return SimpleNamespace(
        connectivity=connectivity,
        equations=equations,
        patterns=patterns,
        coupling=coupling,
        integrators=integrators,
        models=models,
        monitors=monitors,
        simulator=simulator,
    )


_f64_history_cls = None


def _float64_sparse_history():
    """SparseHistory subclass with float64 buffers (see module docstring).

    Overrides only the dtype of the float32 ``NDArray`` descriptors that
    feed TVB's coupling computation; shapes, flags, and all logic are
    inherited unchanged from stock TVB.
    """
    global _f64_history_cls
    if _f64_history_cls is None:
        _require_tvb()
        from tvb.simulator.descriptors import NDArray
        from tvb.simulator.history import SparseHistory

        class _Float64SparseHistory(SparseHistory):
            es_weights = NDArray(("n_node", "n_cvar", "n_node", "n_mode"), "d")
            buffer = NDArray(
                ("n_time", "n_cvar", "n_node", "n_mode"), "d", read_only=False
            )
            current_state = NDArray(
                ("n_cvar", "n_node", "n_mode"), "d", read_only=False
            )
            delayed_state = NDArray(
                ("n_node", "n_cvar", "n_node", "n_mode"), "d", read_only=False
            )
            nnz_weights = NDArray((SparseHistory.n_nnzw,), "d")

        _f64_history_cls = _Float64SparseHistory
    return _f64_history_cls


def build_connectivity(weights: np.ndarray):
    """Zero-delay TVB Connectivity from repository-oriented weights.

    Tract lengths are all zero, so conduction delays vanish and the history
    horizon is a single step regardless of the propagation speed.
    """
    tvb = _require_tvb()
    weights = np.asarray(weights, dtype=float)
    d = weights.shape[0]
    conn = tvb.connectivity.Connectivity(
        weights=to_tvb_orientation(weights),
        tract_lengths=np.zeros((d, d)),
        region_labels=np.array([f"node_{i:03d}" for i in range(d)]),
        centres=np.zeros((d, 3)),
    )
    conn.configure()
    return conn


def simulate_to_equilibrium(
    weights: np.ndarray,
    inputs_row: np.ndarray,
    *,
    dt: float = 0.0625,
    chunk_ms: float = 8.0,
    max_ms: float = 200.0,
    tol: float = 1e-7,
) -> SimpleNamespace:
    """Integrate the linear TVB model under a constant stimulus to equilibrium.

    Runs ``models.Linear(gamma=-1)`` + ``coupling.Linear(a=1)`` +
    deterministic Heun from a zero initial state, in chunks of *chunk_ms*,
    until the equilibrium residual ``‖e - x (I - W)‖_∞`` (see
    :func:`reconstruct_inputs`) of the last recorded sample drops to *tol*,
    or *max_ms* of simulated time is exceeded.

    Never raises on non-convergence — the caller inspects ``converged`` and
    the returned diagnostics.

    Returns
    -------
    SimpleNamespace with fields: x ((d,) final state), converged,
    terminal_residual, settling_time_ms (first sample at or below *tol*;
    NaN if never), sim_time_ms, n_steps, residual_trace (per-chunk terminal
    residuals), runtime_sec.
    """
    import time

    tvb = _require_tvb()
    weights = np.asarray(weights, dtype=float)
    inputs_row = np.asarray(inputs_row, dtype=float).ravel()
    d = weights.shape[0]

    conn = build_connectivity(weights)
    # equations.Linear evaluates "a * var + b"; a=0, b=1 is exactly 1.0 at
    # every time point, making `weight` the constant per-region stimulus.
    stimulus = tvb.patterns.StimuliRegion(
        connectivity=conn,
        weight=inputs_row.copy(),
        temporal=tvb.equations.Linear(parameters={"a": 0.0, "b": 1.0}),
    )
    sim = tvb.simulator.Simulator(
        model=tvb.models.Linear(gamma=np.array([-1.0])),
        connectivity=conn,
        coupling=tvb.coupling.Linear(a=np.array([1.0]), b=np.array([0.0])),
        integrator=tvb.integrators.HeunDeterministic(dt=dt),
        monitors=(tvb.monitors.Raw(),),
        stimulus=stimulus,
        initial_conditions=np.zeros((1, 1, d, 1)),
    )
    sim.configure()
    # Replace the float32 history with a float64 one (module docstring:
    # stock TVB quantizes the coupling path to float32, which caps the
    # reachable residual around 1e-6 — above the tol used here).
    sim.history = _float64_sparse_history().from_simulator(sim)

    start = time.perf_counter()
    steps_done = 0
    settling_time_ms = float("nan")
    residual_trace: list[float] = []
    terminal_residual = float("inf")
    x_final = np.zeros(d)
    converged = False

    while steps_done * dt < max_ms:
        (_, raw), = sim.run(simulation_length=chunk_ms)
        states = raw[:, 0, :, 0]
        residuals = np.abs(
            inputs_row - reconstruct_inputs(weights, states)
        ).max(axis=1)
        if np.isnan(settling_time_ms):
            settled = np.nonzero(residuals <= tol)[0]
            if settled.size:
                settling_time_ms = (steps_done + int(settled[0]) + 1) * dt
        steps_done += states.shape[0]
        x_final = states[-1]
        terminal_residual = float(residuals[-1])
        residual_trace.append(terminal_residual)
        if terminal_residual <= tol:
            converged = True
            break

    return SimpleNamespace(
        x=x_final,
        converged=converged,
        terminal_residual=terminal_residual,
        settling_time_ms=settling_time_ms,
        sim_time_ms=steps_done * dt,
        n_steps=steps_done,
        residual_trace=residual_trace,
        runtime_sec=time.perf_counter() - start,
    )


def simulate_rww_ensemble(
    weights: np.ndarray,
    input_rows: np.ndarray,
    *,
    g: float,
    dt: float = 1.0,
    chunk_ms: float = 512.0,
    max_ms: float = 20_000.0,
    tol: float = 1e-7,
    init: np.ndarray | None = None,
    batch_size: int = 100,
    params: dict | None = None,
) -> SimpleNamespace:
    """Settle many independent Reduced Wong-Wang episodes with TVB.

    Each row of *input_rows* is one episode's per-region external drive
    I_o. Episodes are independent, so batches of them are stacked as one
    block-diagonal TVB simulation (``kron(I_B, W)``) — numerically
    identical to running them one at a time (the sparse coupling sums only
    nonzero weights, and block off-diagonals are exactly zero), but it
    amortizes TVB's per-step overhead across the whole batch. Deterministic
    Heun, zero delays, no noise, float64 history; an episode counts as
    converged when ``max_i |dS_i/dt| <= tol`` (residual via the pure
    :func:`rww_dfun`, so the criterion itself is unit-tested without TVB).

    *init* warm-starts every episode (length-d vector, e.g. the baseline
    fixed point) or each episode individually ((n, d) array).

    Returns SimpleNamespace(states (n, d), converged (n,), residuals (n,),
    sim_time_ms (n,), runtime_sec).
    """
    import time

    tvb = _require_tvb()
    weights = np.asarray(weights, dtype=float)
    input_rows = np.atleast_2d(np.asarray(input_rows, dtype=float))
    n_eps, d = input_rows.shape
    if init is None:
        init_rows = np.full((n_eps, d), 0.3)
    else:
        init = np.asarray(init, dtype=float)
        init_rows = np.broadcast_to(init, (n_eps, d)).copy()
    p = {**RWW_DEFAULTS, **(params or {})}

    states = np.zeros((n_eps, d))
    converged = np.zeros(n_eps, dtype=bool)
    residuals = np.full(n_eps, np.inf)
    sim_times = np.zeros(n_eps)
    started = time.perf_counter()

    for lo in range(0, n_eps, batch_size):
        hi = min(lo + batch_size, n_eps)
        nb = hi - lo
        conn = build_connectivity(np.kron(np.eye(nb), weights))
        model = tvb.models.ReducedWongWang(
            a=np.array([p["a"]]), b=np.array([p["b"]]),
            d=np.array([p["d"]]), gamma=np.array([p["gamma"]]),
            tau_s=np.array([p["tau_s"]]), w=np.array([p["w"]]),
            J_N=np.array([p["j_n"]]),
            I_o=input_rows[lo:hi].ravel(),  # spatialized per-region drive
        )
        sim = tvb.simulator.Simulator(
            model=model,
            connectivity=conn,
            coupling=tvb.coupling.Linear(a=np.array([float(g)]),
                                         b=np.array([0.0])),
            integrator=tvb.integrators.HeunDeterministic(dt=dt),
            monitors=(tvb.monitors.Raw(),),
            initial_conditions=(
                init_rows[lo:hi].ravel()[None, None, :, None].copy()
            ),
        )
        sim.configure()
        sim.history = _float64_sparse_history().from_simulator(sim)

        steps = 0
        s_rows = init_rows[lo:hi]
        batch_res = np.full(nb, np.inf)
        while steps * dt < max_ms:
            (_, raw), = sim.run(simulation_length=chunk_ms)
            steps += raw.shape[0]
            s_rows = raw[-1, 0, :, 0].reshape(nb, d)
            batch_res = np.max(np.abs(
                rww_dfun(s_rows, input_rows[lo:hi], weights, g, params)
            ), axis=1)
            if np.max(batch_res) <= tol:
                break
        states[lo:hi] = s_rows
        converged[lo:hi] = batch_res <= tol
        residuals[lo:hi] = batch_res
        sim_times[lo:hi] = steps * dt

    return SimpleNamespace(
        states=states,
        converged=converged,
        residuals=residuals,
        sim_time_ms=sim_times,
        runtime_sec=time.perf_counter() - started,
    )


def simulate_rww_to_equilibrium(
    weights: np.ndarray,
    inputs_row: np.ndarray,
    *,
    g: float,
    **kwargs,
) -> SimpleNamespace:
    """Single-episode convenience wrapper around :func:`simulate_rww_ensemble`."""
    result = simulate_rww_ensemble(
        weights, np.atleast_2d(inputs_row), g=g, batch_size=1, **kwargs
    )
    return SimpleNamespace(
        x=result.states[0],
        converged=bool(result.converged[0]),
        terminal_residual=float(result.residuals[0]),
        sim_time_ms=float(result.sim_time_ms[0]),
        runtime_sec=result.runtime_sec,
    )


def load_default_connectivity():
    """Load TVB's default 76-region connectome (needs the tvb-data package)."""
    tvb = _require_tvb()
    try:
        conn = tvb.connectivity.Connectivity.from_file()
    except (ImportError, OSError) as exc:
        raise ImportError(_TVB_INSTALL_HINT) from exc
    conn.configure()
    return conn
