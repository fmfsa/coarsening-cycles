"""Pure-Python ICA-LiNG-D (Lacerda et al., UAI 2008).

We re-implement Tetrad's LingD on top of scikit-learn's FastICA. Tetrad's
own implementation is excellent in spec but its bundled FastICA throws
``Eigenvalue Decomposition failed`` on a non-trivial fraction of our
synthetic cyclic-SEM datasets, even with retries and parameter rotation.
sklearn's FastICA is empirically much more robust on the same inputs.

Algorithm
---------
1. Estimate the unmixing matrix ``W`` via FastICA, so ``S ≈ (X - mean(X)) Wᵀ``
   with mutually independent ``S`` columns.
2. **N-rooks permutation search.** Enumerate column permutations ``π`` of
   ``W`` such that every diagonal entry of ``Wπ := W[:, π]`` exceeds
   ``threshold_w`` in magnitude. Each such ``π`` is a member of the
   distributional equivalence class of the data.
3. For every ``Wπ``, normalise the diagonal to one and read off
   ``B̂ = I − diag(Wπ)⁻¹ · Wπ``. Threshold ``|B̂|`` at ``threshold_b``.
4. Filter into **stable** (``ρ(B̂) < 1``) and **unstable** (``ρ(B̂) ≥ 1``).

Returns column-convention B̂ (``B[i, j] != 0  ⇔  j → i``), matching the
rest of ``repare_cycle``. By the condensation-identifiability theorem in
the paper, every member of the equivalence class has the same
condensation, so any representative gives the same cluster DAG.

Stage functions
---------------
The pipeline is factored into four composable stages so that experiment
code can hold the ICA estimate fixed while varying candidate selection:

``fit_ica_unmixing``
    FastICA only — returns the unmixing matrix ``W`` (or ``None``).

``enumerate_admissible_candidates``
    N-rooks search over ``W`` — returns stable/unstable candidate lists
    plus enumeration metadata (count, cap hit).

``choose_enumerated_candidate``
    Selection from an enumerated candidate set: ``first_stable`` or
    ``random_admissible``.

``hungarian_candidate``
    A single admissible representative via linear assignment in
    ``O(d^3)``. Performs **zero** equivalence-class enumeration.

``run_lingd`` composes the stages and is behaviour-compatible with the
original monolithic implementation.

Three pick strategies are supported (see ``run_lingd``):

``"first_stable"`` (default)
    Choose the first stable candidate (``ρ < 1``).  For sparse cyclic SEMs
    this uniquely recovers the true B because the N-rooks reversed-cycle
    permutation always has ``ρ = 1/ρ_true > 1``.

``"random_admissible"``
    Draw uniformly from **all** admissible permutations (stable + unstable).
    Deliberately samples the full distributional equivalence class,
    demonstrating that condensation is invariant across all members while
    the variable-level edge set varies — the identifiability gap.

``"hungarian_any"``
    Find a single admissible permutation by solving a linear-assignment
    problem on the cost matrix ``C[i, k] = -log(|W[i, k]|)``, with entries
    below ``threshold_w`` ruled out.  Avoids the worst-case ``O(d!)``
    N-rooks enumeration: returns *one* admissible representative in
    ``O(d^3)``.  Sufficient when only the condensation graph is needed
    (Theorem 1 in the paper), since every admissible representative yields
    the same condensation.
"""

from __future__ import annotations

import time
from itertools import islice
from typing import Iterator

import numpy as np


def _nrooks_permutations(
    W: np.ndarray,
    threshold_w: float,
    deadline: float | None = None,
) -> Iterator[np.ndarray]:
    """Yield ``W[π, :]`` for every row permutation ``π`` of ``W`` whose
    diagonal is zeroless: ``|W[π[k], k]| > threshold_w`` for all ``k``.

    sklearn's FastICA returns ``W = components_`` with rows indexed by ICA
    sources in an arbitrary order. LiNG-D recovers the variable-aligned
    unmixing matrix by row-permuting ``W`` so that the source explaining
    ``x_k``'s noise lands in row ``k`` — equivalently, the diagonal of the
    permuted matrix is zeroless. Each admissible ``π`` is one member of
    the distributional equivalence class.

    Bound the result via ``islice`` at the call site if you need a cap.
    ``deadline`` (a ``time.monotonic()`` timestamp) bounds the *search
    itself*: the backtracking that proves emptiness or bridges sparse
    solution regions is worst-case exponential in ``d``, so without a
    deadline the generator can stall for hours between yields at d ≳ 20.
    When the deadline passes, remaining subtrees are pruned and the
    generator ends early (callers detect this via their own clock).
    """
    d = W.shape[0]
    # available[k] = list of source-rows i such that |W[i, k]| > threshold_w.
    # Reading column k tells us which sources are "compatible" with variable k.
    available = [
        [i for i in range(d) if abs(W[i, k]) > threshold_w] for k in range(d)
    ]
    used = [False] * d
    pi = [0] * d  # pi[k] = source-row to place at position k

    def search(col: int) -> Iterator[np.ndarray]:
        if deadline is not None and time.monotonic() > deadline:
            return
        if col == d:
            yield W[pi, :].copy()
            return
        for i in available[col]:
            if used[i]:
                continue
            used[i] = True
            pi[col] = i
            yield from search(col + 1)
            used[i] = False

    yield from search(0)


def _b_from_W(Wpi: np.ndarray) -> np.ndarray:
    """Convert a diagonal-zeroless ``Wπ`` into ``B = I − diag(Wπ)⁻¹ Wπ``."""
    diag = np.diag(Wpi).copy()
    # Avoid divide-by-zero (won't happen if threshold_w > 0 was respected,
    # but be defensive).
    diag[np.abs(diag) < 1e-12] = 1e-12
    Wn = Wpi / diag[:, None]
    B = np.eye(Wpi.shape[0]) - Wn
    np.fill_diagonal(B, 0.0)
    return B


def _matching_permutation(
    W: np.ndarray,
    cost: np.ndarray,
) -> np.ndarray | None:
    """Return ``W[π, :]`` for the admissible row permutation π that minimises
    ``cost`` via linear assignment; ``None`` if no admissible matching exists
    (forbidden cells are ``+∞`` in ``cost``)."""
    from scipy.optimize import linear_sum_assignment

    if not np.isfinite(cost).any(axis=0).all():
        # Some column has no admissible source — no perfect matching exists.
        return None
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ValueError:
        return None
    if not np.isfinite(cost[row_ind, col_ind]).all():
        # Solver returned an assignment that uses a forbidden cell.
        return None
    # row_ind is 0..d-1 sorted; col_ind[m] is the column assigned to row
    # row_ind[m]. We want pi such that W[pi[k], k] is the matched entry,
    # i.e. pi[col_ind[m]] = row_ind[m].
    d = W.shape[0]
    pi = np.empty(d, dtype=int)
    pi[col_ind] = row_ind
    return W[pi, :].copy()


def _hungarian_permutation(
    W: np.ndarray,
    threshold_w: float,
) -> np.ndarray | None:
    """Return ``W[π, :]`` for one admissible row permutation π via Hungarian.

    Solves a linear-assignment problem on ``C[i, k] = -log(|W[i, k]|)`` with
    ``C[i, k] = +∞`` whenever ``|W[i, k]| <= threshold_w``. The optimal
    assignment is a permutation π such that ``|W[π(k), k]| > threshold_w``
    for all k whenever any such permutation exists; returns ``None`` if no
    admissible matching exists.

    Worst-case ``O(d^3)``. By Theorem 1 of the paper the condensation does
    not depend on which admissible representative is chosen, so this is
    sufficient when only ``G'`` is needed.
    """
    abs_W = np.abs(W)
    cost = np.where(abs_W > threshold_w, -np.log(abs_W + 1e-300), np.inf)
    return _matching_permutation(W, cost)


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: ICA
# ─────────────────────────────────────────────────────────────────────────
def fit_ica_unmixing(
    obs: np.ndarray,
    *,
    ica_max_iter: int = 5_000,
    ica_tolerance: float = 1e-6,
    random_state: int = 0,
    fastica_retries: int = 4,
) -> tuple[np.ndarray | None, str | None]:
    """FastICA stage — estimate the unmixing matrix ``W``.

    sklearn FastICA can occasionally fail; the seed is bumped and the fit
    retried up to ``fastica_retries`` times. Returns ``(W, None)`` on
    success and ``(None, last_error)`` if every attempt failed.
    """
    from sklearn.decomposition import FastICA

    obs_arr = np.asarray(obs, dtype=np.float64)
    n_features = obs_arr.shape[1]

    last_err: str | None = None
    for k in range(fastica_retries + 1):
        try:
            ica = FastICA(
                n_components=n_features,
                whiten="unit-variance",
                max_iter=ica_max_iter,
                tol=ica_tolerance,
                random_state=random_state + k,
            )
            ica.fit(obs_arr)
            # sklearn convention: ``S_ = (X - mean) @ ica.components_.T``
            # so ``W = ica.components_``.
            return np.asarray(ica.components_, dtype=np.float64), None
        except Exception as exc:
            last_err = repr(exc)
            continue
    return None, last_err


# ─────────────────────────────────────────────────────────────────────────
# Stage 2: candidate enumeration (N-rooks)
# ─────────────────────────────────────────────────────────────────────────
def enumerate_admissible_candidates(
    W: np.ndarray,
    *,
    threshold_b: float = 0.1,
    threshold_w: float = 0.1,
    max_perms: int = 10_000,
    time_budget_sec: float | None = None,
) -> dict:
    """Enumerate the admissible candidate set for ``W`` via N-rooks.

    Two safeguards bound the worst-case-exponential search:

    * A **Hungarian existence pre-check**: if no admissible permutation
      exists at ``threshold_w``, the DFS would have to exhaust the entire
      search tree just to prove emptiness. A single ``O(d^3)`` matching
      detects this case up front and skips the enumeration entirely.
    * An optional **wall-clock budget** (``time_budget_sec``): the DFS is
      pruned once the budget is spent, because even when candidates exist
      the backtracking between sparse solutions can blow up at d ≳ 20.

    Returns a dict with keys

      ``stable``   : list of thresholded B̂ candidates with ``ρ(B̂) < 1``.
      ``unstable`` : list of thresholded B̂ candidates with ``ρ(B̂) ≥ 1``.
      ``n_candidates_enumerated`` : number of admissible permutations
          actually enumerated (≤ ``max_perms``).
      ``enumeration_cap_hit`` : True iff at least one further admissible
          permutation exists beyond ``max_perms`` — determined by probing
          the generator once past the cap, not by comparing counts. When
          True, the candidate lists are a *truncated* subset of the
          equivalence class, not the full class.
      ``enumeration_timed_out`` : True iff the time budget pruned the
          search — the candidate lists are then a truncated subset and
          ``enumeration_cap_hit`` may be under-reported.
    """
    # Hungarian pre-check: polynomial certificate of (non-)existence.
    if _hungarian_permutation(W, threshold_w) is None:
        return {
            "stable": [],
            "unstable": [],
            "n_candidates_enumerated": 0,
            "enumeration_cap_hit": False,
            "enumeration_timed_out": False,
        }

    deadline = (
        time.monotonic() + time_budget_sec
        if time_budget_sec is not None else None
    )
    gen = _nrooks_permutations(W, threshold_w, deadline=deadline)
    stable: list[np.ndarray] = []
    unstable: list[np.ndarray] = []
    n_enum = 0
    for Wpi in islice(gen, max_perms):
        n_enum += 1
        B = _b_from_W(Wpi)
        # Threshold for edge-set extraction.
        B_thresh = np.where(np.abs(B) > threshold_b, B, 0.0)
        rho = float(np.max(np.abs(np.linalg.eigvals(B_thresh))))
        if rho < 1.0:
            stable.append(B_thresh)
        else:
            unstable.append(B_thresh)
    timed_out = deadline is not None and time.monotonic() > deadline
    cap_hit = (
        n_enum == max_perms
        and not timed_out
        and next(gen, None) is not None
    )
    return {
        "stable": stable,
        "unstable": unstable,
        "n_candidates_enumerated": n_enum,
        "enumeration_cap_hit": cap_hit,
        "enumeration_timed_out": timed_out,
    }


# ─────────────────────────────────────────────────────────────────────────
# Stage 3a: selection from an enumerated candidate set
# ─────────────────────────────────────────────────────────────────────────
def choose_enumerated_candidate(
    candidates: dict,
    *,
    pick_strategy: str = "first_stable",
    random_state: int = 0,
) -> tuple[np.ndarray | None, bool | None]:
    """Select ``B_chosen`` from ``enumerate_admissible_candidates`` output.

    ``"random_admissible"`` draws uniformly from stable + unstable (in that
    concatenation order, seeded by ``random_state``); any other strategy
    takes the first stable candidate, falling back to the first unstable
    one. Returns ``(None, None)`` when the candidate set is empty.
    """
    stable = candidates["stable"]
    unstable = candidates["unstable"]
    all_candidates = stable + unstable
    if pick_strategy == "random_admissible" and all_candidates:
        rng_pick = np.random.default_rng(random_state)
        idx = int(rng_pick.integers(len(all_candidates)))
        return all_candidates[idx], idx < len(stable)
    if stable:
        return stable[0], True
    if unstable:
        return unstable[0], False
    return None, None


# ─────────────────────────────────────────────────────────────────────────
# Stage 3b: single-representative selection without enumeration
# ─────────────────────────────────────────────────────────────────────────
def hungarian_candidate(
    W: np.ndarray,
    *,
    threshold_b: float = 0.1,
    threshold_w: float = 0.1,
) -> np.ndarray | None:
    """One admissible thresholded B̂ via linear assignment, or ``None``.

    Enumerates **zero** equivalence-class candidates: the representative
    comes from a single ``O(d^3)`` matching on ``W``, so its stability is
    not assessed against alternatives (callers get ``is_stable=None``).
    """
    Wpi = _hungarian_permutation(W, threshold_w)
    if Wpi is None:
        return None
    B = _b_from_W(Wpi)
    return np.where(np.abs(B) > threshold_b, B, 0.0)


def random_matching_candidate(
    W: np.ndarray,
    *,
    threshold_b: float = 0.1,
    threshold_w: float = 0.1,
    random_state: int = 0,
) -> np.ndarray | None:
    """One *randomised* admissible thresholded B̂ WITHOUT enumeration.

    Assigns i.i.d. Uniform(0, 1) costs to the allowed cells
    (``|W[i, k]| > threshold_w``) and solves a single perfect-matching
    problem — ``O(d^3)``, zero equivalence-class enumeration. The result is
    a *randomised admissible solution*: which admissible permutation is
    returned varies with ``random_state``, but the induced distribution is
    NOT uniform over the admissible set (matching with i.i.d. costs biases
    toward permutations that are optimal for some cost draw). Use
    ``choose_enumerated_candidate(..., "random_admissible")`` when a
    uniform sample over the (enumerated) class is required.
    """
    rng = np.random.default_rng(random_state)
    cost = np.where(np.abs(W) > threshold_w, rng.random(W.shape), np.inf)
    Wpi = _matching_permutation(W, cost)
    if Wpi is None:
        return None
    B = _b_from_W(Wpi)
    return np.where(np.abs(B) > threshold_b, B, 0.0)


def arbitrary_permutation_candidate(
    W: np.ndarray,
    *,
    threshold_b: float = 0.1,
    threshold_w: float = 0.1,
    random_state: int = 0,
) -> tuple[np.ndarray, bool, float]:
    """Thresholded B̂ from a uniformly random row permutation of ``W``.

    This deliberately does *not* enforce the LiNG-D admissibility constraint.
    The candidate is normalised and returned even when one or more selected
    diagonal entries fall below ``threshold_w``.  It is therefore a negative
    control, not an equivalence-class representative.

    Returns ``(B, admissible, min_abs_diagonal)``.  ``admissible`` records
    whether the sampled permutation happened to satisfy the same diagonal
    threshold used by the admissible-selection procedures.
    """
    rng = np.random.default_rng(random_state)
    Wpi = W[rng.permutation(W.shape[0]), :].copy()
    min_abs_diagonal = float(np.min(np.abs(np.diag(Wpi))))
    admissible = bool(min_abs_diagonal > threshold_w)
    B = _b_from_W(Wpi)
    return (
        np.where(np.abs(B) > threshold_b, B, 0.0),
        admissible,
        min_abs_diagonal,
    )


def _failure_dict(n_features: int, last_err: str | None) -> dict:
    return {
        "B_chosen": np.zeros((n_features, n_features)),
        "stable": [],
        "unstable": [],
        "is_stable": False,
        "failed": True,
        "n_perms": 0,
        "n_candidates_enumerated": 0,
        "n_candidates_returned": 0,
        "enumeration_cap_hit": False,
        "enumeration_timed_out": False,
        "last_error": last_err,
    }


def run_lingd(
    obs: np.ndarray,
    *,
    threshold_b: float = 0.1,
    threshold_w: float = 0.1,
    ica_max_iter: int = 5_000,
    ica_tolerance: float = 1e-6,
    random_state: int = 0,
    max_perms: int = 10_000,
    fastica_retries: int = 4,
    pick_strategy: str = "first_stable",
    **_: object,  # accept and ignore legacy kwargs (max_retries, ica_a, …)
) -> dict:
    """LiNG-D on ``obs`` — returns the equivalence class.

    Composes ``fit_ica_unmixing`` + (``enumerate_admissible_candidates`` →
    ``choose_enumerated_candidate`` | ``hungarian_candidate``). When no
    admissible permutation exists at ``threshold_w``, the threshold is
    halved (down to 0.01) and selection re-runs on the *same* ``W``.

    Parameters
    ----------
    obs : ``(n_samples, n_features)`` array.
    threshold_b : threshold on ``|B̂|`` for declaring an edge.
    threshold_w : threshold on ``|W|`` during N-rooks permutation search.
    ica_max_iter, ica_tolerance, random_state : forwarded to
        ``sklearn.decomposition.FastICA``.
    max_perms : safety bound on the N-rooks search (default 10 000); typical
        cyclic-SEM cases find 1–10 admissible permutations long before that.
    fastica_retries : sklearn FastICA can occasionally fail to converge.
        We bump the seed and retry up to this many times. Default 4.
    pick_strategy : how to select ``B_chosen`` from the equivalence class.

        ``"first_stable"`` (default)
            Take the first admissible permutation with ``ρ(B̂) < 1``.
            For sparse cyclic SEMs this uniquely recovers the true B.

        ``"random_admissible"``
            Draw uniformly at random from **all** admissible permutations
            (stable AND unstable).  Used to demonstrate the identifiability
            gap: all members share the same condensation, but variable-level
            edges differ across members.

        ``"hungarian_any"``
            Skip N-rooks enumeration and return a single admissible
            representative found via the Hungarian algorithm in ``O(d^3)``.
            Sufficient when only the condensation is needed (Theorem 1 of
            the paper guarantees the choice does not affect ``G'``). The
            return dict has ``stable=unstable=[]``, ``is_stable=None``,
            ``n_candidates_enumerated=0`` and ``n_candidates_returned=1``.

    Returns
    -------
    dict with keys
      ``B_chosen``  : ``(n, n)`` preferred B̂ in column convention
                      (``B[i,j] != 0  ⇔  j → i``).
      ``stable``    : list of B̂ candidates with ``ρ(B̂) < 1``.
      ``unstable``  : list of B̂ candidates with ``ρ(B̂) ≥ 1``.
      ``is_stable`` : whether ``B_chosen`` came from ``stable``.
      ``failed``    : True if FastICA itself failed every retry.
      ``n_perms``   : legacy alias — admissible permutations enumerated by
                      N-rooks (``1`` for the Hungarian path, which
                      historically reported its single representative here;
                      prefer the two explicit fields below).
      ``n_candidates_enumerated`` : equivalence-class members enumerated
                      (``0`` for the Hungarian path — it never enumerates).
      ``n_candidates_returned``   : candidates materialised in
                      ``stable`` + ``unstable`` (``1`` for Hungarian).
      ``enumeration_cap_hit``     : True iff the ``max_perms`` cap truncated
                      the enumeration (see ``enumerate_admissible_candidates``).
      ``ica_runtime_sec``, ``selection_runtime_sec`` : per-stage wall time.
    """
    obs_arr = np.asarray(obs, dtype=np.float64)
    n_features = obs_arr.shape[1]

    t0 = time.perf_counter()
    W, last_err = fit_ica_unmixing(
        obs_arr,
        ica_max_iter=ica_max_iter,
        ica_tolerance=ica_tolerance,
        random_state=random_state,
        fastica_retries=fastica_retries,
    )
    ica_runtime_sec = time.perf_counter() - t0

    if W is None:
        out = _failure_dict(n_features, last_err)
        out["ica_runtime_sec"] = ica_runtime_sec
        out["selection_runtime_sec"] = 0.0
        return out

    t1 = time.perf_counter()

    if pick_strategy == "hungarian_any":
        thr_w = threshold_w
        B_hun: np.ndarray | None = None
        while True:
            B_hun = hungarian_candidate(
                W, threshold_b=threshold_b, threshold_w=thr_w
            )
            if B_hun is not None or thr_w <= 0.01:
                break
            thr_w /= 2  # no admissible matching — lower the bar and retry
        found = B_hun is not None
        return {
            "B_chosen": B_hun if found else np.zeros((n_features, n_features)),
            "stable": [],
            "unstable": [],
            "is_stable": None,
            "failed": False,
            "n_perms": 1 if found else 0,
            "n_candidates_enumerated": 0,
            "n_candidates_returned": 1 if found else 0,
            "enumeration_cap_hit": False,
            "enumeration_timed_out": False,
            "last_error": None,
            "ica_runtime_sec": ica_runtime_sec,
            "selection_runtime_sec": time.perf_counter() - t1,
        }

    thr_w = threshold_w
    while True:
        candidates = enumerate_admissible_candidates(
            W, threshold_b=threshold_b, threshold_w=thr_w, max_perms=max_perms
        )
        B_chosen, is_stable = choose_enumerated_candidate(
            candidates, pick_strategy=pick_strategy, random_state=random_state
        )
        if B_chosen is not None or thr_w <= 0.01:
            break
        thr_w /= 2  # no admissible permutation — lower the bar and retry

    if B_chosen is None:
        B_chosen = np.zeros((n_features, n_features))
        is_stable = False

    return {
        "B_chosen": B_chosen,
        "stable": candidates["stable"],
        "unstable": candidates["unstable"],
        "is_stable": is_stable,
        "failed": False,
        "n_perms": candidates["n_candidates_enumerated"],
        "n_candidates_enumerated": candidates["n_candidates_enumerated"],
        "n_candidates_returned": (
            len(candidates["stable"]) + len(candidates["unstable"])
        ),
        "enumeration_cap_hit": candidates["enumeration_cap_hit"],
        "enumeration_timed_out": candidates["enumeration_timed_out"],
        "last_error": None,
        "ica_runtime_sec": ica_runtime_sec,
        "selection_runtime_sec": time.perf_counter() - t1,
    }
