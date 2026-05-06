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

from itertools import islice
from typing import Iterator

import numpy as np


def _nrooks_permutations(
    W: np.ndarray,
    threshold_w: float,
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
    from scipy.optimize import linear_sum_assignment

    abs_W = np.abs(W)
    cost = np.where(abs_W > threshold_w, -np.log(abs_W + 1e-300), np.inf)
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
            and ``n_perms=1``.

    Returns
    -------
    dict with keys
      ``B_chosen``  : ``(n, n)`` preferred B̂ in column convention
                      (``B[i,j] != 0  ⇔  j → i``).
      ``stable``    : list of B̂ candidates with ``ρ(B̂) < 1``.
      ``unstable``  : list of B̂ candidates with ``ρ(B̂) ≥ 1``.
      ``is_stable`` : whether ``B_chosen`` came from ``stable``.
      ``failed``    : True if FastICA itself failed every retry.
      ``n_perms``   : how many admissible permutations were enumerated.
    """
    from sklearn.decomposition import FastICA

    obs_arr = np.asarray(obs, dtype=np.float64)
    n_features = obs_arr.shape[1]

    # FastICA — sklearn returns the unmixing matrix as ``W = whitening⁻¹ ·
    # rotation⁻¹``. The transformation ``S = (X - mean) @ W.T`` produces
    # mutually independent components.
    W: np.ndarray | None = None
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
            W = np.asarray(ica.components_, dtype=np.float64)
            break
        except Exception as exc:
            last_err = repr(exc)
            continue

    if W is None:
        return {
            "B_chosen": np.zeros((n_features, n_features)),
            "stable": [],
            "unstable": [],
            "is_stable": False,
            "failed": True,
            "n_perms": 0,
            "last_error": last_err,
        }

    # Hungarian fast path: skip N-rooks enumeration entirely. By Theorem 1
    # of the paper, the recovered condensation is invariant across the
    # equivalence class, so a single admissible representative suffices.
    if pick_strategy == "hungarian_any":
        Wpi = _hungarian_permutation(W, threshold_w)
        if Wpi is not None:
            B = _b_from_W(Wpi)
            B_thresh = np.where(np.abs(B) > threshold_b, B, 0.0)
            return {
                "B_chosen": B_thresh,
                "stable": [],
                "unstable": [],
                "is_stable": None,
                "failed": False,
                "n_perms": 1,
                "last_error": None,
            }
        # Fall through to threshold halving below.
        if threshold_w > 0.01:
            return run_lingd(
                obs,
                threshold_b=threshold_b,
                threshold_w=threshold_w / 2,
                ica_max_iter=ica_max_iter,
                ica_tolerance=ica_tolerance,
                random_state=random_state,
                max_perms=max_perms,
                fastica_retries=0,
                pick_strategy=pick_strategy,
            )
        return {
            "B_chosen": np.zeros((n_features, n_features)),
            "stable": [],
            "unstable": [],
            "is_stable": None,
            "failed": False,
            "n_perms": 0,
            "last_error": None,
        }

    # N-rooks: enumerate diagonal-zeroless column permutations of W.
    stable: list[np.ndarray] = []
    unstable: list[np.ndarray] = []
    n_perms = 0
    for Wpi in islice(_nrooks_permutations(W, threshold_w), max_perms):
        n_perms += 1
        B = _b_from_W(Wpi)
        # Threshold for edge-set extraction.
        B_thresh = np.where(np.abs(B) > threshold_b, B, 0.0)
        rho = float(np.max(np.abs(np.linalg.eigvals(B_thresh))))
        if rho < 1.0:
            stable.append(B_thresh)
        else:
            unstable.append(B_thresh)

    all_candidates = stable + unstable
    if pick_strategy == "random_admissible" and all_candidates:
        rng_pick = np.random.default_rng(random_state)
        idx = int(rng_pick.integers(len(all_candidates)))
        B_chosen = all_candidates[idx]
        is_stable = idx < len(stable)
    elif stable:
        B_chosen = stable[0]
        is_stable = True
    elif unstable:
        B_chosen = unstable[0]
        is_stable = False
    else:
        # No admissible permutation found at this threshold_w. Lower the bar
        # to threshold_w/2 once before giving up.
        if threshold_w > 0.01:
            return run_lingd(
                obs,
                threshold_b=threshold_b,
                threshold_w=threshold_w / 2,
                ica_max_iter=ica_max_iter,
                ica_tolerance=ica_tolerance,
                random_state=random_state,
                max_perms=max_perms,
                fastica_retries=0,  # already have W; just rerun N-rooks
                pick_strategy=pick_strategy,
            )
        B_chosen = np.zeros((n_features, n_features))
        is_stable = False

    return {
        "B_chosen": B_chosen,
        "stable": stable,
        "unstable": unstable,
        "is_stable": is_stable,
        "failed": False,
        "n_perms": n_perms,
        "last_error": None,
    }
