"""OICA/ICA-based identification of the finest DAG-coarsening of a cyclic DG.

Background
----------
For a linear non-Gaussian cyclic SEM (Lacerda et al., UAI 2008; Dai et al., ICLR 2026):

    V = B V + E   (column convention, B[i,j] = weight of edge i→j)
    V = (I - B)^{-1} E =: A E

where A is the mixing matrix and E has mutually independent, non-Gaussian components.

OICA identifiability (Eriksson & Koivunen 2004; Salehkaleybar et al. 2020):
When A has no proportional columns, it is identifiable up to column permutation
and diagonal scaling from the distribution of V.

Key structural fact (path rank, Lemma 2 in Dai et al.):
    rank(A_{Z,Y}) = ρ_G(Z, Y)  (generically)

In particular, A[i,j] ≠ 0 iff there is a directed path from j to i in G.
Therefore support(A) = adjacency matrix of the transitive closure of G.

SCCs of the transitive closure = SCCs of G (Tarjan's theorem).

So the algorithm for finding the finest DAG-coarsening from observational data is:
  1. Estimate A = (I-B)^{-1} via FastICA.
  2. Threshold |A| to get binary support.
  3. Build directed graph from support: edge j→i iff support[i,j] = 1.
  4. Run Tarjan's (via nx.strongly_connected_components) on this graph.

Alternatively, for a more direct estimate of the graph itself:
  B_est = I - inv(A),  then threshold |B_est|.
Using support(A) for SCCs avoids matrix inversion noise and is therefore
preferred for the partitioning step.

Conventions match the rest of repare_cycle:
  - weights[i,j] = weight of edge i→j
  - data = noise @ _solution_matrix  where _solution_matrix = inv(I - B)
  - sklearn FastICA: X = S @ mixing_.T  ⇒  mixing_ ≈ A P D (A in col. convention)
  - A[i,j] = (I-B)^{-1}[i,j] = total causal effect from source j on variable i
             = 0 iff no directed path from j to i in G

References
----------
Lacerda et al. (2008). Discovering cyclic causal models by independent
    components analysis. UAI.
Salehkaleybar et al. (2020). Learning linear Non-Gaussian causal models
    in the presence of latent variables. JMLR.
Dai, Albrecht, Spirtes, Zhang (2026). Distributional Equivalence in Linear
    Non-Gaussian Latent-Variable Cyclic Causal Models. ICLR.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np


def estimate_mixing_matrix(
    X: np.ndarray,
    max_iter: int = 10_000,
    random_state: int = 0,
    tol: float = 1e-6,
) -> np.ndarray:
    """Estimate the mixing matrix A = (I-B)^{-1} from data via FastICA.

    Returns an (n, n) matrix approximating A up to the identifiability
    ambiguity (column permutation and scaling) resolved by:
      1. Hungarian algorithm (maximize absolute diagonal entries).
      2. Diagonal normalization (so A[i,i] = 1 for all i, matching the
         true A = I + B + B^2 + ... whose diagonal is dominated by I).

    Parameters
    ----------
    X : (n_samples, n_vars) data matrix.
    max_iter : maximum FastICA iterations.
    random_state : seed for FastICA reproducibility.
    tol : convergence tolerance for FastICA.

    Returns
    -------
    mixing : (n_vars, n_vars) estimated A, column j = influence of source j.
        mixing[i,j] ≈ (I-B)^{-1}[i,j] = total causal effect from j to i.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.decomposition import FastICA

    n_vars = X.shape[1]

    ica = FastICA(
        n_components=n_vars,
        max_iter=max_iter,
        random_state=random_state,
        tol=tol,
        whiten="unit-variance",
    )
    try:
        ica.fit(X)
    except Exception as exc:
        raise RuntimeError(f"FastICA failed to converge: {exc}") from exc

    mixing = ica.mixing_.copy()  # (n_vars, n_vars), ≈ A P D

    # --- Resolve permutation ambiguity ---
    # True A has A[i,i] = 1 (dominates due to I in the Neumann series).
    # Find the column permutation that maximises the sum of |diagonal| entries.
    _, col_ind = linear_sum_assignment(-np.abs(mixing))
    mixing = mixing[:, col_ind]

    # --- Resolve sign ambiguity ---
    # Flip each column so its diagonal entry is positive.
    for j in range(n_vars):
        if mixing[j, j] < 0:
            mixing[:, j] *= -1

    # --- Diagonal normalization (so A[i,i] ≈ 1) ---
    diag = np.abs(np.diag(mixing))
    for j in range(n_vars):
        if diag[j] < 0.1 * np.linalg.norm(mixing[:, j]):
            warnings.warn(
                f"Column {j} of estimated mixing matrix has a small diagonal "
                f"entry ({diag[j]:.4f}) relative to its norm "
                f"({np.linalg.norm(mixing[:, j]):.4f}). "
                "The permutation step may have failed. "
                "ICA results for this run may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            mixing[:, j] /= diag[j]

    # --- Condition number guard ---
    cond = np.linalg.cond(mixing)
    if cond > 1e6:
        warnings.warn(
            f"Estimated mixing matrix is ill-conditioned (κ = {cond:.2e}). "
            "Downstream estimates of B = I - inv(A) may be unreliable. "
            "Consider using the support of A directly for SCC identification.",
            RuntimeWarning,
            stacklevel=2,
        )

    return mixing


def mixing_to_adjacency(mixing: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Convert the estimated mixing matrix to a binary adjacency matrix of B.

    Computes B_est = I - inv(mixing) and thresholds to get binary adjacency.
    Under the column convention used by estimate_mixing_matrix
    (mixing[i,j] = effect from j to i), the recovered B_est satisfies
    B_est[i,j] ≠ 0 iff there is a direct edge j → i in the estimated graph.

    Parameters
    ----------
    mixing : (n, n) estimated mixing matrix from estimate_mixing_matrix.
    threshold : entries of B_est with |value| < threshold are treated as zero.

    Returns
    -------
    adj : (n, n) binary adjacency matrix, adj[i,j] = 1 iff estimated edge j → i.

    Notes
    -----
    This inverts the mixing matrix, which amplifies estimation errors by a
    factor proportional to the condition number. For SCC identification alone,
    using oica_scc_partition (which works directly on the support of mixing)
    is more robust. Use mixing_to_adjacency only when the full B estimate is needed.
    """
    n = mixing.shape[0]
    try:
        I_minus_B = np.linalg.inv(mixing)  # ≈ (I - B)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Mixing matrix is singular; cannot recover B. "
            "Use oica_scc_partition with the support of A directly."
        ) from exc

    B_est = np.eye(n) - I_minus_B
    np.fill_diagonal(B_est, 0.0)  # no self-loops
    return (np.abs(B_est) > threshold).astype(int)


def oica_scc_partition(
    X: np.ndarray,
    threshold: float = 0.1,
    max_iter: int = 10_000,
    random_state: int = 0,
) -> list[frozenset]:
    """Identify the SCC partition (finest valid DAG-coarsening) from observational data.

    Uses FastICA to estimate the mixing matrix A = (I-B)^{-1}, then derives SCCs
    directly from the support of A (= adjacency of the transitive closure of G).
    This avoids the noise-amplification of matrix inversion.

    Key: SCCs(transitive closure of G) = SCCs(G).  So thresholding |A| and
    running Tarjan's gives the same partition as running Tarjan's on G itself.

    Parameters
    ----------
    X : (n_samples, n_vars) observational data.
    threshold : entries of |A| below this are treated as zero.
    max_iter : max FastICA iterations.
    random_state : FastICA seed.

    Returns
    -------
    sccs : list of frozensets, one per SCC. These form the finest valid
        DAG-coarsening identifiable from observational data alone.
    """
    mixing = estimate_mixing_matrix(X, max_iter=max_iter, random_state=random_state)

    n = mixing.shape[0]

    # Support of A: entry [i,j] ≠ 0 means j can reach i (directed path j→...→i).
    # So edge j→i exists in the transitive closure.
    support = np.abs(mixing) > threshold
    np.fill_diagonal(support, 0)

    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    rows, cols = np.where(support)
    for i, j in zip(rows.tolist(), cols.tolist()):
        # support[i,j] = 1 means j → i (j causes i, path j→...→i)
        g.add_edge(j, i)

    return [frozenset(scc) for scc in nx.strongly_connected_components(g)]
