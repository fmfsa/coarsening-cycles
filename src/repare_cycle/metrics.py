"""Partition-error diagnostics for estimated SCC partitions.

Aggregate scores (ARI, cluster-DAG F1) show *that* recovery improves with
sample size but not *how* it fails. This module reports the failure mode
directly at the level of unordered node pairs:

  split error — a pair of nodes in the same true SCC assigned to
      different estimated clusters (a true SCC was broken apart);
  merge error — a pair of nodes in different true SCCs assigned to
      the same estimated cluster (distinct SCCs were fused).

Rates are normalised by the number of pairs eligible for each error type,
so a run can be classified as exact / split-only / merge-only / mixed even
when both error types occur simultaneously or the estimated number of
clusters happens to match the truth.
"""

from __future__ import annotations

import numpy as np


def partition_pair_errors(
    true_labels: np.ndarray,
    est_labels: np.ndarray,
) -> dict:
    """Pairwise split/merge diagnostics between two partitions.

    Parameters
    ----------
    true_labels, est_labels : ``(d,)`` integer label vectors. Nodes sharing
        a label belong to the same cluster. Label values are arbitrary;
        only the induced partitions matter.

    Returns
    -------
    dict with keys
      ``n_same_true_pairs`` : pairs in the same true cluster (split-eligible).
      ``n_diff_true_pairs`` : pairs in different true clusters (merge-eligible).
      ``n_split_pairs``     : split-eligible pairs estimated apart.
      ``n_merge_pairs``     : merge-eligible pairs estimated together.
      ``split_rate``        : ``n_split_pairs / n_same_true_pairs``;
                              NaN when the denominator is zero (all true
                              clusters are singletons — no pair can split).
      ``merge_rate``        : ``n_merge_pairs / n_diff_true_pairs``;
                              NaN when the denominator is zero (the truth
                              is a single cluster — no pair can merge).
      ``error_type``        : ``"exact"`` | ``"split_only"`` |
                              ``"merge_only"`` | ``"mixed"``.
    """
    true_arr = np.asarray(true_labels)
    est_arr = np.asarray(est_labels)
    if true_arr.shape != est_arr.shape or true_arr.ndim != 1:
        raise ValueError(
            f"Label vectors must be 1-D and equal length; got "
            f"{true_arr.shape} vs {est_arr.shape}."
        )

    # Upper-triangle boolean co-membership over all C(d, 2) pairs.
    iu, ju = np.triu_indices(true_arr.shape[0], k=1)
    same_true = true_arr[iu] == true_arr[ju]
    same_est = est_arr[iu] == est_arr[ju]

    n_same_true = int(same_true.sum())
    n_diff_true = int((~same_true).sum())
    n_split = int((same_true & ~same_est).sum())
    n_merge = int((~same_true & same_est).sum())

    split_rate = n_split / n_same_true if n_same_true > 0 else float("nan")
    merge_rate = n_merge / n_diff_true if n_diff_true > 0 else float("nan")

    if n_split == 0 and n_merge == 0:
        error_type = "exact"
    elif n_merge == 0:
        error_type = "split_only"
    elif n_split == 0:
        error_type = "merge_only"
    else:
        error_type = "mixed"

    return {
        "n_same_true_pairs": n_same_true,
        "n_diff_true_pairs": n_diff_true,
        "n_split_pairs": n_split,
        "n_merge_pairs": n_merge,
        "split_rate": split_rate,
        "merge_rate": merge_rate,
        "error_type": error_type,
    }
