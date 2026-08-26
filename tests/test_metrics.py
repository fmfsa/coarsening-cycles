"""Tests for split/merge partition diagnostics."""

import numpy as np
import pytest

from repare_cycle.metrics import partition_pair_errors


def test_exact_partition():
    true = np.array([0, 0, 1, 1, 2])
    est = np.array([5, 5, 3, 3, 9])  # same partition, different label values
    r = partition_pair_errors(true, est)
    assert r["n_split_pairs"] == 0 and r["n_merge_pairs"] == 0
    assert r["split_rate"] == 0.0 and r["merge_rate"] == 0.0
    assert r["error_type"] == "exact"


def test_split_only_hand_counted():
    # True SCC {0,1,2} split into {0,1} | {2}; singleton {3} untouched.
    true = np.array([0, 0, 0, 1])
    est = np.array([0, 0, 2, 1])
    r = partition_pair_errors(true, est)
    # Same-true pairs: (0,1), (0,2), (1,2) -> 3; split pairs: (0,2), (1,2).
    assert r["n_same_true_pairs"] == 3
    assert r["n_split_pairs"] == 2
    assert r["split_rate"] == pytest.approx(2 / 3)
    assert r["n_merge_pairs"] == 0
    assert r["error_type"] == "split_only"


def test_merge_only_hand_counted():
    # True SCCs {0,1} and {2,3} merged into one estimated cluster.
    true = np.array([0, 0, 1, 1])
    est = np.array([0, 0, 0, 0])
    r = partition_pair_errors(true, est)
    # Diff-true pairs: (0,2), (0,3), (1,2), (1,3) -> 4; all merged.
    assert r["n_diff_true_pairs"] == 4
    assert r["n_merge_pairs"] == 4
    assert r["merge_rate"] == 1.0
    assert r["n_split_pairs"] == 0
    assert r["error_type"] == "merge_only"


def test_mixed_split_and_merge_same_cluster_count():
    # Truth {0,1}{2,3}; estimate {0,2}{1,3}: both split AND merged while
    # the estimated number of clusters equals the true number — invisible
    # to a cluster-count diagnostic, caught by pair errors.
    true = np.array([0, 0, 1, 1])
    est = np.array([0, 1, 0, 1])
    r = partition_pair_errors(true, est)
    assert r["n_split_pairs"] == 2  # (0,1) and (2,3)
    assert r["n_merge_pairs"] == 2  # (0,2) and (1,3)
    assert r["error_type"] == "mixed"


def test_zero_split_denominator_all_singletons():
    # All-singleton truth: no pair can split; merge still measurable.
    true = np.array([0, 1, 2])
    est = np.array([0, 0, 1])
    r = partition_pair_errors(true, est)
    assert r["n_same_true_pairs"] == 0
    assert np.isnan(r["split_rate"])
    assert r["n_merge_pairs"] == 1
    assert r["error_type"] == "merge_only"


def test_zero_merge_denominator_single_true_cluster():
    # Single-cluster truth: no pair can merge; split still measurable.
    true = np.array([0, 0, 0])
    est = np.array([0, 1, 1])
    r = partition_pair_errors(true, est)
    assert r["n_diff_true_pairs"] == 0
    assert np.isnan(r["merge_rate"])
    assert r["n_split_pairs"] == 2
    assert r["error_type"] == "split_only"


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        partition_pair_errors(np.array([0, 1]), np.array([0, 1, 2]))
