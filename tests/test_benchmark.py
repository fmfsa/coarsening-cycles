import numpy as np
import pytest

from repare_cycle.benchmark import group_lingam_adapter, condensation_metrics


def test_native_groups_unknown_edges_and_orientation():
    # 0 <-> 1, then 1 -> 2. The baseline does not identify internal edges.
    weights = np.array([[0, .5, 0], [.5, 0, .8], [0, 0, 0]])
    b = np.array([[0, np.nan, 0], [np.nan, 0, 0], [0, .8, 0]])
    groups, adj = group_lingam_adapter([[0, 1], [2]], b)
    assert adj[1, 2] and not adj[2, 1]
    assert not adj[0, 1] and not adj[1, 0]
    assert np.isnan(b[0, 1])  # adapter does not mutate the original
    scores = condensation_metrics(weights, groups, adj)
    assert scores["exact_condensation"] == 1
    assert scores["ari_scc"] == scores["oracle_cluster_f1"] == 1


def test_wrong_partition_cannot_pass_on_oracle_edges():
    weights = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])
    scores = condensation_metrics(weights, [[0], [1], [2]], weights != 0)
    assert scores["oracle_cluster_f1"] == 1
    assert scores["exact_condensation"] == 0


def test_exact_checks_node_membership_not_isomorphism():
    weights = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    predicted = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    assert condensation_metrics(weights, [[0], [1], [2]], predicted)["exact_condensation"] == 0


@pytest.mark.parametrize("groups", [[[0], [0, 1]], [[0]], [[0], [2]], [[], [0, 1]]])
def test_reject_invalid_partition(groups):
    with pytest.raises(ValueError, match="partition"):
        group_lingam_adapter(groups, np.zeros((2, 2)))


def test_unknown_between_groups_rejected():
    with pytest.raises(ValueError, match="across"):
        group_lingam_adapter([[0], [1]], [[0, np.nan], [0, 0]])
