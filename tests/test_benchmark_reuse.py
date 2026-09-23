"""A cached timing is valid only for the exact same scientific computation."""
import importlib.util
from pathlib import Path
import platform

import pytest

spec = importlib.util.spec_from_file_location(
    "reuse_group_lingam", Path(__file__).parents[1]
    / "src/expt/workflow/scripts/reuse_group_lingam.py")
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)


@pytest.fixture
def completed():
    return dict(status="ok", method="group_lingam", seed=0, samp_size=1000,
                source_sha256={"estimator": "abc"}, versions={"lingam": "1.13.0"},
                dataset_sha256="data123", numerical_threads=1,
                python=platform.python_version(), platform=platform.platform(),
                fit_runtime_sec=30, timeout_sec=120)


def check(result, **kwargs):
    return reuse.compatible(result, {"method": "group_lingam", "seed": 0, "samp_size": 1000},
                            {"estimator": "abc"}, {"lingam": "1.13.0"},
                            kwargs.get("dataset_hash", "data123"), kwargs.get("timeout", 3600))


def test_completed_result_reusable_with_larger_budget_without_mutating(completed):
    before = completed.copy()
    assert check(completed)
    assert completed == before


@pytest.mark.parametrize("key,value", [
    ("status", "timeout"), ("status", "error"),
    ("method", "hungarian"), ("seed", 1), ("samp_size", 2000),
    ("source_sha256", {"estimator": "changed"}),
    ("versions", {"lingam": "different"}), ("numerical_threads", 2),
    ("python", "different"), ("platform", "different"),
])
def test_reject_incompatible_run(completed, key, value):
    completed[key] = value
    assert not check(completed)


def test_reject_changed_observations_and_insufficient_budget(completed):
    assert not check(completed, dataset_hash="different")
    assert not check(completed, timeout=20)
