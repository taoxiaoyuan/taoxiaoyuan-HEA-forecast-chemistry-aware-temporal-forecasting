import numpy as np

from hea_forecast.modeling import _cluster_bootstrap_intervals, _metrics


def test_metrics_reward_perfect_ranking() -> None:
    y_true = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1])
    metrics = _metrics(y_true, scores)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["brier_score"] < 0.05
    assert metrics["expected_calibration_error"] < 0.2


def test_cluster_bootstrap_is_deterministic() -> None:
    y_true = np.array([1, 0, 0, 1, 0, 0])
    scores = np.array([0.9, 0.2, 0.1, 0.8, 0.3, 0.4])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    first = _cluster_bootstrap_intervals(y_true, scores, groups, seed=7, repetitions=100)
    second = _cluster_bootstrap_intervals(y_true, scores, groups, seed=7, repetitions=100)
    assert first == second
    assert first["pr_auc_ci_low"] is not None
