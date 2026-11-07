"""Tests for binary evaluation utilities."""

import numpy as np

from machinepulse.models.evaluation import binary_metrics, select_f1_threshold


def test_threshold_selection_and_metrics() -> None:
    """A separable validation set produces perfect thresholded metrics."""
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    scores = np.array([0.05, 0.20, 0.80, 0.95], dtype=np.float64)

    threshold = select_f1_threshold(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)

    assert 0.2 < threshold <= 0.8
    assert metrics["f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["true_negatives"] == 2
