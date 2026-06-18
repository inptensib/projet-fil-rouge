"""Tests for rakuten/common/metrics.py."""

from __future__ import annotations

import pytest
from sklearn.metrics import f1_score

from rakuten.common.metrics import (
    compute_all_metrics,
    compute_f1_by_tranche,
    compute_f1_metrics,
    constrained_f1,
    hierarchical_consistency_rate,
)


class TestComputeF1Metrics:
    def test_perfect_predictions(self):
        labels = [0, 1, 2, 0, 1, 2]
        preds = [0, 1, 2, 0, 1, 2]
        metrics = compute_f1_metrics(labels, preds, labels, preds)
        assert metrics["f1_l1_macro"] == 1.0
        assert metrics["f1_l2_macro"] == 1.0

    def test_wrong_predictions(self):
        labels = [0, 1, 2]
        preds = [1, 2, 0]
        metrics = compute_f1_metrics(labels, preds, labels, preds)
        assert metrics["f1_l1_macro"] == 0.0
        assert metrics["f1_l2_macro"] == 0.0

    def test_partial_predictions(self):
        labels_l1 = [0, 0, 1, 1]
        preds_l1 = [0, 0, 1, 0]  # 3/4 correct
        labels_l2 = [0, 1, 2, 3]
        preds_l2 = [0, 1, 2, 3]  # all correct
        metrics = compute_f1_metrics(labels_l1, preds_l1, labels_l2, preds_l2)
        assert 0.0 < metrics["f1_l1_macro"] < 1.0
        assert metrics["f1_l2_macro"] == 1.0


class TestConstrainedF1:
    def test_all_l1_correct(self):
        """When all L1 predictions are correct, constrained F1 == regular F1."""
        labels_l1 = [0, 0, 1, 1]
        preds_l1 = [0, 0, 1, 1]
        labels_l2 = [0, 1, 2, 3]
        preds_l2 = [0, 1, 2, 3]
        mapping = {0: 0, 1: 0, 2: 1, 3: 1}

        cf1 = constrained_f1(labels_l1, preds_l1, labels_l2, preds_l2, mapping)
        assert cf1 == 1.0

    def test_l1_wrong_penalizes_l2(self):
        """When L1 is wrong, L2 prediction is treated as invalid."""
        labels_l1 = [0, 1]
        preds_l1 = [1, 0]  # both wrong
        labels_l2 = [0, 1]
        preds_l2 = [0, 1]  # both "correct" but L1 wrong
        mapping = {0: 0, 1: 1}

        cf1 = constrained_f1(labels_l1, preds_l1, labels_l2, preds_l2, mapping)
        assert cf1 == 0.0


class TestHierarchicalConsistencyRate:
    def test_perfect_consistency(self):
        preds_l1 = [0, 0, 1, 1]
        preds_l2 = [0, 1, 2, 3]
        mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        hcr = hierarchical_consistency_rate(preds_l1, preds_l2, mapping)
        assert hcr == 1.0

    def test_no_consistency(self):
        preds_l1 = [1, 1, 0, 0]  # swapped
        preds_l2 = [0, 1, 2, 3]
        mapping = {0: 0, 1: 0, 2: 1, 3: 1}
        hcr = hierarchical_consistency_rate(preds_l1, preds_l2, mapping)
        assert hcr == 0.0

    def test_partial_consistency(self):
        preds_l1 = [0, 1, 1, 0]
        preds_l2 = [0, 2, 3, 3]
        mapping = {0: 0, 2: 1, 3: 1}
        hcr = hierarchical_consistency_rate(preds_l1, preds_l2, mapping)
        assert 0.0 < hcr < 1.0

    def test_empty_preds(self):
        assert hierarchical_consistency_rate([], [], {}) == 0.0


class TestComputeAllMetrics:
    def test_without_mapping(self):
        labels = [0, 1, 2]
        preds = [0, 1, 2]
        metrics = compute_all_metrics(labels, preds, labels, preds)
        assert "f1_l1_macro" in metrics
        assert "constrained_f1_macro" not in metrics
        assert "hcr" not in metrics

    def test_with_mapping(self):
        labels_l1 = [0, 0, 1]
        preds_l1 = [0, 0, 1]
        labels_l2 = [0, 1, 2]
        preds_l2 = [0, 1, 2]
        mapping = {0: 0, 1: 0, 2: 1}

        metrics = compute_all_metrics(
            labels_l1, preds_l1, labels_l2, preds_l2, mapping
        )
        assert "constrained_f1_macro" in metrics
        assert "hcr" in metrics
        assert metrics["hcr"] == 1.0


class TestComputeF1ByTranche:
    def test_three_tranches(self):
        class_counts = {0: 500, 1: 50, 2: 5}
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 0, 2, 1]
        result = compute_f1_by_tranche(y_true, y_pred, class_counts)

        assert set(result) == {"many", "medium", "few"}
        assert result["many"]["n_classes"] == 1
        assert result["many"]["support"] == 2
        assert result["many"]["f1_macro"] == pytest.approx(0.8)
        assert result["medium"]["n_classes"] == 1
        assert result["medium"]["support"] == 2
        assert result["medium"]["f1_macro"] == pytest.approx(0.5)
        assert result["few"]["n_classes"] == 1
        assert result["few"]["support"] == 2
        assert result["few"]["f1_macro"] == pytest.approx(2 / 3)

    def test_threshold_boundaries(self):
        class_counts = {0: 100, 1: 99, 2: 20, 3: 19}
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        result = compute_f1_by_tranche(y_true, y_pred, class_counts)

        assert result["many"]["n_classes"] == 1
        assert result["medium"]["n_classes"] == 2
        assert result["few"]["n_classes"] == 1

    def test_empty_tranche(self):
        class_counts = {0: 200, 1: 150}
        y_true = [0, 1, 0]
        y_pred = [0, 1, 1]
        result = compute_f1_by_tranche(y_true, y_pred, class_counts)

        for tranche in ("medium", "few"):
            assert result[tranche]["f1_macro"] is None
            assert result[tranche]["n_classes"] == 0
            assert result[tranche]["support"] == 0
        assert result["many"]["n_classes"] == 2
        assert result["many"]["support"] == 3

    def test_unknown_label_counts_as_few(self):
        class_counts = {0: 200}
        y_true = [0, 9]
        y_pred = [0, 9]
        result = compute_f1_by_tranche(y_true, y_pred, class_counts)

        assert result["few"]["n_classes"] == 1
        assert result["few"]["support"] == 1
        assert result["few"]["f1_macro"] == 1.0

    def test_single_tranche_matches_global_f1(self):
        class_counts = {0: 300, 1: 250, 2: 400}
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]
        result = compute_f1_by_tranche(y_true, y_pred, class_counts)

        expected = f1_score(y_true, y_pred, average="macro", zero_division=0)
        assert result["many"]["f1_macro"] == pytest.approx(expected)
        assert result["many"]["support"] == len(y_true)
