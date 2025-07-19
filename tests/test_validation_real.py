"""
Test cases for actual validation.py functions to improve coverage
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    CVLogger,
    CVStrategy,
    aggregate_cv_scores,
    calculate_accuracy,
    calculate_auc,
    calculate_prediction_distribution,
    check_data_integrity,
    validate_target_distribution,
)


class TestCVStrategyComplete:
    """Complete tests for CVStrategy class"""

    def test_cv_strategy_init_default(self):
        """Test default initialization"""
        strategy = CVStrategy()
        assert strategy.n_splits == 5
        assert strategy.shuffle is True
        assert strategy.random_state == 42
        assert strategy._cv is not None

    def test_cv_strategy_init_custom(self):
        """Test custom initialization"""
        strategy = CVStrategy(n_splits=3, shuffle=False, random_state=123)
        assert strategy.n_splits == 3
        assert strategy.shuffle is False
        assert strategy.random_state == 123

    def test_get_splits(self):
        """Test getting CV splits"""
        strategy = CVStrategy(n_splits=3)
        X = np.random.random((30, 5))
        y = np.array([0, 1] * 15)

        splits = list(strategy.get_splits(X, y))
        assert len(splits) == 3

        # Check that all indices are used
        all_train_idx = []
        all_test_idx = []
        for train_idx, test_idx in splits:
            all_train_idx.extend(train_idx)
            all_test_idx.extend(test_idx)
            # Each fold should have train and test data
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(train_idx) + len(test_idx) == 30

    def test_get_config(self):
        """Test configuration export"""
        strategy = CVStrategy(n_splits=4, shuffle=False, random_state=99)
        config = strategy.get_config()

        assert config["n_splits"] == 4
        assert config["shuffle"] is False
        assert config["random_state"] == 99
        assert config["strategy_type"] == "StratifiedKFold"


class TestCVLoggerComplete:
    """Complete tests for CVLogger class"""

    def test_cv_logger_init_default(self):
        """Test default initialization"""
        logger = CVLogger()
        assert logger.log_dir == Path("outputs/cv_logs")

    def test_cv_logger_init_custom(self):
        """Test custom initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CVLogger(log_dir=tmpdir)
            assert logger.log_dir == Path(tmpdir)
            assert logger.log_dir.exists()

    def test_create_log_entry(self):
        """Test creating log entry"""
        logger = CVLogger()

        cv_results = {
            "mean_score": 0.95,
            "std_score": 0.02,
            "fold_scores": [0.94, 0.95, 0.96],
        }

        model_params = {"learning_rate": 0.1, "num_leaves": 31}
        feature_names = ["feature1", "feature2", "feature3"]

        entry = logger.create_log_entry(
            experiment_name="test_experiment",
            model_type="LightGBM",
            cv_results=cv_results,
            model_params=model_params,
            feature_names=feature_names,
            notes="Test run with default params",
        )

        # Check required fields
        assert entry["experiment_name"] == "test_experiment"
        assert entry["model_type"] == "LightGBM"
        assert entry["cv_score"] == 0.95
        assert entry["cv_std"] == 0.02
        assert entry["fold_scores"] == [0.94, 0.95, 0.96]
        assert entry["model_params"] == model_params
        assert entry["feature_names"] == feature_names
        assert entry["feature_count"] == 3
        assert entry["notes"] == "Test run with default params"
        assert "timestamp" in entry

    def test_save_json_log(self):
        """Test saving JSON log"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CVLogger(log_dir=tmpdir)

            log_entry = {
                "experiment_name": "test",
                "cv_score": 0.95,
                "timestamp": "2024-01-01T12: 00: 00",
            }

            filepath = logger.save_json_log(log_entry, "test_log.json")

            assert filepath.exists()
            assert filepath.parent == Path(tmpdir)

            # Load and verify
            with open(filepath, "r") as f:
                loaded = json.load(f)
            assert loaded == log_entry

    def test_save_csv_log(self):
        """Test saving CSV log"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CVLogger(log_dir=tmpdir)

            df = pd.DataFrame(
                {
                    "fold": [0, 1, 2],
                    "accuracy": [0.94, 0.95, 0.96],
                    "auc": [0.97, 0.98, 0.99],
                }
            )

            filepath = logger.save_csv_log(df, "test_results.csv")

            assert filepath.exists()

            # Load and verify
            loaded_df = pd.read_csv(filepath)
            assert len(loaded_df) == 3
            assert list(loaded_df.columns) == ["fold", "accuracy", "auc"]
            pd.testing.assert_frame_equal(loaded_df, df)


class TestDataIntegrityComplete:
    """Complete tests for data integrity functions"""

    def test_check_data_integrity_all_valid(self):
        """Test data integrity check with valid data"""
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)

        checks = check_data_integrity(X, y)

        assert checks["has_nan"] is False
        assert checks["has_inf"] is False
        assert checks["shape_match"] is True
        assert checks["min_samples_ok"] is True
        assert checks["all_checks_passed"] is True

    def test_check_data_integrity_with_nan(self):
        """Test data integrity with NaN values"""
        X = np.random.random((100, 5))
        X[10, 2] = np.nan
        X[20, 4] = np.nan
        y = np.random.randint(0, 2, 100)

        checks = check_data_integrity(X, y)

        assert checks["has_nan"] is True
        assert checks["has_inf"] is False
        assert checks["shape_match"] is True
        assert checks["all_checks_passed"] is False

    def test_check_data_integrity_with_inf(self):
        """Test data integrity with infinite values"""
        X = np.random.random((100, 5))
        X[5, 1] = np.inf
        X[15, 3] = -np.inf
        y = np.random.randint(0, 2, 100)

        checks = check_data_integrity(X, y)

        assert checks["has_nan"] is False
        assert checks["has_inf"] is True
        assert checks["shape_match"] is True
        assert checks["all_checks_passed"] is False

    def test_check_data_integrity_shape_mismatch(self):
        """Test data integrity with shape mismatch"""
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 80)  # Wrong size

        checks = check_data_integrity(X, y)

        assert checks["shape_match"] is False
        assert checks["all_checks_passed"] is False

    def test_check_data_integrity_too_few_samples(self):
        """Test data integrity with insufficient samples"""
        X = np.random.random((8, 5))
        y = np.random.randint(0, 2, 8)

        checks = check_data_integrity(X, y, min_samples=10)

        assert checks["min_samples_ok"] is False
        assert checks["all_checks_passed"] is False

    def test_check_data_integrity_custom_min_samples(self):
        """Test data integrity with custom minimum samples"""
        X = np.random.random((50, 5))
        y = np.random.randint(0, 2, 50)

        # Default should pass
        checks1 = check_data_integrity(X, y)
        assert checks1["min_samples_ok"] is True

        # Custom high threshold should fail
        checks2 = check_data_integrity(X, y, min_samples=100)
        assert checks2["min_samples_ok"] is False


class TestTargetDistributionComplete:
    """Complete tests for target distribution validation"""

    def test_validate_target_distribution_balanced(self):
        """Test balanced target distribution"""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # Perfectly balanced

        dist = validate_target_distribution(y)

        assert dist["n_classes"] == 2
        assert dist["class_counts"][0] == 4
        assert dist["class_counts"][1] == 4
        assert dist["class_ratios"][0] == 0.5
        assert dist["class_ratios"][1] == 0.5
        assert dist["is_binary"] is True
        assert dist["is_balanced"] is True

    def test_validate_target_distribution_imbalanced(self):
        """Test imbalanced target distribution"""
        y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1])  # 7: 2 ratio

        dist = validate_target_distribution(y)

        assert dist["n_classes"] == 2
        assert dist["class_counts"][0] == 7
        assert dist["class_counts"][1] == 2
        assert dist["is_balanced"] is False
        assert dist["imbalance_ratio"] == pytest.approx(7 / 9, rel=1e-6)

    def test_validate_target_distribution_multiclass(self):
        """Test multiclass target distribution"""
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        dist = validate_target_distribution(y)

        assert dist["n_classes"] == 3
        assert dist["is_binary"] is False
        assert all(count == 3 for count in dist["class_counts"].values())

    def test_validate_target_distribution_threshold(self):
        """Test balanced threshold"""
        # Just within threshold (0.35 ratio)
        y1 = np.array([0] * 35 + [1] * 65)
        dist1 = validate_target_distribution(y1, balance_threshold=0.35)
        assert dist1["is_balanced"] is True

        # Just outside threshold
        y2 = np.array([0] * 30 + [1] * 70)
        dist2 = validate_target_distribution(y2, balance_threshold=0.35)
        assert dist2["is_balanced"] is False


class TestMetricsComplete:
    """Complete tests for metric calculation functions"""

    def test_calculate_accuracy(self):
        """Test accuracy calculation"""
        # Perfect accuracy
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert calculate_accuracy(y_true, y_pred) == 1.0

        # 75% accuracy
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        assert calculate_accuracy(y_true, y_pred) == 0.75

        # 0% accuracy
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        assert calculate_accuracy(y_true, y_pred) == 0.0

    def test_calculate_auc(self):
        """Test AUC calculation"""
        # Perfect separation
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert calculate_auc(y_true, y_score) == 1.0

        # Random scores
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        assert calculate_auc(y_true, y_score) == 0.5

        # Inverted scores (worst case)
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        assert calculate_auc(y_true, y_score) == 0.0

    def test_calculate_prediction_distribution(self):
        """Test prediction distribution calculation"""
        # Balanced predictions
        predictions = np.array([0, 1, 0, 1, 0, 1])
        dist = calculate_prediction_distribution(predictions)

        assert dist["class_0_count"] == 3
        assert dist["class_1_count"] == 3
        assert dist["class_0_ratio"] == 0.5
        assert dist["class_1_ratio"] == 0.5
        assert dist["extrovert_ratio"] == 0.5  # class 1

        # Imbalanced predictions
        predictions = np.array([1, 1, 1, 1, 0])
        dist = calculate_prediction_distribution(predictions)

        assert dist["class_0_count"] == 1
        assert dist["class_1_count"] == 4
        assert dist["extrovert_ratio"] == 0.8


class TestCVAggregationComplete:
    """Complete tests for CV aggregation functions"""

    def test_aggregate_cv_scores_basic(self):
        """Test basic CV score aggregation"""
        fold_scores = [0.90, 0.92, 0.88, 0.91, 0.89]

        aggregated = aggregate_cv_scores(fold_scores)

        assert aggregated["mean_score"] == 0.90
        assert aggregated["std_score"] == pytest.approx(0.0158, rel=1e-2)
        assert aggregated["min_score"] == 0.88
        assert aggregated["max_score"] == 0.92
        assert aggregated["fold_scores"] == fold_scores

    def test_aggregate_cv_scores_perfect(self):
        """Test aggregation with perfect scores"""
        fold_scores = [1.0, 1.0, 1.0]

        aggregated = aggregate_cv_scores(fold_scores)

        assert aggregated["mean_score"] == 1.0
        assert aggregated["std_score"] == 0.0
        assert aggregated["min_score"] == 1.0
        assert aggregated["max_score"] == 1.0

    def test_aggregate_cv_scores_single(self):
        """Test aggregation with single fold"""
        fold_scores = [0.95]

        aggregated = aggregate_cv_scores(fold_scores)

        assert aggregated["mean_score"] == 0.95
        assert aggregated["std_score"] == 0.0
        assert aggregated["min_score"] == 0.95
        assert aggregated["max_score"] == 0.95
