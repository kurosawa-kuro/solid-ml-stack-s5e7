"""
Comprehensive test coverage for src/validation.py
Targeting high-impact functions to improve coverage from 48% to 80%+
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.validation import (
    CVStrategy,
    CVLogger,
    aggregate_cv_scores,
    calculate_accuracy,
    calculate_auc,
    calculate_prediction_distribution,
    check_data_integrity,
    default_cv_strategy,
    save_cv_log,
    validate_target_distribution,
    CV_CONFIG,
)


class TestCVStrategy:
    """Test CVStrategy class functionality"""

    def test_init_default(self):
        """Test CVStrategy initialization with defaults"""
        strategy = CVStrategy()
        
        assert strategy.n_splits == 5
        assert strategy.shuffle is True
        assert strategy.random_state == 42
        assert isinstance(strategy.cv, StratifiedKFold)

    def test_init_custom(self):
        """Test CVStrategy initialization with custom parameters"""
        strategy = CVStrategy(n_splits=3, shuffle=False, random_state=123)
        
        assert strategy.n_splits == 3
        assert strategy.shuffle is False
        assert strategy.random_state == 123

    def test_split(self):
        """Test CV split generation"""
        strategy = CVStrategy(n_splits=3, random_state=42)
        
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        
        splits = strategy.split(X, y)
        
        assert len(splits) == 3
        
        # Check that splits are proper train/val indices
        for train_idx, val_idx in splits:
            assert len(train_idx) + len(val_idx) == 100
            assert len(set(train_idx).intersection(set(val_idx))) == 0  # No overlap

    def test_get_config(self):
        """Test get_config method"""
        strategy = CVStrategy(n_splits=7, shuffle=True, random_state=999)
        config = strategy.get_config()
        
        expected_config = {
            "n_splits": 7,
            "shuffle": True,
            "random_state": 999,
            "stratify": True
        }
        
        assert config == expected_config


class TestMetricFunctions:
    """Test metric calculation functions"""

    def test_calculate_accuracy_perfect(self):
        """Test accuracy calculation with perfect predictions"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        accuracy = calculate_accuracy(y_true, y_pred)
        assert accuracy == 1.0

    def test_calculate_accuracy_partial(self):
        """Test accuracy calculation with partial correct predictions"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])  # 4/5 correct
        
        accuracy = calculate_accuracy(y_true, y_pred)
        assert accuracy == 0.8

    def test_calculate_auc_perfect(self):
        """Test AUC calculation with perfect predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])
        
        auc = calculate_auc(y_true, y_pred_proba)
        assert auc == 1.0

    def test_calculate_auc_random(self):
        """Test AUC calculation with random-like predictions"""
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.5, 0.5, 0.5, 0.5])
        
        auc = calculate_auc(y_true, y_pred_proba)
        assert auc == 0.5

    def test_calculate_prediction_distribution_balanced(self):
        """Test prediction distribution with balanced predictions"""
        predictions = np.array([0, 1, 0, 1, 0, 1])
        
        distribution = calculate_prediction_distribution(predictions)
        
        assert distribution["total_predictions"] == 6
        assert distribution["extrovert_count"] == 3
        assert distribution["introvert_count"] == 3
        assert distribution["extrovert_ratio"] == 0.5
        assert distribution["introvert_ratio"] == 0.5

    def test_calculate_prediction_distribution_skewed(self):
        """Test prediction distribution with skewed predictions"""
        predictions = np.array([1, 1, 1, 1, 0])  # 80% extrovert
        
        distribution = calculate_prediction_distribution(predictions)
        
        assert distribution["total_predictions"] == 5
        assert distribution["extrovert_count"] == 4
        assert distribution["introvert_count"] == 1
        assert distribution["extrovert_ratio"] == 0.8
        assert distribution["introvert_ratio"] == 0.2


class TestScoreAggregation:
    """Test score aggregation functions"""

    def test_aggregate_cv_scores_consistent(self):
        """Test score aggregation with consistent scores"""
        fold_scores = [0.95, 0.95, 0.95, 0.95, 0.95]
        
        aggregated = aggregate_cv_scores(fold_scores)
        
        assert aggregated["mean_score"] == 0.95
        assert aggregated["std_score"] == 0.0
        assert aggregated["min_score"] == 0.95
        assert aggregated["max_score"] == 0.95
        assert aggregated["median_score"] == 0.95

    def test_aggregate_cv_scores_varied(self):
        """Test score aggregation with varied scores"""
        fold_scores = [0.90, 0.95, 1.00, 0.85, 0.90]
        
        aggregated = aggregate_cv_scores(fold_scores)
        
        assert aggregated["mean_score"] == 0.92
        assert aggregated["std_score"] > 0
        assert aggregated["min_score"] == 0.85
        assert aggregated["max_score"] == 1.00
        assert aggregated["median_score"] == 0.90

    def test_aggregate_cv_scores_single(self):
        """Test score aggregation with single score"""
        fold_scores = [0.95]
        
        aggregated = aggregate_cv_scores(fold_scores)
        
        assert aggregated["mean_score"] == 0.95
        assert aggregated["std_score"] == 0.0
        assert aggregated["min_score"] == 0.95
        assert aggregated["max_score"] == 0.95
        assert aggregated["median_score"] == 0.95


class TestDataIntegrity:
    """Test data integrity checking functions"""

    def test_check_data_integrity_valid(self):
        """Test data integrity with valid data"""
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        
        checks = check_data_integrity(X, y)
        
        assert checks["shape_consistent"] is True
        assert checks["no_missing_features"] is True
        assert checks["no_infinite_features"] is True
        assert checks["no_missing_targets"] is True
        assert checks["binary_targets"] is True
        assert checks["sufficient_samples"] is True
        assert checks["balanced_classes"] is True

    def test_check_data_integrity_shape_mismatch(self):
        """Test data integrity with shape mismatch"""
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 50)  # Different size
        
        checks = check_data_integrity(X, y)
        
        assert checks["shape_consistent"] is False

    def test_check_data_integrity_missing_features(self):
        """Test data integrity with missing features"""
        X = np.random.random((100, 5))
        X[10, 2] = np.nan  # Add NaN
        y = np.random.randint(0, 2, 100)
        
        checks = check_data_integrity(X, y)
        
        assert checks["no_missing_features"] is False

    def test_check_data_integrity_infinite_features(self):
        """Test data integrity with infinite features"""
        X = np.random.random((100, 5))
        X[10, 2] = np.inf  # Add infinity
        y = np.random.randint(0, 2, 100)
        
        checks = check_data_integrity(X, y)
        
        assert checks["no_infinite_features"] is False

    def test_check_data_integrity_missing_targets(self):
        """Test data integrity with missing targets"""
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100).astype(float)
        y[10] = np.nan  # Add NaN to targets
        
        checks = check_data_integrity(X, y)
        
        assert checks["no_missing_targets"] is False

    def test_check_data_integrity_non_binary_targets(self):
        """Test data integrity with non-binary targets"""
        X = np.random.random((100, 5))
        y = np.random.randint(0, 3, 100)  # 3 classes instead of 2
        
        checks = check_data_integrity(X, y)
        
        assert checks["binary_targets"] is False

    def test_check_data_integrity_insufficient_samples(self):
        """Test data integrity with insufficient samples"""
        X = np.random.random((5, 5))  # Only 5 samples
        y = np.random.randint(0, 2, 5)
        
        checks = check_data_integrity(X, y)
        
        assert checks["sufficient_samples"] is False

    def test_check_data_integrity_unbalanced_classes(self):
        """Test data integrity with unbalanced classes (missing class)"""
        X = np.random.random((100, 5))
        y = np.ones(100)  # All class 1, no class 0
        
        checks = check_data_integrity(X, y)
        
        assert checks["balanced_classes"] is False


class TestTargetDistributionValidation:
    """Test target distribution validation"""

    def test_validate_target_distribution_binary(self):
        """Test target distribution validation with binary targets"""
        y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        
        distribution = validate_target_distribution(y)
        
        assert distribution["unique_values"] == [0, 1]
        assert distribution["class_counts"] == [4, 4]
        assert distribution["class_ratios"] == [0.5, 0.5]
        assert distribution["is_binary"] is True
        assert distribution["has_both_classes"] is True

    def test_validate_target_distribution_imbalanced(self):
        """Test target distribution validation with imbalanced targets"""
        y = np.array([0, 1, 1, 1, 1, 1, 1, 1])  # 1 class 0, 7 class 1
        
        distribution = validate_target_distribution(y)
        
        assert distribution["unique_values"] == [0, 1]
        assert distribution["class_counts"] == [1, 7]
        assert distribution["class_ratios"] == [0.125, 0.875]
        assert distribution["is_binary"] is True
        assert distribution["has_both_classes"] is True

    def test_validate_target_distribution_multiclass(self):
        """Test target distribution validation with multiclass targets"""
        y = np.array([0, 1, 2, 0, 1, 2])
        
        distribution = validate_target_distribution(y)
        
        assert distribution["unique_values"] == [0, 1, 2]
        assert distribution["class_counts"] == [2, 2, 2]
        assert distribution["is_binary"] is False
        assert distribution["has_both_classes"] is True

    def test_validate_target_distribution_single_class(self):
        """Test target distribution validation with single class"""
        y = np.array([1, 1, 1, 1, 1])
        
        distribution = validate_target_distribution(y)
        
        assert distribution["unique_values"] == [1]
        assert distribution["class_counts"] == [0, 5]  # bincount includes 0 count for class 0
        assert distribution["is_binary"] is False
        assert distribution["has_both_classes"] is False


class TestCVLogger:
    """Test CVLogger functionality"""

    def test_init_default_log_dir(self):
        """Test CVLogger initialization with default log directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "outputs/logs")
            logger = CVLogger(log_dir=log_dir)
            
            assert logger.log_dir == Path(log_dir)
            assert logger.log_dir.exists()

    def test_create_log_entry(self):
        """Test log entry creation"""
        logger = CVLogger()
        
        fold_scores = [0.95, 0.96, 0.94]
        training_time = 120.5
        
        log_entry = logger.create_log_entry(
            model_type="LightGBM",
            cv_config={"n_splits": 3, "random_state": 42},
            fold_scores=fold_scores,
            training_time=training_time,
            extra_param="test_value"
        )
        
        assert log_entry["model_type"] == "LightGBM"
        assert log_entry["fold_scores"] == fold_scores
        assert log_entry["training_time"] == training_time
        assert log_entry["extra_param"] == "test_value"
        assert "timestamp" in log_entry
        assert "mean_score" in log_entry
        assert "std_score" in log_entry

    def test_save_json_log_auto_filename(self):
        """Test JSON log saving with auto-generated filename"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = CVLogger(log_dir=temp_dir)
            
            log_entry = {
                "model_type": "LightGBM",
                "fold_scores": [0.95, 0.96, 0.94],
                "timestamp": "2024-01-01T12:00:00"
            }
            
            log_path = logger.save_json_log(log_entry)
            
            assert os.path.exists(log_path)
            assert log_path.endswith(".json")
            
            # Verify content
            with open(log_path, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == log_entry

    def test_save_json_log_custom_filename(self):
        """Test JSON log saving with custom filename"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = CVLogger(log_dir=temp_dir)
            
            log_entry = {"test": "data"}
            filename = "custom_log.json"
            
            log_path = logger.save_json_log(log_entry, filename=filename)
            
            assert log_path.endswith(filename)
            assert os.path.exists(log_path)

    def test_save_csv_log_auto_filename(self):
        """Test CSV log saving with auto-generated filename"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = CVLogger(log_dir=temp_dir)
            
            cv_results = pd.DataFrame({
                "fold": [1, 2, 3],
                "accuracy": [0.95, 0.96, 0.94],
                "auc": [0.98, 0.97, 0.96]
            })
            
            csv_path = logger.save_csv_log(cv_results)
            
            assert os.path.exists(csv_path)
            assert csv_path.endswith(".csv")
            
            # Verify content
            loaded_df = pd.read_csv(csv_path)
            pd.testing.assert_frame_equal(loaded_df, cv_results)

    def test_save_csv_log_custom_filename(self):
        """Test CSV log saving with custom filename"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = CVLogger(log_dir=temp_dir)
            
            cv_results = pd.DataFrame({"test": [1, 2, 3]})
            filename = "custom_results.csv"
            
            csv_path = logger.save_csv_log(cv_results, filename=filename)
            
            assert csv_path.endswith(filename)
            assert os.path.exists(csv_path)


class TestUtilityFunctions:
    """Test utility functions"""

    def test_save_cv_log(self):
        """Test save_cv_log convenience function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_data = {
                "model_type": "LightGBM",
                "mean_score": 0.95,
                "fold_scores": [0.94, 0.95, 0.96]
            }
            
            filepath = os.path.join(temp_dir, "test_log.json")
            
            save_cv_log(log_data, filepath)
            
            assert os.path.exists(filepath)
            
            # Verify content
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == log_data

    def test_save_cv_log_creates_directory(self):
        """Test save_cv_log creates necessary directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_data = {"test": "data"}
            nested_path = os.path.join(temp_dir, "nested", "dir", "log.json")
            
            save_cv_log(log_data, nested_path)
            
            assert os.path.exists(nested_path)


class TestConstants:
    """Test module constants"""

    def test_cv_config_structure(self):
        """Test CV_CONFIG constant structure"""
        assert "n_splits" in CV_CONFIG
        assert "shuffle" in CV_CONFIG
        assert "random_state" in CV_CONFIG
        assert CV_CONFIG["n_splits"] == 5
        assert CV_CONFIG["shuffle"] is True
        assert CV_CONFIG["random_state"] == 42

    def test_default_cv_strategy(self):
        """Test default_cv_strategy instance"""
        assert isinstance(default_cv_strategy, CVStrategy)
        assert default_cv_strategy.n_splits == CV_CONFIG["n_splits"]
        assert default_cv_strategy.shuffle == CV_CONFIG["shuffle"]
        assert default_cv_strategy.random_state == CV_CONFIG["random_state"]


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_fold_scores(self):
        """Test aggregation with empty fold scores"""
        fold_scores = []
        
        # This should handle empty input gracefully
        with pytest.raises((ValueError, ZeroDivisionError)):
            aggregate_cv_scores(fold_scores)

    def test_single_sample_data_integrity(self):
        """Test data integrity with minimal data"""
        X = np.array([[1, 2, 3]])  # Single sample
        y = np.array([1])
        
        checks = check_data_integrity(X, y)
        
        assert checks["shape_consistent"] is True
        assert checks["sufficient_samples"] is False  # Less than 10 samples
        assert checks["balanced_classes"] is False  # Only one class

    def test_logger_with_empty_log_entry(self):
        """Test logger with minimal log entry"""
        logger = CVLogger()
        
        log_entry = logger.create_log_entry(
            model_type="Test",
            cv_config={},
            fold_scores=[0.5],
            training_time=1.0
        )
        
        assert log_entry["model_type"] == "Test"
        assert "timestamp" in log_entry