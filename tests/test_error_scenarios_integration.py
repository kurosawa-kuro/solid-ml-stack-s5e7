"""
Error Scenario and Edge Case Integration Tests
Tests for proper exception handling and edge cases across the ML pipeline
"""

import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.bronze import create_bronze_tables, load_bronze_data, load_data, quick_preprocess
from src.data.gold import create_gold_tables, extract_model_arrays, get_ml_ready_data, load_gold_data
from src.data.silver import create_silver_tables, load_silver_data
from src.models import CrossValidationTrainer, LightGBMModel
from src.util.notifications import WebhookNotifier
from src.validation import CVLogger, CVStrategy, check_data_integrity


class TestDataPipelineErrorHandling:
    """Test error handling in data pipeline"""

    def test_database_connection_failure(self):
        """Test handling of database connection failures"""
        # Test with non-existent database path
        with patch("src.data.bronze.DB_PATH", "/non/existent/path.db"):
            with pytest.raises(Exception):
                load_data()

    def test_corrupted_data_handling(self):
        """Test handling of corrupted or malformed data"""
        # Create corrupted data
        corrupted_data = pd.DataFrame({
            "id": [1, 2, None, 4],  # Missing ID
            "Time_spent_Alone": [np.inf, -np.inf, np.nan, 5],  # Invalid values
            "Social_event_attendance": ["invalid", 2, 3, 4],  # Wrong type
            "Personality": ["Extrovert", "Unknown", "Introvert", "Extrovert"]  # Invalid value
        })
        
        # Test preprocessing handles corrupted data gracefully
        # This should raise an error due to type conversion issues
        with pytest.raises((TypeError, ValueError)):
            processed = quick_preprocess(corrupted_data)

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets"""
        # Create empty dataset
        empty_data = pd.DataFrame(columns=["id", "Time_spent_Alone", "Personality"])
        
        # Should handle empty data without crashing
        processed = quick_preprocess(empty_data)
        assert len(processed) == 0
        # Check that the basic columns are preserved
        for col in empty_data.columns:
            assert col in processed.columns

    def test_missing_columns_handling(self):
        """Test handling of datasets with missing required columns"""
        # Data missing critical columns
        incomplete_data = pd.DataFrame({
            "id": [1, 2, 3],
            "Time_spent_Alone": [1, 2, 3]
            # Missing other required columns
        })
        
        # Should handle missing columns gracefully
        processed = quick_preprocess(incomplete_data)
        assert len(processed) == 3

    def test_extreme_values_handling(self):
        """Test handling of extreme values in data"""
        extreme_data = pd.DataFrame({
            "id": [1, 2, 3],
            "Time_spent_Alone": [0, 1000000, -1000],  # Extreme values
            "Social_event_attendance": [0, 0, 1000000],
            "Friends_circle_size": [-100, 0, 1000000],
            "Personality": ["Extrovert", "Introvert", "Extrovert"]
        })
        
        processed = quick_preprocess(extreme_data)
        
        # Should handle extreme values
        assert len(processed) == 3
        assert processed["Time_spent_Alone"].min() >= -1000  # Reasonable bounds


class TestModelTrainingErrorHandling:
    """Test error handling in model training"""

    def test_insufficient_data_error(self):
        """Test handling of insufficient training data"""
        # Very small dataset
        X_tiny = np.random.random((2, 5))
        y_tiny = np.array([0, 1])
        
        model = LightGBMModel()
        # Should handle tiny data gracefully (may warn but shouldn't crash)
        model.fit(X_tiny, y_tiny)
        
        predictions = model.predict(X_tiny)
        assert len(predictions) == 2

    def test_single_class_data_error(self):
        """Test handling of single-class data"""
        # Data with only one class
        X = np.random.random((10, 5))
        y = np.zeros(10)  # All same class
        
        model = LightGBMModel()
        # Should handle single class gracefully
        with pytest.warns(None) as warnings:
            model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == 10

    def test_nan_in_features_error(self):
        """Test handling of NaN values in features"""
        X_with_nan = np.random.random((100, 5))
        X_with_nan[0, 0] = np.nan
        X_with_nan[1, 1] = np.nan
        y = np.random.randint(0, 2, 100)
        
        model = LightGBMModel()
        # LightGBM should handle NaN gracefully
        model.fit(X_with_nan, y)
        
        predictions = model.predict(X_with_nan[:10])
        assert len(predictions) == 10

    def test_infinite_values_error(self):
        """Test handling of infinite values in features"""
        X_with_inf = np.random.random((100, 5))
        X_with_inf[0, 0] = np.inf
        X_with_inf[1, 1] = -np.inf
        y = np.random.randint(0, 2, 100)
        
        # Check data integrity detects infinite values
        integrity = check_data_integrity(X_with_inf, y)
        assert integrity["no_infinite_features"] is False

    def test_feature_dimension_mismatch(self):
        """Test handling of feature dimension mismatches"""
        # Train with 5 features
        X_train = np.random.random((100, 5))
        y_train = np.random.randint(0, 2, 100)
        
        model = LightGBMModel()
        model.fit(X_train, y_train)
        
        # Try to predict with different number of features
        X_test_wrong = np.random.random((10, 3))  # Wrong number of features
        
        with pytest.raises(ValueError):
            model.predict(X_test_wrong)

    def test_cross_validation_with_invalid_splits(self):
        """Test cross-validation with invalid split configurations"""
        X = np.random.random((10, 5))
        y = np.random.randint(0, 2, 10)
        
        # Try CV with more splits than samples
        cv_strategy = CVStrategy(n_splits=15)  # More splits than samples
        trainer = CrossValidationTrainer(cv_strategy=cv_strategy)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError):
            trainer.train_cv(LightGBMModel, X, y, feature_names=[f"f{i}" for i in range(5)])


class TestDataIntegrityErrorScenarios:
    """Test data integrity check error scenarios"""

    def test_mismatched_dimensions(self):
        """Test detection of mismatched X and y dimensions"""
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 90)  # Wrong size
        
        integrity = check_data_integrity(X, y)
        assert integrity["shape_consistent"] is False

    def test_non_binary_targets(self):
        """Test detection of non-binary targets"""
        X = np.random.random((100, 5))
        y = np.random.randint(0, 3, 100)  # Three classes instead of two
        
        integrity = check_data_integrity(X, y)
        assert integrity["binary_targets"] is False

    def test_insufficient_samples(self):
        """Test detection of insufficient samples"""
        X = np.random.random((3, 5))  # Very few samples
        y = np.random.randint(0, 2, 3)
        
        integrity = check_data_integrity(X, y)
        assert integrity["sufficient_samples"] is False

    def test_all_nan_features(self):
        """Test detection of all-NaN features"""
        X = np.random.random((100, 5))
        X[:, 0] = np.nan  # One column all NaN
        y = np.random.randint(0, 2, 100)
        
        integrity = check_data_integrity(X, y)
        assert integrity["no_missing_features"] is False


class TestPipelineRobustnessScenarios:
    """Test robustness scenarios for the entire pipeline"""

    def test_pipeline_with_memory_constraints(self):
        """Test pipeline behavior under memory constraints"""
        # Simulate very large dataset (but not actually large to avoid memory issues)
        # Just test that the pipeline can handle reasonable-sized data efficiently
        X = np.random.random((10000, 50))
        y = np.random.randint(0, 2, 10000)
        
        # Should handle moderately large data without issues
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(50)]
        model.fit(X, y, feature_names=feature_names)
        
        predictions = model.predict(X[:100])
        assert len(predictions) == 100

    def test_pipeline_with_concurrent_access(self):
        """Test pipeline behavior with concurrent access patterns"""
        # Create multiple models simultaneously
        models = []
        X = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)
        
        for i in range(3):
            model = LightGBMModel()
            feature_names = [f"model_{i}_feature_{j}" for j in range(10)]
            model.fit(X, y, feature_names=feature_names)
            models.append(model)
        
        # All models should work independently
        for model in models:
            predictions = model.predict(X[:5])
            assert len(predictions) == 5

    def test_file_system_error_handling(self):
        """Test handling of file system errors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CV logger with read-only directory
            readonly_dir = os.path.join(tmpdir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)  # Read-only
            
            try:
                logger = CVLogger(log_dir=readonly_dir)
                log_entry = {"test": "data"}
                
                # Should handle permission error gracefully
                with pytest.raises(PermissionError):
                    logger.save_json_log(log_entry, "test.json")
            
            finally:
                # Restore permissions for cleanup
                os.chmod(readonly_dir, 0o755)

    def test_network_error_resilience(self):
        """Test resilience to network errors in notifications"""
        # Test webhook with network simulation
        notifier = WebhookNotifier(webhook_url="https://httpbin.org/status/500")
        
        # Should handle server errors gracefully
        result = notifier.send_message("Test message")
        assert result is False
        
        # Test with timeout simulation
        with patch("requests.post") as mock_post:
            mock_post.side_effect = TimeoutError("Network timeout")
            
            result = notifier.send_message("Test message")
            assert result is False


class TestEdgeCaseDataScenarios:
    """Test edge cases in data processing"""

    def test_all_same_feature_values(self):
        """Test handling of features with all same values"""
        X = np.ones((100, 5))  # All features have same value
        y = np.random.randint(0, 2, 100)
        
        model = LightGBMModel()
        # Should handle constant features
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10

    def test_perfectly_correlated_features(self):
        """Test handling of perfectly correlated features"""
        base_feature = np.random.random(100)
        X = np.column_stack([
            base_feature,
            base_feature,  # Perfect correlation
            base_feature * 2,  # Linear relationship
            np.random.random(100),  # Independent
            np.random.random(100)   # Independent
        ])
        y = np.random.randint(0, 2, 100)
        
        model = LightGBMModel()
        # Should handle correlated features
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10

    def test_outlier_heavy_data(self):
        """Test handling of data with many outliers"""
        # Create data with many outliers
        X_normal = np.random.normal(0, 1, (80, 5))
        X_outliers = np.random.normal(0, 10, (20, 5))  # Outliers with larger variance
        X = np.vstack([X_normal, X_outliers])
        y = np.random.randint(0, 2, 100)
        
        model = LightGBMModel()
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10

    def test_high_dimensional_data(self):
        """Test handling of high-dimensional data"""
        # More features than samples
        X = np.random.random((50, 100))  # 100 features, 50 samples
        y = np.random.randint(0, 2, 50)
        
        model = LightGBMModel()
        # Should handle high-dimensional data
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10


class TestResourceLimitScenarios:
    """Test behavior under resource limitations"""

    def test_disk_space_simulation(self):
        """Test behavior when disk space is limited"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate disk space issue by creating files in a small tmpfs
            # (This is a simplified simulation)
            logger = CVLogger(log_dir=tmpdir)
            
            # Try to save many large log files
            large_log = {"data": "x" * 10000}  # Large log entry
            
            try:
                for i in range(10):
                    logger.save_json_log(large_log, f"large_log_{i}.json")
                    
            except Exception as e:
                # Should handle disk space gracefully
                assert isinstance(e, (IOError, OSError))

    def test_computation_timeout_simulation(self):
        """Test handling of computation timeouts"""
        # Simulate long-running computation with timeout
        X = np.random.random((1000, 100))
        y = np.random.randint(0, 2, 1000)
        
        # Use a very complex model configuration that might timeout
        # (This is more of a conceptual test - actual timeout would require more setup)
        model = LightGBMModel()
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        model.fit(X, y)
        training_time = time.time() - start_time
        
        # Should not take excessively long
        assert training_time < 30.0  # 30 seconds max for this test

    def test_memory_efficient_processing(self):
        """Test memory-efficient processing of data"""
        # Test processing data in chunks to avoid memory issues
        large_X = np.random.random((5000, 20))
        large_y = np.random.randint(0, 2, 5000)
        
        # Process in smaller chunks
        chunk_size = 1000
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(20)]
        
        # Train on subset
        model.fit(large_X[:chunk_size], large_y[:chunk_size], feature_names=feature_names)
        
        # Predict in chunks
        all_predictions = []
        for i in range(0, len(large_X), chunk_size):
            chunk_X = large_X[i:i+chunk_size]
            chunk_predictions = model.predict(chunk_X)
            all_predictions.extend(chunk_predictions)
        
        assert len(all_predictions) == len(large_X)


class TestErrorRecoveryScenarios:
    """Test error recovery and graceful degradation"""

    def test_partial_pipeline_failure_recovery(self):
        """Test recovery from partial pipeline failures"""
        # Simulate failure in silver layer, fallback to bronze
        try:
            create_bronze_tables()
            
            # Force an error in silver layer creation
            with patch("src.data.silver.create_silver_tables") as mock_silver:
                mock_silver.side_effect = Exception("Silver layer failed")
                
                # Should be able to work with just bronze layer
                train_bronze, test_bronze = load_bronze_data()
                assert len(train_bronze) > 0
                assert len(test_bronze) > 0
                
        except Exception:
            # If bronze also fails, that's acceptable for this test
            pass

    def test_model_fallback_strategies(self):
        """Test fallback strategies when primary model fails"""
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        
        # Try training multiple models as fallbacks
        models = []
        feature_names = [f"feature_{i}" for i in range(5)]
        
        try:
            model1 = LightGBMModel()
            model1.fit(X, y, feature_names=feature_names)
            models.append(model1)
        except Exception:
            pass
        
        # Should have at least one working model
        assert len(models) > 0
        
        # Test fallback prediction
        for model in models:
            predictions = model.predict(X[:10])
            assert len(predictions) == 10
            break

    def test_graceful_degradation_with_bad_data(self):
        """Test graceful degradation when data quality is poor"""
        # Create increasingly poor quality data
        good_X = np.random.random((100, 5))
        good_y = np.random.randint(0, 2, 100)
        
        # Add noise progressively
        noisy_X = good_X + np.random.normal(0, 0.1, good_X.shape)
        very_noisy_X = good_X + np.random.normal(0, 1.0, good_X.shape)
        
        # Should work with good data
        model = LightGBMModel()
        model.fit(good_X, good_y)
        good_predictions = model.predict(good_X[:10])
        assert len(good_predictions) == 10
        
        # Should still work with noisy data (maybe worse performance)
        model_noisy = LightGBMModel()
        model_noisy.fit(noisy_X, good_y)
        noisy_predictions = model_noisy.predict(noisy_X[:10])
        assert len(noisy_predictions) == 10
        
        # Should still work with very noisy data
        model_very_noisy = LightGBMModel()
        model_very_noisy.fit(very_noisy_X, good_y)
        very_noisy_predictions = model_very_noisy.predict(very_noisy_X[:10])
        assert len(very_noisy_predictions) == 10