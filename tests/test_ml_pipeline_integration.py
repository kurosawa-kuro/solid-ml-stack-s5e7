"""
End-to-End ML Pipeline Integration Tests
Tests the complete ML workflow from raw data to predictions and submissions
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.data.bronze import create_bronze_tables, load_bronze_data
from src.data.gold import create_gold_tables, extract_model_arrays, get_ml_ready_data, load_gold_data
from src.data.silver import create_silver_tables, load_silver_data
from src.models import CrossValidationTrainer, LightGBMModel
from src.validation import CVLogger, CVStrategy, calculate_prediction_distribution


class TestMLPipelineIntegration:
    """Test complete ML pipeline integration"""

    def test_full_pipeline_bronze_to_predictions(self):
        """Test complete pipeline from bronze layer to final predictions"""
        # 1. Data Pipeline: Bronze → Silver → Gold
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        # 2. Load model-ready data
        X_train, y_train, X_test, test_ids = get_ml_ready_data(scale_features=True)
        
        # Verify data quality
        assert X_train.shape[0] > 0, "Should have training samples"
        assert X_train.shape[1] > 0, "Should have features"
        assert len(y_train) == len(X_train), "Target should match training samples"
        assert len(X_test) > 0, "Should have test samples"
        assert len(test_ids) == len(X_test), "Test IDs should match test samples"
        
        # 3. Model Training with Cross-Validation
        cv_strategy = CVStrategy(n_splits=3, random_state=42)  # Use 3 folds for speed
        trainer = CrossValidationTrainer(cv_strategy=cv_strategy)
        
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        cv_results = trainer.train_cv(
            model_class=LightGBMModel,
            X=X_train,
            y=y_train,
            feature_names=feature_names
        )
        
        # Verify CV results
        assert cv_results["mean_score"] > 0.5, "Should achieve better than random performance"
        assert len(cv_results["models"]) == 3, "Should have 3 CV models"
        assert len(cv_results["oof_predictions"]) == len(y_train), "OOF predictions should match training size"
        
        # 4. Generate Test Predictions
        best_model = cv_results["models"][0]
        test_predictions = best_model.predict(X_test)
        
        # Verify predictions
        assert len(test_predictions) == len(X_test), "Predictions should match test size"
        assert all(pred in [0, 1] for pred in test_predictions), "Should be binary predictions"
        
        # 5. Analyze Prediction Distribution
        pred_dist = calculate_prediction_distribution(test_predictions)
        assert pred_dist["total_predictions"] == len(test_predictions)
        assert pred_dist["extrovert_count"] + pred_dist["introvert_count"] == len(test_predictions)
        
        return {
            "cv_results": cv_results,
            "test_predictions": test_predictions,
            "prediction_distribution": pred_dist,
            "feature_names": feature_names
        }

    def test_pipeline_with_different_models(self):
        """Test pipeline with different model types"""
        # Setup data
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, X_test, test_ids = get_ml_ready_data()
        
        # Test with different models
        models_to_test = [LightGBMModel]  # Can add more when available
        
        results = {}
        cv_strategy = CVStrategy(n_splits=3, random_state=42)
        trainer = CrossValidationTrainer(cv_strategy=cv_strategy)
        
        for model_class in models_to_test:
            feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
            cv_results = trainer.train_cv(
                model_class=model_class,
                X=X_train,
                y=y_train,
                feature_names=feature_names
            )
            
            results[model_class.__name__] = cv_results
            
            # Basic performance checks
            assert cv_results["mean_score"] > 0.4, f"{model_class.__name__} should achieve reasonable performance"
            assert cv_results["std_score"] >= 0, f"{model_class.__name__} std should be non-negative"
        
        return results

    def test_feature_importance_extraction(self):
        """Test feature importance extraction through the pipeline"""
        # Setup data
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, _, _ = get_ml_ready_data()
        
        # Train model
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Extract feature importance
        importance = model.get_feature_importance()
        
        # Verify importance structure
        assert isinstance(importance, pd.DataFrame), "Importance should be DataFrame"
        assert "feature" in importance.columns, "Should have feature column"
        assert "importance" in importance.columns, "Should have importance column"
        assert len(importance) == len(feature_names), "Should have importance for all features"
        
        # Verify importance values
        assert (importance["importance"] >= 0).all(), "Importance should be non-negative"
        assert importance["importance"].sum() > 0, "At least some features should be important"
        
        return importance

    def test_cross_validation_consistency(self):
        """Test consistency of cross-validation results"""
        # Setup data
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, _, _ = get_ml_ready_data()
        
        # Run CV multiple times with same seed
        cv_strategy = CVStrategy(n_splits=3, random_state=42)
        trainer = CrossValidationTrainer(cv_strategy=cv_strategy)
        
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        
        results1 = trainer.train_cv(
            model_class=LightGBMModel,
            X=X_train,
            y=y_train,
            feature_names=feature_names
        )
        
        results2 = trainer.train_cv(
            model_class=LightGBMModel,
            X=X_train,
            y=y_train,
            feature_names=feature_names
        )
        
        # Results should be similar (but not identical due to some randomness)
        score_diff = abs(results1["mean_score"] - results2["mean_score"])
        assert score_diff < 0.1, "CV results should be reasonably consistent"

    def test_pipeline_with_minimal_data(self):
        """Test pipeline works with minimal data subset"""
        # Create tables first
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        # Load full data
        train_gold, test_gold = load_gold_data()
        
        # Take small subset for testing
        train_subset = train_gold.head(100).copy()
        test_subset = test_gold.head(50).copy()
        
        # Extract arrays from subsets
        X_train, y_train, feature_names = extract_model_arrays(train_subset)
        # Use the same feature columns as training for test data
        X_test = test_subset[feature_names].values
        
        # Test training on small data
        model = LightGBMModel()
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Test predictions
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)

    def test_data_leakage_prevention(self):
        """Test that the pipeline prevents data leakage"""
        # Setup data
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, X_test, _ = get_ml_ready_data(scale_features=True)
        
        # Check that test data doesn't contain target information
        # (This is mainly about checking the pipeline doesn't accidentally include target in features)
        
        # Train model on first half, test on second half
        split_idx = len(X_train) // 2
        X_train_1, y_train_1 = X_train[:split_idx], y_train[:split_idx]
        X_train_2, y_train_2 = X_train[split_idx:], y_train[split_idx:]
        
        # Train on first half
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train_1.shape[1])]
        model.fit(X_train_1, y_train_1, feature_names=feature_names)
        
        # Predict on second half
        predictions = model.predict(X_train_2)
        accuracy = accuracy_score(y_train_2, predictions)
        
        # Should be reasonable but not perfect (perfect would indicate leakage)
        assert 0.5 <= accuracy <= 0.99, f"Accuracy {accuracy} suggests potential leakage or poor model"


class TestPipelineRobustness:
    """Test pipeline robustness and error handling"""

    def test_pipeline_with_missing_values(self):
        """Test pipeline handles missing values correctly"""
        # Create synthetic data with missing values
        np.random.seed(42)
        n_samples, n_features = 100, 10
        
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 2, n_samples)
        
        # Introduce missing values
        missing_mask = np.random.random((n_samples, n_features)) < 0.1
        X[missing_mask] = np.nan
        
        # Test that model can handle this
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(n_features)]
        
        # Should handle NaN gracefully
        model.fit(X, y, feature_names=feature_names)
        predictions = model.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

    def test_pipeline_empty_data_handling(self):
        """Test pipeline behavior with edge cases"""
        # Test with minimal data
        X_tiny = np.random.random((5, 3))
        y_tiny = np.array([0, 1, 0, 1, 0])
        
        model = LightGBMModel()
        feature_names = ["f1", "f2", "f3"]
        
        # Should work with tiny data
        model.fit(X_tiny, y_tiny, feature_names=feature_names)
        predictions = model.predict(X_tiny)
        
        assert len(predictions) == 5

    def test_cv_logging_integration(self):
        """Test CV logging integration with the pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            create_bronze_tables()
            create_silver_tables()
            create_gold_tables()
            
            X_train, y_train, _, _ = get_ml_ready_data()
            
            # Setup CV logger
            logger = CVLogger(log_dir=tmpdir)
            
            # Run CV
            cv_strategy = CVStrategy(n_splits=3, random_state=42)
            trainer = CrossValidationTrainer(cv_strategy=cv_strategy)
            
            feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
            cv_results = trainer.train_cv(
                model_class=LightGBMModel,
                X=X_train,
                y=y_train,
                feature_names=feature_names
            )
            
            # Create log entry
            log_entry = logger.create_log_entry(
                model_type="LightGBM",
                cv_config=cv_strategy.get_config(),
                fold_scores=cv_results["fold_scores"],
                training_time=cv_results.get("training_time", 60.0),
                experiment_name="pipeline_test",
                feature_count=len(feature_names)
            )
            
            # Save log
            log_path = logger.save_json_log(log_entry, "pipeline_test.json")
            
            # Verify log was saved
            assert os.path.exists(log_path)
            
            # Verify log content
            with open(log_path, 'r') as f:
                saved_log = json.load(f)
            
            assert saved_log["model_type"] == "LightGBM"
            assert saved_log["experiment_name"] == "pipeline_test"
            assert len(saved_log["fold_scores"]) == 3


class TestPipelinePerformance:
    """Test pipeline performance characteristics"""

    def test_training_speed_benchmarks(self):
        """Test training speed meets reasonable benchmarks"""
        import time
        
        # Setup data
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, _, _ = get_ml_ready_data()
        
        # Benchmark single model training
        start_time = time.time()
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        model.fit(X_train, y_train, feature_names=feature_names)
        training_time = time.time() - start_time
        
        # Should train reasonably quickly
        assert training_time < 10.0, f"Training took {training_time:.2f}s, should be under 10s"
        
        # Benchmark prediction speed
        start_time = time.time()
        predictions = model.predict(X_train[:100])
        prediction_time = time.time() - start_time
        
        assert prediction_time < 1.0, f"Prediction took {prediction_time:.2f}s, should be under 1s"

    def test_memory_usage_reasonable(self):
        """Test that memory usage stays reasonable"""
        import psutil
        import os
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Setup and run pipeline
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, X_test, _ = get_ml_ready_data()
        
        # Train model
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - baseline_memory
        
        # Should not use excessive memory (adjust threshold as needed)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"


class TestPipelineOutputFormats:
    """Test pipeline output formats and compatibility"""

    def test_submission_format_compatibility(self):
        """Test that pipeline outputs are compatible with Kaggle submission format"""
        # Setup data
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, X_test, test_ids = get_ml_ready_data()
        
        # Train model and get predictions
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        model.fit(X_train, y_train, feature_names=feature_names)
        
        predictions = model.predict(X_test)
        
        # Create submission format
        submission = pd.DataFrame({
            "id": test_ids,
            "Personality": ["Extrovert" if pred == 1 else "Introvert" for pred in predictions]
        })
        
        # Verify submission format
        assert len(submission) == len(X_test), "Submission should have all test samples"
        assert list(submission.columns) == ["id", "Personality"], "Should have correct columns"
        assert submission["Personality"].isin(["Extrovert", "Introvert"]).all(), "Should have valid personality labels"
        assert submission["id"].notna().all(), "All IDs should be present"
        assert len(submission["id"].unique()) == len(submission), "IDs should be unique"

    def test_model_serialization_compatibility(self):
        """Test that trained models can be serialized and deserialized"""
        import pickle
        
        # Setup and train model
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, X_test, _ = get_ml_ready_data()
        
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Get original predictions
        original_predictions = model.predict(X_test)
        
        # Serialize and deserialize
        with tempfile.NamedTemporaryFile() as tmp:
            pickle.dump(model, tmp)
            tmp.flush()
            tmp.seek(0)
            loaded_model = pickle.load(tmp)
        
        # Test loaded model
        loaded_predictions = loaded_model.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(original_predictions, loaded_predictions)