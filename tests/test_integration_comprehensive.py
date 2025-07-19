"""
Integration tests for complete ML workflows and remaining coverage gaps
"""

import json
import os
import tempfile
import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pathlib import Path
import pytest

from src.data.bronze import basic_features, quick_preprocess
from src.data.gold import create_submission, encode_target, prepare_model_data, extract_model_arrays
from src.data.silver import advanced_features, scaling_features
from src.models import CrossValidationTrainer, LightGBMModel
from src.util.notifications import WebhookNotifier, notify_complete, notify_start
from src.util.time_tracker import WorkflowTimer, WorkflowTimeTracker, time_workflow
from src.validation import CVLogger, CVStrategy, check_data_integrity, validate_target_distribution


class TestCompleteMLWorkflow:
    """Integration test for complete ML workflow"""

    def test_end_to_end_ml_pipeline(self):
        """Test a complete end-to-end ML pipeline"""
        # 1. Create sample data (simulating bronze layer)
        raw_data = pd.DataFrame(
            {
                "id": range(100),
                "Time_spent_Alone": np.random.randint(1, 10, 100),
                "Social_event_attendance": np.random.randint(1, 10, 100),
                "Going_outside": np.random.randint(0, 5, 100),
                "Friends_circle_size": np.random.randint(5, 100, 100),
                "Post_frequency": np.random.randint(1, 30, 100),
                "Stage_fear": np.random.choice(["Yes", "No"], 100),
                "Drained_after_socializing": np.random.choice(["Yes", "No"], 100),
                "Personality": np.random.choice(["Introvert", "Extrovert"], 100),
            }
        )

        # 2. Bronze processing
        processed_data = quick_preprocess(raw_data)
        featured_data = basic_features(processed_data)

        # 3. Silver processing
        advanced_data = advanced_features(featured_data)
        scaled_data = scaling_features(advanced_data)

        # 4. Gold processing
        encoded_data = encode_target(scaled_data)
        model_ready_data = prepare_model_data(encoded_data, target_col="Personality")
        X, y, feature_names = extract_model_arrays(model_ready_data)

        # 5. Data validation
        integrity_checks = check_data_integrity(X, y)
        assert integrity_checks["shape_consistent"]
        assert integrity_checks["no_missing_features"]
        assert integrity_checks["binary_targets"]

        target_dist = validate_target_distribution(y)
        assert target_dist["is_binary"]

        # 6. Model training with CV
        trainer = CrossValidationTrainer()
        cv_results = trainer.train_cv(model_class=LightGBMModel, X=X, y=y, feature_names=feature_names)

        # 7. Verify results
        assert cv_results["mean_score"] > 0.4  # Should be better than random
        assert len(cv_results["models"]) == 5
        assert len(cv_results["oof_predictions"]) == len(y)

        # 8. Create predictions for new data
        best_model = cv_results["models"][0]
        X_test = X[:10]  # Use first 10 samples as test
        
        predictions = best_model.predict(X_test)
        
        # For create_submission test, we need to create a temporary test file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            # Note: create_submission expects predictions array and saves to file
            # We'll modify this to be more testable
            temp_filename = tmp.name
        
        # Test that predictions are reasonable
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)  # Binary predictions


class TestTimeTrackingIntegration:
    """Test time tracking functionality"""

    def test_workflow_time_tracking(self):
        """Test complete workflow time tracking"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "times.json")
            tracker = WorkflowTimeTracker(db_path)

            # Test basic workflow timing
            start_time = tracker.start_workflow("data_processing")
            time.sleep(0.1)
            tracker.end_workflow("data_processing", start_time)

            # Test context manager
            with WorkflowTimer(tracker, "model_training"):
                time.sleep(0.1)

            # Test decorator
            @time_workflow("prediction", db_path)
            def make_predictions():
                time.sleep(0.1)
                return "predictions_made"

            result = make_predictions()
            assert result == "predictions_made"

            # Verify all workflows were tracked
            workflows = tracker.list_workflows()
            assert "data_processing" in workflows
            assert "model_training" in workflows
            assert "prediction" in workflows

            # Test statistics
            stats = tracker.get_workflow_stats("data_processing")
            assert stats["count"] == 1
            assert stats["average"] > 0.1


class TestNotificationSystem:
    """Test notification system functionality"""

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_notification_workflow(self, mock_post):
        """Test complete notification workflow"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        # Test training start notification
        config = {"model": "LightGBM", "features": 10}
        result1 = notify_start("TestModel", config)
        assert result1 is True

        # Test completion notification
        metrics = {"accuracy": 0.95, "auc": 0.98}
        result2 = notify_complete("TestModel", metrics, duration=120.5)
        assert result2 is True

        # Verify webhook was called
        assert mock_post.call_count == 2

    def test_webhook_notifier_methods(self):
        """Test WebhookNotifier methods"""
        # Test with webhook URL provided
        notifier = WebhookNotifier(webhook_url="https://test.webhook")

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 204
            mock_post.return_value = mock_response

            # Test different notification types
            result1 = notifier.notify_training_start("Model", {})
            result2 = notifier.notify_training_complete("Model", {"acc": 0.9}, 100)
            result3 = notifier.notify_error("training", "Test error")
            result4 = notifier.notify_submission(0.95, rank=10, improvement=0.01)

            assert all([result1, result2, result3, result4])
            assert mock_post.call_count == 4


class TestCVLoggingIntegration:
    """Test CV logging functionality"""

    def test_cv_logging_workflow(self):
        """Test complete CV logging workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CVLogger(log_dir=tmpdir)

            # Create realistic CV results
            cv_results = {
                "mean_score": 0.9567,
                "std_score": 0.0123,
                "fold_scores": [0.95, 0.96, 0.97, 0.94, 0.96],
                "mean_auc": 0.9789,
                "std_auc": 0.0089,
                "training_time": 157.3,
            }

            model_params = {
                "objective": "binary",
                "learning_rate": 0.1,
                "num_leaves": 31,
                "random_state": 42,
            }

            feature_names = [f"feature_{i}" for i in range(15)]
            cv_config = {"n_splits": 5, "shuffle": True}

            # Create log entry
            log_entry = logger.create_log_entry(
                experiment_name="integration_test",
                model_type="LightGBM",
                cv_results=cv_results,
                model_params=model_params,
                feature_names=feature_names,
                cv_config=cv_config,
                fold_scores=cv_results["fold_scores"],
                training_time=cv_results["training_time"],
                notes="Integration test run",
            )

            # Save as JSON
            json_path = logger.save_json_log(log_entry, "integration_test.json")
            assert json_path.exists()

            # Create and save fold results CSV
            fold_df = pd.DataFrame(
                {
                    "fold": range(5),
                    "accuracy": cv_results["fold_scores"],
                    "auc": [0.98, 0.97, 0.99, 0.97, 0.98],
                }
            )

            csv_path = logger.save_csv_log(fold_df, "fold_results.csv")
            assert csv_path.exists()

            # Verify file contents
            with open(json_path, "r") as f:
                saved_log = json.load(f)
            assert saved_log["experiment_name"] == "integration_test"
            assert saved_log["cv_score"] == 0.9567

            saved_df = pd.read_csv(csv_path)
            assert len(saved_df) == 5
            assert saved_df["accuracy"].mean() == np.mean(cv_results["fold_scores"])


class TestDataValidationIntegration:
    """Test data validation with realistic scenarios"""

    def test_data_integrity_comprehensive(self):
        """Test comprehensive data integrity checking"""
        # Valid data
        X_valid = np.random.random((100, 10))
        y_valid = np.random.randint(0, 2, 100)

        checks_valid = check_data_integrity(X_valid, y_valid)
        assert checks_valid["shape_consistent"]
        assert checks_valid["no_missing_features"]
        assert checks_valid["no_infinite_features"]
        assert checks_valid["binary_targets"]
        assert checks_valid["sufficient_samples"]

        # Data with issues
        X_issues = X_valid.copy()
        X_issues[0, 0] = np.nan
        X_issues[1, 1] = np.inf

        checks_issues = check_data_integrity(X_issues, y_valid)
        assert checks_issues["no_missing_features"] is False  # Has NaN
        assert checks_issues["no_infinite_features"] is False  # Has inf

    def test_target_distribution_scenarios(self):
        """Test target distribution with various scenarios"""
        # Balanced binary
        y_balanced = np.array([0, 1] * 50)
        dist_balanced = validate_target_distribution(y_balanced)
        assert dist_balanced["is_binary"] is True
        assert dist_balanced["has_both_classes"] is True

        # Imbalanced binary
        y_imbalanced = np.array([0] * 80 + [1] * 20)
        dist_imbalanced = validate_target_distribution(y_imbalanced)
        assert dist_imbalanced["is_binary"] is True
        assert dist_imbalanced["has_both_classes"] is True

        # Multiclass
        y_multi = np.array([0, 1, 2] * 30)
        dist_multi = validate_target_distribution(y_multi)
        assert dist_multi["is_binary"] is False
        assert len(dist_multi["unique_values"]) == 3


class TestModelIntegrationScenarios:
    """Test model integration scenarios"""

    def test_model_training_with_edge_cases(self):
        """Test model training with various edge cases"""
        # Test with minimal data
        X_small = np.random.random((50, 3))
        y_small = np.random.randint(0, 2, 50)

        model = LightGBMModel()
        model.fit(X_small, y_small)

        # Should still work with small data
        assert model.is_fitted
        predictions = model.predict(X_small[:10])
        assert len(predictions) == 10

        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 3

        # Test with many features
        X_wide = np.random.random((100, 50))
        y_wide = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i: 02d}" for i in range(50)]

        model_wide = LightGBMModel()
        model_wide.fit(X_wide, y_wide, feature_names=feature_names)

        importance_wide = model_wide.get_feature_importance()
        assert len(importance_wide) == 50
        assert set(importance_wide["feature"]) == set(feature_names)

    def test_cross_validation_robust(self):
        """Test robust cross-validation"""
        # Create data with clear pattern
        n_samples = 200
        X = np.random.random((n_samples, 5))
        # Create target with some signal
        y = ((X[:, 0] + X[:, 1]) > 1).astype(int)

        # Test with different CV strategies
        strategy_3fold = CVStrategy(n_splits=3, random_state=42)
        trainer = CrossValidationTrainer(cv_strategy=strategy_3fold)

        results = trainer.train_cv(model_class=LightGBMModel, X=X, y=y, feature_names=[f"f{i}" for i in range(5)])

        # Should achieve reasonable performance on this simple task
        assert results["mean_score"] > 0.6
        assert len(results["fold_scores"]) == 3
        assert results["std_score"] >= 0  # Non-negative

        # Test model consistency
        assert len(results["models"]) == 3
        for model in results["models"]:
            assert model.is_fitted
            test_pred = model.predict(X[:5])
            assert len(test_pred) == 5


class TestDataModelIntegrationFromCross:
    """Integrated from test_cross_module_integration.py"""

    def test_full_data_to_model_pipeline(self):
        """Test complete pipeline from data loading to model training"""
        # Setup data pipeline
        from src.data.bronze import create_bronze_tables
        from src.data.silver import create_silver_tables
        from src.data.gold import create_gold_tables, get_ml_ready_data
        from src.validation import calculate_prediction_distribution
        
        create_bronze_tables()
        create_silver_tables() 
        create_gold_tables()
        
        # Get model-ready data
        X_train, y_train, X_test, test_ids = get_ml_ready_data(scale_features=True)
        
        # Train model
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Get predictions
        predictions = model.predict(X_test)
        
        # Validate integration worked
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test prediction distribution
        pred_dist = calculate_prediction_distribution(predictions)
        assert pred_dist["total_predictions"] == len(predictions)
        
        return {
            "data_shape": X_train.shape,
            "model": model,
            "predictions": predictions,
            "distribution": pred_dist
        }


class TestModelValidationIntegrationFromCross:
    """Test integration between models and validation from cross module"""

    def test_cv_with_validation_logging(self):
        """Test cross-validation with validation logging"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup data
            from src.data.bronze import create_bronze_tables
            from src.data.silver import create_silver_tables
            from src.data.gold import create_gold_tables, get_ml_ready_data
            
            create_bronze_tables()
            create_silver_tables()
            create_gold_tables()
            
            X_train, y_train, _, _ = get_ml_ready_data()
            
            # Setup CV and logging
            cv_strategy = CVStrategy(n_splits=3, random_state=42)
            trainer = CrossValidationTrainer(cv_strategy=cv_strategy)
            logger = CVLogger(log_dir=tmpdir)
            
            # Run CV
            feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
            cv_results = trainer.train_cv(
                model_class=LightGBMModel,
                X=X_train,
                y=y_train,
                feature_names=feature_names
            )
            
            # Log results
            log_entry = logger.create_log_entry(
                model_type="LightGBM",
                cv_config=cv_strategy.get_config(),
                fold_scores=cv_results["fold_scores"],
                training_time=60.0,
                experiment_name="cross_module_test"
            )
            
            log_path = logger.save_json_log(log_entry, "cv_integration_test.json")
            
            # Verify integration
            assert cv_results["mean_score"] > 0.4
            assert len(cv_results["models"]) == 3
            assert Path(log_path).exists()
            
            return cv_results


class TestNotificationIntegrationFromCross:
    """Test integration of notifications with other modules from cross module"""

    @patch.dict("os.environ", {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_training_workflow_with_notifications(self, mock_post):
        """Test training workflow with notification integration"""
        # Mock successful webhook responses
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        # Setup data
        from src.data.bronze import create_bronze_tables
        from src.data.silver import create_silver_tables
        from src.data.gold import create_gold_tables, get_ml_ready_data
        
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, _, _ = get_ml_ready_data()
        
        # Start notification
        config = {"model": "LightGBM", "features": X_train.shape[1]}
        notify_start("CrossModuleTest", config)
        
        # Train model
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Complete notification
        metrics = {"accuracy": 0.95, "features": X_train.shape[1]}
        notify_complete("CrossModuleTest", metrics, 120.0)
        
        # Verify notifications were sent
        assert mock_post.call_count == 2
        
        return model


class TestTimeTrackingIntegrationFromCross:
    """Test integration of time tracking with other modules from cross module"""

    def test_workflow_timing_with_model_training(self):
        """Test workflow timing integration with model training"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(db_path=f"{tmpdir}/times.json")
            
            # Setup data
            from src.data.bronze import create_bronze_tables
            from src.data.silver import create_silver_tables
            from src.data.gold import create_gold_tables, get_ml_ready_data
            
            create_bronze_tables()
            create_silver_tables()
            create_gold_tables()
            
            X_train, y_train, _, _ = get_ml_ready_data()
            
            # Time the training workflow
            @time_workflow("model_training", f"{tmpdir}/times.json")
            def train_model():
                model = LightGBMModel()
                feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
                model.fit(X_train, y_train, feature_names=feature_names)
                return model
            
            # Time data preparation
            start_time = tracker.start_workflow("data_preparation")
            # Simulate data prep time
            time.sleep(0.1)
            tracker.end_workflow("data_preparation", start_time)
            
            # Train with timing
            model = train_model()
            
            # Verify timing integration
            workflows = tracker.list_workflows()
            assert "data_preparation" in workflows
            # The decorator saves to a different tracker instance, so check it exists in the times file
            with open(f"{tmpdir}/times.json", "r") as f:
                times_data = json.load(f)
            assert "model_training" in times_data.get("workflows", {})
            
            stats = tracker.get_workflow_stats("data_preparation")
            assert stats["count"] == 1
            assert stats["average"] >= 0.1
            
            return model, tracker


class TestDataValidationIntegrationFromCross:
    """Test integration between data pipeline and validation from cross module"""

    def test_data_pipeline_with_validation_checks(self):
        """Test data pipeline with integrated validation checks"""
        # Setup data pipeline
        from src.data.bronze import create_bronze_tables
        from src.data.silver import create_silver_tables
        from src.data.gold import create_gold_tables, get_ml_ready_data
        
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        # Get data and validate
        X_train, y_train, X_test, test_ids = get_ml_ready_data()
        
        # Import validation functions
        from src.validation import check_data_integrity, validate_target_distribution
        
        # Check data integrity
        integrity = check_data_integrity(X_train, y_train)
        
        # Verify integration
        assert integrity["shape_consistent"]
        assert integrity["no_missing_features"]
        assert integrity["binary_targets"]
        
        # Check target distribution
        target_dist = validate_target_distribution(y_train)
        assert target_dist["is_binary"]
        assert target_dist["has_both_classes"]
        
        return {
            "integrity_checks": integrity,
            "target_distribution": target_dist,
            "data_shape": X_train.shape
        }


class TestFullSystemIntegrationFromCross:
    """Test full system integration across all modules from cross module"""

    @patch.dict("os.environ", {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_complete_ml_system_integration(self, mock_post):
        """Test complete ML system integration"""
        # Mock webhook responses
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            tracker = WorkflowTimeTracker(db_path=f"{tmpdir}/times.json")
            logger = CVLogger(log_dir=tmpdir)
            
            # 1. Data Pipeline with timing
            start_time = tracker.start_workflow("data_pipeline")
            from src.data.bronze import create_bronze_tables
            from src.data.silver import create_silver_tables
            from src.data.gold import create_gold_tables, get_ml_ready_data
            from src.validation import check_data_integrity, calculate_prediction_distribution
            
            create_bronze_tables()
            create_silver_tables()
            create_gold_tables()
            tracker.end_workflow("data_pipeline", start_time)
            
            # 2. Data preparation with validation
            X_train, y_train, X_test, test_ids = get_ml_ready_data(scale_features=True)
            
            integrity = check_data_integrity(X_train, y_train)
            assert integrity["shape_consistent"]
            
            # 3. Model training with notifications and CV
            config = {"model": "LightGBM", "features": X_train.shape[1], "samples": len(X_train)}
            notify_start("FullSystemTest", config)
            
            cv_strategy = CVStrategy(n_splits=3, random_state=42)
            trainer = CrossValidationTrainer(cv_strategy=cv_strategy)
            
            feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
            cv_results = trainer.train_cv(
                model_class=LightGBMModel,
                X=X_train,
                y=y_train,
                feature_names=feature_names
            )
            
            # 4. Logging and notifications
            log_entry = logger.create_log_entry(
                model_type="LightGBM",
                cv_config=cv_strategy.get_config(),
                fold_scores=cv_results["fold_scores"],
                training_time=60.0,
                experiment_name="full_system_test",
                data_shape=X_train.shape
            )
            
            log_path = logger.save_json_log(log_entry, "full_system_test.json")
            
            metrics = {
                "cv_score": cv_results["mean_score"],
                "cv_std": cv_results["std_score"],
                "features": X_train.shape[1]
            }
            notify_complete("FullSystemTest", metrics, 180.0)
            
            # 5. Predictions and analysis
            best_model = cv_results["models"][0]
            predictions = best_model.predict(X_test)
            
            pred_dist = calculate_prediction_distribution(predictions)
            
            # Verify full integration
            assert cv_results["mean_score"] > 0.4
            assert len(predictions) == len(X_test)
            assert mock_post.call_count == 2  # start + complete notifications
            assert Path(log_path).exists()
            
            workflows = tracker.list_workflows()
            assert "data_pipeline" in workflows
            
            return {
                "cv_results": cv_results,
                "predictions": predictions,
                "prediction_distribution": pred_dist,
                "log_path": log_path,
                "workflows": workflows
            }


class TestAPIConsistencyIntegrationFromCross:
    """Test API consistency across modules from cross module"""

    def test_consistent_return_types(self):
        """Test that modules return consistent data types"""
        # Setup data
        from src.data.bronze import create_bronze_tables
        from src.data.silver import create_silver_tables
        from src.data.gold import create_gold_tables, get_ml_ready_data
        
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        # Test data loading returns
        X_train, y_train, X_test, test_ids = get_ml_ready_data()
        
        # Verify types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(test_ids, np.ndarray)
        
        # Test model returns
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        model.fit(X_train, y_train, feature_names=feature_names)
        
        predictions = model.predict(X_test)
        importance = model.get_feature_importance()
        
        assert isinstance(predictions, np.ndarray)
        assert isinstance(importance, pd.DataFrame)
        
        # Test validation returns
        from src.validation import check_data_integrity, validate_target_distribution
        integrity = check_data_integrity(X_train, y_train)
        target_dist = validate_target_distribution(y_train)
        
        assert isinstance(integrity, dict)
        assert isinstance(target_dist, dict)
        
        return True

    def test_consistent_error_handling(self):
        """Test that modules handle errors consistently"""
        # Test with invalid data
        X_invalid = np.array([[1, 2], [3, 4]])  # Wrong shape for trained model
        
        # Setup a trained model
        from src.data.bronze import create_bronze_tables
        from src.data.silver import create_silver_tables
        from src.data.gold import create_gold_tables, get_ml_ready_data
        
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        X_train, y_train, _, _ = get_ml_ready_data()
        
        model = LightGBMModel()
        feature_names = [f"feature_{i:02d}" for i in range(X_train.shape[1])]
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Should raise consistent error types
        with pytest.raises(ValueError):
            model.predict(X_invalid)
        
        return True
