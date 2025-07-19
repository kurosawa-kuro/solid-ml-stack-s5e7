"""
Cross-Module Integration Tests
Tests for validating interactions between different modules
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.bronze import create_bronze_tables
from src.data.gold import create_gold_tables, get_ml_ready_data
from src.data.silver import create_silver_tables
from src.models import CrossValidationTrainer, LightGBMModel
from src.util.notifications import notify_complete, notify_start
from src.util.time_tracker import WorkflowTimeTracker, time_workflow
from src.validation import CVLogger, CVStrategy, calculate_prediction_distribution


class TestDataModelIntegration:
    """Test integration between data pipeline and model training"""

    def test_full_data_to_model_pipeline(self):
        """Test complete pipeline from data loading to model training"""
        # Setup data pipeline
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


class TestModelValidationIntegration:
    """Test integration between models and validation"""

    def test_cv_with_validation_logging(self):
        """Test cross-validation with validation logging"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup data
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


class TestNotificationIntegration:
    """Test integration of notifications with other modules"""

    @patch.dict("os.environ", {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_training_workflow_with_notifications(self, mock_post):
        """Test training workflow with notification integration"""
        # Mock successful webhook responses
        from unittest.mock import Mock
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        # Setup data
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


class TestTimeTrackingIntegration:
    """Test integration of time tracking with other modules"""

    def test_workflow_timing_with_model_training(self):
        """Test workflow timing integration with model training"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(db_path=f"{tmpdir}/times.json")
            
            # Setup data
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
            import time
            time.sleep(0.1)
            tracker.end_workflow("data_preparation", start_time)
            
            # Train with timing
            model = train_model()
            
            # Verify timing integration
            workflows = tracker.list_workflows()
            assert "data_preparation" in workflows
            # The decorator saves to a different tracker instance, so check it exists in the times file
            import json
            with open(f"{tmpdir}/times.json", "r") as f:
                times_data = json.load(f)
            assert "model_training" in times_data.get("workflows", {})
            
            stats = tracker.get_workflow_stats("data_preparation")
            assert stats["count"] == 1
            assert stats["average"] >= 0.1
            
            return model, tracker


class TestDataValidationIntegration:
    """Test integration between data pipeline and validation"""

    def test_data_pipeline_with_validation_checks(self):
        """Test data pipeline with integrated validation checks"""
        # Setup data pipeline
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


class TestFullSystemIntegration:
    """Test full system integration across all modules"""

    @patch.dict("os.environ", {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_complete_ml_system_integration(self, mock_post):
        """Test complete ML system integration"""
        # Mock webhook responses
        from unittest.mock import Mock
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            tracker = WorkflowTimeTracker(db_path=f"{tmpdir}/times.json")
            logger = CVLogger(log_dir=tmpdir)
            
            # 1. Data Pipeline with timing
            start_time = tracker.start_workflow("data_pipeline")
            create_bronze_tables()
            create_silver_tables()
            create_gold_tables()
            tracker.end_workflow("data_pipeline", start_time)
            
            # 2. Data preparation with validation
            X_train, y_train, X_test, test_ids = get_ml_ready_data(scale_features=True)
            
            from src.validation import check_data_integrity
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


class TestAPIConsistencyIntegration:
    """Test API consistency across modules"""

    def test_consistent_return_types(self):
        """Test that modules return consistent data types"""
        # Setup data
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