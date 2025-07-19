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

from src.data.bronze import basic_features, quick_preprocess
from src.data.gold import create_submission, encode_target, prepare_model_data
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
        X, y, feature_names = prepare_model_data(encoded_data)

        # 5. Data validation
        integrity_checks = check_data_integrity(X, y)
        assert integrity_checks["all_checks_passed"]

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
        test_ids = np.arange(1000, 1010)

        predictions = best_model.predict(X_test)
        submission = create_submission(test_ids, predictions)

        assert len(submission) == 10
        assert list(submission.columns) == ["id", "Personality"]


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

    @patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https: //test.webhook"})
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
        notifier = WebhookNotifier()

        # Test without webhook URL
        result = notifier.send_message("test")
        assert result is False

        # Test with webhook URL
        notifier.webhook_url = "https: //test.webhook"

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
        assert checks_valid["all_checks_passed"]
        assert checks_valid["has_nan"] is False
        assert checks_valid["has_inf"] is False
        assert checks_valid["shape_match"] is True
        assert checks_valid["min_samples_ok"] is True

        # Data with issues
        X_issues = X_valid.copy()
        X_issues[0, 0] = np.nan
        X_issues[1, 1] = np.inf

        checks_issues = check_data_integrity(X_issues, y_valid)
        assert checks_issues["all_checks_passed"] is False
        assert checks_issues["has_nan"] is True
        assert checks_issues["has_inf"] is True

    def test_target_distribution_scenarios(self):
        """Test target distribution with various scenarios"""
        # Balanced binary
        y_balanced = np.array([0, 1] * 50)
        dist_balanced = validate_target_distribution(y_balanced)
        assert dist_balanced["is_balanced"] is True
        assert dist_balanced["is_binary"] is True

        # Imbalanced binary
        y_imbalanced = np.array([0] * 80 + [1] * 20)
        dist_imbalanced = validate_target_distribution(y_imbalanced)
        assert dist_imbalanced["is_balanced"] is False
        assert dist_imbalanced["is_binary"] is True
        assert dist_imbalanced["imbalance_ratio"] == 0.8

        # Multiclass
        y_multi = np.array([0, 1, 2] * 30)
        dist_multi = validate_target_distribution(y_multi)
        assert dist_multi["is_binary"] is False
        assert dist_multi["n_classes"] == 3


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
