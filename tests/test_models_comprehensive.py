"""
Comprehensive test cases for models.py to achieve 95% coverage
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from src.models import (
    LIGHTGBM_PARAMS,
    CrossValidationTrainer,
    LightGBMModel,
    create_learning_curve_data,
    evaluate_model_metrics,
    load_model_with_metadata,
    save_model_with_metadata,
)


class TestLightGBMModelFull:
    """Comprehensive tests for LightGBMModel class"""

    def test_model_initialization_default(self):
        """Test model initialization with default parameters"""
        model = LightGBMModel()
        assert model.params == LIGHTGBM_PARAMS
        assert model.model is None
        assert model.feature_names is None
        assert not model.is_fitted

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters"""
        custom_params = {"learning_rate": 0.05, "num_leaves": 50}
        model = LightGBMModel(params=custom_params)

        # Should merge with default params
        assert model.params["learning_rate"] == 0.05
        assert model.params["num_leaves"] == 50
        assert model.params["objective"] == "binary"  # from default

    def test_model_fit(self):
        """Test model fitting"""
        model = LightGBMModel()

        # Create sample data
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        # Fit model
        model.fit(X, y, feature_names=feature_names)

        assert model.is_fitted
        assert model.feature_names == feature_names
        assert model.model is not None

    def test_model_fit_with_validation(self):
        """Test model fitting with validation set"""
        model = LightGBMModel()

        # Create sample data
        X_train = np.random.random((80, 5))
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.random((20, 5))
        y_val = np.random.randint(0, 2, 20)

        # Fit model with validation
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        assert model.is_fitted

    def test_model_predict(self):
        """Test model prediction"""
        model = LightGBMModel()

        # Train model first
        X_train = np.random.random((100, 5))
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)

        # Make predictions
        X_test = np.random.random((20, 5))
        predictions = model.predict(X_test)

        assert len(predictions) == 20
        assert all(p in [0, 1] for p in predictions)

    def test_model_predict_not_fitted(self):
        """Test prediction error when model not fitted"""
        model = LightGBMModel()
        X_test = np.random.random((20, 5))

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X_test)

    def test_model_predict_proba(self):
        """Test probability prediction"""
        model = LightGBMModel()

        # Train model first
        X_train = np.random.random((100, 5))
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train)

        # Get probabilities
        X_test = np.random.random((20, 5))
        probabilities = model.predict_proba(X_test)

        assert probabilities.shape == (20, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        model = LightGBMModel()

        # Train model first
        X_train = np.random.random((100, 5))
        y_train = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(5)]
        model.fit(X_train, y_train, feature_names=feature_names)

        # Get feature importance
        importance_df = model.get_feature_importance()

        assert len(importance_df) == 5
        assert list(importance_df.columns) == ["feature", "importance"]
        assert all(f in importance_df["feature"].values for f in feature_names)

    def test_model_save_load(self):
        """Test model save and load functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LightGBMModel()

            # Train model
            X_train = np.random.random((100, 5))
            y_train = np.random.randint(0, 2, 100)
            feature_names = [f"feature_{i}" for i in range(5)]
            model.fit(X_train, y_train, feature_names=feature_names)

            # Save model
            filepath = os.path.join(tmpdir, "model.pkl")
            model.save(filepath)

            assert os.path.exists(filepath)

            # Load model
            loaded_model = LightGBMModel.load(filepath)

            assert loaded_model.is_fitted
            assert loaded_model.feature_names == feature_names
            assert loaded_model.params == model.params

    def test_model_save_not_fitted(self):
        """Test save error when model not fitted"""
        model = LightGBMModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pkl")

            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                model.save(filepath)


class TestCrossValidationTrainerFull:
    """Comprehensive tests for CrossValidationTrainer class"""

    def test_cv_trainer_initialization(self):
        """Test CV trainer initialization"""
        trainer = CrossValidationTrainer()
        assert trainer.cv_strategy is not None
        assert hasattr(trainer.cv_strategy, "get_splits")

    def test_cv_trainer_initialization_custom(self):
        """Test CV trainer with custom strategy"""
        from src.validation import CVStrategy

        custom_strategy = CVStrategy(n_splits=3, shuffle=False)
        trainer = CrossValidationTrainer(cv_strategy=custom_strategy)
        assert trainer.cv_strategy.n_splits == 3
        assert not trainer.cv_strategy.shuffle

    def test_train_cv_basic(self):
        """Test basic cross-validation training"""
        trainer = CrossValidationTrainer()

        # Create sample data
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(5)]

        # Train with CV
        results = trainer.train_cv(model_class=LightGBMModel, X=X, y=y, feature_names=feature_names)

        # Check results structure
        assert "fold_scores" in results
        assert "mean_score" in results
        assert "std_score" in results
        assert "oof_predictions" in results
        assert "models" in results
        assert "training_time" in results

        # Check values
        assert len(results["fold_scores"]) == 5  # default 5 folds
        assert len(results["oof_predictions"]) == 100
        assert results["mean_score"] >= 0 and results["mean_score"] <= 1
        assert results["training_time"] > 0

    def test_train_cv_with_model_params(self):
        """Test CV with custom model parameters"""
        trainer = CrossValidationTrainer()

        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)

        model_params = {"learning_rate": 0.05, "num_leaves": 20}

        results = trainer.train_cv(model_class=LightGBMModel, X=X, y=y, model_params=model_params)

        # Check that models were created with custom params
        assert len(results["models"]) == 5
        for model in results["models"]:
            assert model.params["learning_rate"] == 0.05
            assert model.params["num_leaves"] == 20

    @patch("src.models.logger")
    def test_train_cv_logging(self, mock_logger):
        """Test CV logging"""
        trainer = CrossValidationTrainer()

        X = np.random.random((50, 3))
        y = np.random.randint(0, 2, 50)

        trainer.train_cv(model_class=LightGBMModel, X=X, y=y)

        # Check that logging was called
        assert mock_logger.info.called
        assert any("Starting cross-validation" in str(call) for call in mock_logger.info.call_args_list)


class TestModelUtilitiesFull:
    """Comprehensive tests for model utility functions"""

    def test_evaluate_model_metrics(self):
        """Test model evaluation function"""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.3])

        metrics = evaluate_model_metrics(y_true, y_pred, y_pred_proba)

        assert "accuracy" in metrics
        assert "auc" in metrics
        assert metrics["accuracy"] == 5 / 6  # 5 correct out of 6
        assert 0 <= metrics["auc"] <= 1

    def test_create_learning_curve_data(self):
        """Test learning curve data creation"""
        train_scores = [0.8, 0.85, 0.87, 0.88]
        val_scores = [0.75, 0.78, 0.79, 0.79]

        curve_data = create_learning_curve_data(train_scores, val_scores)

        assert curve_data["train_scores"] == train_scores
        assert curve_data["val_scores"] == val_scores
        assert curve_data["epochs"] == [0, 1, 2, 3]

    def test_save_and_load_model_with_metadata(self):
        """Test model save/load with metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train model
            model = LightGBMModel()
            X = np.random.random((100, 5))
            y = np.random.randint(0, 2, 100)
            feature_names = [f"feature_{i}" for i in range(5)]
            model.fit(X, y, feature_names=feature_names)

            # Create CV results
            cv_results = {
                "mean_score": 0.95,
                "std_score": 0.02,
                "fold_scores": [0.94, 0.95, 0.96],
            }

            # Additional metadata
            metadata = {
                "experiment_name": "test_experiment",
                "dataset_version": "v1.0",
            }

            # Save model
            filepath = os.path.join(tmpdir, "model_with_meta.pkl")
            save_model_with_metadata(model, cv_results, filepath, metadata)

            assert os.path.exists(filepath)

            # Load model
            loaded_model, loaded_metadata = load_model_with_metadata(filepath)

            assert loaded_model.is_fitted
            assert loaded_model.feature_names == feature_names
            assert loaded_metadata["cv_score"] == 0.95
            assert loaded_metadata["experiment_name"] == "test_experiment"

    def test_load_model_file_not_found(self):
        """Test load model with non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_model_with_metadata("non_existent_file.pkl")


class TestIntegrationScenarios:
    """Integration tests for complete workflows"""

    def test_full_training_pipeline(self):
        """Test complete training pipeline"""
        # Create sample dataset
        n_samples = 200
        n_features = 10
        X = np.random.random((n_samples, n_features))
        y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple rule for reproducibility
        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Initialize trainer
        trainer = CrossValidationTrainer()

        # Train with cross-validation
        cv_results = trainer.train_cv(model_class=LightGBMModel, X=X, y=y, feature_names=feature_names)

        # Verify results
        assert cv_results["mean_score"] > 0.5  # Should be better than random
        assert len(cv_results["models"]) == 5
        assert cv_results["oof_predictions"].shape == (n_samples,)

        # Get best model and make predictions
        best_model = cv_results["models"][0]
        X_test = np.random.random((50, n_features))
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)

        assert len(predictions) == 50
        assert probabilities.shape == (50, 2)

        # Get feature importance
        importance = best_model.get_feature_importance()
        assert len(importance) == n_features

    def test_model_persistence_workflow(self):
        """Test complete model save/load workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train model
            X = np.random.random((100, 5))
            y = np.random.randint(0, 2, 100)

            trainer = CrossValidationTrainer()
            cv_results = trainer.train_cv(LightGBMModel, X, y)

            # Save with metadata
            filepath = os.path.join(tmpdir, "production_model.pkl")
            metadata = {"production_version": "1.0.0"}
            save_model_with_metadata(cv_results["models"][0], cv_results, filepath, metadata)

            # Load and use
            loaded_model, loaded_meta = load_model_with_metadata(filepath)
            X_new = np.random.random((10, 5))
            predictions = loaded_model.predict(X_new)

            assert len(predictions) == 10
            assert loaded_meta["production_version"] == "1.0.0"


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling"""

    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        model = LightGBMModel()
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])

        with pytest.raises(Exception):  # LightGBM will raise an error
            model.fit(X_empty, y_empty)

    def test_single_class_data(self):
        """Test handling of single-class data"""
        model = LightGBMModel()
        X = np.random.random((100, 5))
        y = np.zeros(100)  # All same class

        # This should raise an error or warning
        with pytest.raises(Exception):
            model.fit(X, y)

    def test_mismatched_dimensions(self):
        """Test handling of mismatched X and y dimensions"""
        model = LightGBMModel()
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 80)  # Wrong size

        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_invalid_feature_names(self):
        """Test handling of invalid feature names"""
        model = LightGBMModel()
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        feature_names = ["f1", "f2", "f3"]  # Wrong number

        with pytest.raises(ValueError):
            model.fit(X, y, feature_names=feature_names)

    def test_cv_with_too_few_samples(self):
        """Test CV with insufficient samples"""
        trainer = CrossValidationTrainer()
        X = np.random.random((4, 5))  # Only 4 samples
        y = np.array([0, 1, 0, 1])

        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError):
            trainer.train_cv(LightGBMModel, X, y)
