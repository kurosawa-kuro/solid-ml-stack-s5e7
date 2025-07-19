"""
Focused tests to improve specific coverage gaps identified in coverage report
"""

import os
import tempfile

import numpy as np
import pandas as pd

from src.data.bronze import basic_features, quick_preprocess
from src.data.gold import encode_target, prepare_model_data
from src.data.silver import advanced_features, scaling_features

# Import exactly what exists
from src.models import (
    CrossValidationTrainer,
    LightGBMModel,
    create_learning_curve_data,
    evaluate_model_metrics,
    load_model_with_metadata,
    save_model_with_metadata,
)
from src.validation import CVLogger, CVStrategy


class TestModelsFocused:
    """Test uncovered parts of models.py"""

    def test_lightgbm_model_fit_method(self):
        """Test the actual fit method"""
        model = LightGBMModel()
        X = np.random.random((50, 3))
        y = np.random.randint(0, 2, 50)
        feature_names = ["f1", "f2", "f3"]

        # Test basic fit
        model.fit(X, y, feature_names=feature_names)
        assert model.is_fitted
        assert model.feature_names == feature_names
        assert model.model is not None

    def test_lightgbm_model_predict_methods(self):
        """Test predict and predict_proba methods"""
        model = LightGBMModel()
        X_train = np.random.random((50, 3))
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)

        X_test = np.random.random((10, 3))

        # Test predict
        predictions = model.predict(X_test)
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

        # Test predict_proba
        probas = model.predict_proba(X_test)
        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_lightgbm_model_feature_importance(self):
        """Test get_feature_importance method"""
        model = LightGBMModel()
        X = np.random.random((50, 3))
        y = np.random.randint(0, 2, 50)
        feature_names = ["feature_a", "feature_b", "feature_c"]

        model.fit(X, y, feature_names=feature_names)
        importance_df = model.get_feature_importance()

        assert len(importance_df) == 3
        assert list(importance_df.columns) == ["feature", "importance"]
        assert set(importance_df["feature"]) == set(feature_names)

    def test_lightgbm_model_save_load(self):
        """Test save and load methods"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LightGBMModel()
            X = np.random.random((50, 3))
            y = np.random.randint(0, 2, 50)
            feature_names = ["f1", "f2", "f3"]

            model.fit(X, y, feature_names=feature_names)

            # Test save
            filepath = os.path.join(tmpdir, "test_model.pkl")
            model.save(filepath)
            assert os.path.exists(filepath)

            # Test load
            loaded_model = LightGBMModel.load(filepath)
            assert loaded_model.is_fitted
            assert loaded_model.feature_names == feature_names

    def test_cv_trainer_train_cv(self):
        """Test CrossValidationTrainer.train_cv method"""
        trainer = CrossValidationTrainer()
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]

        results = trainer.train_cv(model_class=LightGBMModel, X=X, y=y, feature_names=feature_names)

        # Check all expected keys
        expected_keys = [
            "fold_scores",
            "mean_score",
            "std_score",
            "fold_auc_scores",
            "mean_auc",
            "std_auc",
            "oof_predictions",
            "oof_binary_predictions",
            "prediction_distribution",
            "feature_importance",
            "training_time",
            "models",
            "cv_config",
            "data_integrity_checks",
        ]
        for key in expected_keys:
            assert key in results

        assert len(results["fold_scores"]) == 5  # default 5 folds
        assert len(results["models"]) == 5
        assert len(results["oof_predictions"]) == 100

    def test_evaluate_model_metrics(self):
        """Test evaluate_model_metrics function"""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.3])

        metrics = evaluate_model_metrics(y_true, y_pred, y_pred_proba)

        assert "accuracy" in metrics
        assert "auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc"] <= 1

    def test_create_learning_curve_data(self):
        """Test create_learning_curve_data function"""
        train_scores = [0.8, 0.85, 0.87, 0.88]
        val_scores = [0.75, 0.78, 0.79, 0.80]

        curve_data = create_learning_curve_data(train_scores, val_scores)

        assert curve_data["train_scores"] == train_scores
        assert curve_data["val_scores"] == val_scores
        assert curve_data["epochs"] == [0, 1, 2, 3]

    def test_save_load_model_with_metadata(self):
        """Test save_model_with_metadata and load_model_with_metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train model
            model = LightGBMModel()
            X = np.random.random((50, 3))
            y = np.random.randint(0, 2, 50)
            feature_names = ["f1", "f2", "f3"]
            model.fit(X, y, feature_names=feature_names)

            # Test save with metadata
            filepath = os.path.join(tmpdir, "model_with_meta.pkl")
            metadata = {"experiment": "test", "version": 1}
            test_cv_results = {
                "mean_score": 0.95,
                "std_score": 0.02,
                "fold_scores": [0.94, 0.95, 0.96],
            }
            save_model_with_metadata(model, test_cv_results, filepath, metadata)

            assert os.path.exists(filepath)

            # Test load
            loaded_model, loaded_metadata = load_model_with_metadata(filepath)

            assert loaded_model.is_fitted
            assert loaded_model.feature_names == feature_names
            assert loaded_metadata["cv_score"] == 0.95
            assert loaded_metadata["experiment"] == "test"


class TestValidationFocused:
    """Test uncovered parts of validation.py"""

    def test_cv_strategy_methods(self):
        """Test CVStrategy methods"""
        strategy = CVStrategy(n_splits=3, shuffle=True, random_state=42)

        # Test split method
        X = np.random.random((30, 5))
        y = np.array([0, 1] * 15)
        splits = strategy.split(X, y)

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) + len(test_idx) == 30

        # Test basic attributes
        assert strategy.n_splits == 3
        assert strategy.shuffle is True
        assert strategy.random_state == 42

    def test_cv_logger_methods(self):
        """Test CVLogger methods"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CVLogger(log_dir=tmpdir)

            # Test create_log_entry
            entry = logger.create_log_entry(
                model_type="LightGBM",
                cv_config={"n_splits": 5},
                fold_scores=[0.93, 0.95, 0.97],
                training_time=120.5,
                notes="test",
            )

            assert entry["model_type"] == "LightGBM"
            assert "mean_score" in entry
            assert "fold_scores" in entry
            assert entry["training_time"] == 120.5

            # Test save_json_log
            filepath = logger.save_json_log(entry, "test.json")
            assert os.path.exists(filepath)

            # Test save_csv_log
            df = pd.DataFrame({"fold": [0, 1, 2], "score": [0.93, 0.95, 0.97]})
            csv_path = logger.save_csv_log(df, "results.csv")
            assert os.path.exists(csv_path)


class TestDataFocused:
    """Test uncovered parts of data modules"""

    def test_bronze_quick_preprocess(self):
        """Test quick_preprocess function"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1, 2, np.nan],
                "Social_event_attendance": [10, np.nan, 30],
                "Stage_fear": ["Yes", "No", "Yes"],
                "Drained_after_socializing": ["No", "Yes", "No"],
            }
        )

        processed = quick_preprocess(df)

        # Check NaN handling
        assert processed["Time_spent_Alone"].isna().any() is False
        assert processed["Social_event_attendance"].isna().any() is False

        # Check encoding
        assert "Stage_fear_encoded" in processed.columns
        assert "Drained_after_socializing_encoded" in processed.columns

    def test_bronze_basic_features(self):
        """Test basic_features function"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1, 5],
                "Social_event_attendance": [10, 5],
                "Going_outside": [2, 1],
                "Friends_circle_size": [50, 30],
                "Post_frequency": [20, 10],
            }
        )

        features = basic_features(df)

        assert "social_ratio" in features.columns
        assert "activity_sum" in features.columns

    def test_silver_advanced_features(self):
        """Test advanced_features function"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1, 5],
                "Social_event_attendance": [10, 5],
                "Going_outside": [2, 1],
                "Friends_circle_size": [50, 30],
                "Post_frequency": [20, 10],
            }
        )

        features = advanced_features(df)

        assert "social_ratio" in features.columns
        assert "activity_sum" in features.columns
        assert "total_activity" in features.columns
        assert "post_per_friend" in features.columns
        assert "extrovert_score" in features.columns

    def test_silver_scaling_features(self):
        """Test scaling_features function"""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "id": [1, 2, 3, 4, 5],
            }
        )

        scaled = scaling_features(df)

        # Check scaling created new scaled columns
        assert "feature1_scaled" in scaled.columns
        assert "feature2_scaled" in scaled.columns
        assert abs(scaled["feature1_scaled"].mean()) < 1e-10
        assert abs(scaled["feature2_scaled"].mean()) < 1e-10
        # Check id preserved
        assert "id" in scaled.columns

    def test_gold_functions(self):
        """Test gold layer functions"""
        # Test prepare_model_data - returns filtered DataFrame
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "Social_event_attendance": [0.1, 0.2, 0.3],
                "Time_spent_Alone": [10, 20, 30],
                "Personality": ["Introvert", "Extrovert", "Introvert"],
            }
        )

        model_df = prepare_model_data(df, target_col="Personality")
        assert "id" in model_df.columns
        assert "Social_event_attendance" in model_df.columns
        assert "Time_spent_Alone" in model_df.columns
        assert "Personality" in model_df.columns

        # Test encode_target
        encoded = encode_target(df)
        assert "Personality_encoded" in encoded.columns
        assert encoded["Personality_encoded"].tolist() == [0, 1, 0]
