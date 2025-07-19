"""
Test cases for actual data module functions to improve coverage
"""

import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import create_all_tables, quick_start
from src.data.bronze import basic_features, quick_preprocess
from src.data.gold import create_submission, encode_target, get_feature_names, get_ml_ready_data, prepare_model_data
from src.data.silver import advanced_features, scaling_features


class TestDataPackageInit:
    """Test data package initialization functions"""

    @patch("src.data.bronze.create_bronze_tables")
    @patch("src.data.silver.create_silver_tables")
    @patch("src.data.gold.create_gold_tables")
    def test_create_all_tables(self, mock_gold, mock_silver, mock_bronze):
        """Test creating all medallion tables"""
        create_all_tables()

        mock_bronze.assert_called_once()
        mock_silver.assert_called_once()
        mock_gold.assert_called_once()

    @patch("src.data.bronze.load_bronze_data")
    def test_quick_start_bronze(self, mock_load):
        """Test quick start with bronze level"""
        mock_data = (
            pd.DataFrame({"id": [1, 2], "feature": [0.1, 0.2]}),
            pd.DataFrame({"id": [3, 4], "feature": [0.3, 0.4]}),
        )
        mock_load.return_value = mock_data

        train, test = quick_start("bronze")
        assert train.equals(mock_data[0])
        assert test.equals(mock_data[1])
        mock_load.assert_called_once()

    @patch("src.data.silver.load_silver_data")
    def test_quick_start_silver(self, mock_load):
        """Test quick start with silver level"""
        mock_data = (
            pd.DataFrame({"id": [1, 2], "feature": [0.1, 0.2]}),
            pd.DataFrame({"id": [3, 4], "feature": [0.3, 0.4]}),
        )
        mock_load.return_value = mock_data

        train, test = quick_start("silver")
        assert train.equals(mock_data[0])
        assert test.equals(mock_data[1])

    @patch("src.data.gold.load_gold_data")
    def test_quick_start_gold(self, mock_load):
        """Test quick start with gold level"""
        mock_data = (
            pd.DataFrame({"id": [1, 2], "feature": [0.1, 0.2]}),
            pd.DataFrame({"id": [3, 4], "feature": [0.3, 0.4]}),
        )
        mock_load.return_value = mock_data

        train, test = quick_start("gold")
        assert train.equals(mock_data[0])
        assert test.equals(mock_data[1])

    def test_quick_start_invalid_level(self):
        """Test quick start with invalid level"""
        with pytest.raises(ValueError, match="Unsupported level"):
            quick_start("platinum")


class TestBronzeDataLayer:
    """Test bronze data layer functions"""

    def test_quick_preprocess_basic(self):
        """Test basic preprocessing functionality"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "Time_spent_Alone": [1, 2, np.nan, 4],
                "Social_event_attendance": [10, np.nan, 30, 40],
                "Going_outside": [1, 0, 1, 0],
                "Friends_circle_size": [50, 30, 10, 20],
                "Post_frequency": [20, 10, np.nan, 15],
                "Stage_fear": ["Yes", "No", "Yes", "No"],
                "Drained_after_socializing": ["Yes", "No", "Yes", "No"],
            }
        )

        processed = quick_preprocess(df)

        # Check that NaN values are filled
        assert not processed["Time_spent_Alone"].isna().any()
        assert not processed["Social_event_attendance"].isna().any()
        assert not processed["Post_frequency"].isna().any()

        # Check that binary encoding was applied
        assert "Stage_fear_Yes" in processed.columns
        assert "Drained_after_socializing_Yes" in processed.columns

        # Check encoding values
        assert processed["Stage_fear_Yes"].tolist() == [1, 0, 1, 0]
        assert processed["Drained_after_socializing_Yes"].tolist() == [1, 0, 1, 0]

    def test_quick_preprocess_missing_columns(self):
        """Test preprocessing with missing expected columns"""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "Time_spent_Alone": [1, 2],
                "unknown_column": [10, 20],
            }
        )

        # Should not crash and return processed dataframe
        processed = quick_preprocess(df)
        assert "Time_spent_Alone" in processed.columns
        assert len(processed) == 2

    def test_basic_features_complete(self):
        """Test basic feature engineering with all columns"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1, 5, 10],
                "Social_event_attendance": [10, 5, 1],
                "Going_outside": [2, 1, 0],
                "Friends_circle_size": [50, 30, 10],
                "Post_frequency": [20, 10, 5],
            }
        )

        features = basic_features(df)

        # Check that new features were created
        assert "social_score" in features.columns
        assert "isolation_score" in features.columns
        assert "online_offline_ratio" in features.columns

        # Check calculations make sense
        expected_social_0 = 10 + 2 + 50  # First row
        assert features.loc[0, "social_score"] == expected_social_0

        expected_isolation_0 = 1 + 20  # First row
        assert features.loc[0, "isolation_score"] == expected_isolation_0

    def test_basic_features_with_missing_columns(self):
        """Test basic features with incomplete data"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1, 2, 3],
                "Social_event_attendance": [10, 20, 30],
                # Missing other expected columns
            }
        )

        # Should return original dataframe when required columns missing
        features = basic_features(df)
        assert features.equals(df)


class TestSilverDataLayer:
    """Test silver data layer functions"""

    def test_advanced_features_complete(self):
        """Test advanced feature engineering"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1, 5, 10],
                "Social_event_attendance": [10, 5, 1],
                "Going_outside": [2, 1, 0],
                "Friends_circle_size": [50, 30, 10],
                "Post_frequency": [20, 10, 5],
            }
        )

        features = advanced_features(df)

        # Check interaction features
        assert "alone_x_social" in features.columns
        assert "friends_x_posts" in features.columns

        # Check ratio features
        assert "alone_to_social_ratio" in features.columns
        assert "post_per_friend" in features.columns

        # Verify calculations
        assert features.loc[0, "alone_x_social"] == 1 * 10
        assert features.loc[0, "friends_x_posts"] == 50 * 20

    def test_advanced_features_with_zeros(self):
        """Test advanced features with zero values (division by zero cases)"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [0, 5, 10],
                "Social_event_attendance": [10, 0, 1],
                "Going_outside": [2, 1, 0],
                "Friends_circle_size": [0, 30, 10],
                "Post_frequency": [20, 10, 0],
            }
        )

        features = advanced_features(df)

        # Should handle division by zero gracefully
        assert not features["alone_to_social_ratio"].isna().any()
        assert not features["post_per_friend"].isna().any()

    def test_scaling_features(self):
        """Test feature scaling functionality"""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "id": [1, 2, 3, 4, 5],  # Should be preserved
                "Personality": ["A", "B", "A", "B", "A"],  # Should be preserved
            }
        )

        scaled = scaling_features(df)

        # Check that non-numeric columns are preserved
        assert "id" in scaled.columns
        assert "Personality" in scaled.columns
        assert scaled["id"].tolist() == [1, 2, 3, 4, 5]

        # Check that numeric features are scaled (mean=0, std=1)
        assert abs(scaled["feature1"].mean()) < 1e-10
        assert abs(scaled["feature2"].mean()) < 1e-10
        assert abs(scaled["feature1"].std() - 1.0) < 1e-2
        assert abs(scaled["feature2"].std() - 1.0) < 1e-2


class TestGoldDataLayer:
    """Test gold data layer functions"""

    def test_prepare_model_data_with_target(self):
        """Test model data preparation with target column"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "feature1": [0.1, 0.2, 0.3, 0.4],
                "feature2": [10, 20, 30, 40],
                "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert"],
            }
        )

        X, y, feature_names = prepare_model_data(df)

        assert X.shape == (4, 2)
        assert len(y) == 4
        assert feature_names == ["feature1", "feature2"]
        assert y.tolist() == [0, 1, 0, 1]  # Encoded values

    def test_prepare_model_data_without_target(self):
        """Test model data preparation without target"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [10, 20, 30],
            }
        )

        X, y, feature_names = prepare_model_data(df, target_col=None)

        assert X.shape == (3, 2)
        assert y is None
        assert feature_names == ["feature1", "feature2"]

    def test_encode_target_default(self):
        """Test default target encoding"""
        df = pd.DataFrame(
            {
                "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert"],
                "other_col": [1, 2, 3, 4],
            }
        )

        encoded = encode_target(df)

        assert "Personality_encoded" in encoded.columns
        assert encoded["Personality_encoded"].tolist() == [0, 1, 0, 1]
        assert encoded["other_col"].tolist() == [1, 2, 3, 4]  # Preserved

    def test_encode_target_custom_column(self):
        """Test target encoding with custom column name"""
        df = pd.DataFrame(
            {
                "target": ["A", "B", "A", "B", "C"],
            }
        )

        encoded = encode_target(df, target_col="target")

        assert "target_encoded" in encoded.columns
        # LabelEncoder will assign 0, 1, 2 to A, B, C
        assert len(set(encoded["target_encoded"])) == 3

    def test_get_feature_names(self):
        """Test feature name extraction"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [10, 20, 30],
                "Personality": ["A", "B", "C"],
                "Personality_encoded": [0, 1, 2],
            }
        )

        feature_names = get_feature_names(df)

        # Should exclude id and target columns
        assert feature_names == ["feature1", "feature2"]

    def test_get_ml_ready_data(self):
        """Test ML-ready data preparation"""
        train_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [10, 20, 30],
                "Personality": ["Introvert", "Extrovert", "Introvert"],
            }
        )

        test_df = pd.DataFrame(
            {
                "id": [4, 5],
                "feature1": [0.4, 0.5],
                "feature2": [40, 50],
            }
        )

        X_train, X_test, y_train, feature_names = get_ml_ready_data(train_df, test_df)

        assert X_train.shape == (3, 2)
        assert X_test.shape == (2, 2)
        assert len(y_train) == 3
        assert feature_names == ["feature1", "feature2"]
        assert y_train.tolist() == [0, 1, 0]

    def test_create_submission_basic(self):
        """Test submission file creation"""
        test_ids = np.array([100, 101, 102, 103])
        predictions = np.array([0, 1, 1, 0])

        submission = create_submission(test_ids, predictions)

        assert len(submission) == 4
        assert list(submission.columns) == ["id", "Personality"]
        assert submission["id"].tolist() == [100, 101, 102, 103]
        assert submission["Personality"].tolist() == ["Introvert", "Extrovert", "Extrovert", "Introvert"]

    def test_create_submission_with_save(self):
        """Test submission creation with file save"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            test_ids = np.array([1, 2, 3])
            predictions = np.array([1, 0, 1])
            filename = os.path.join(tmpdir, "test_submission.csv")

            submission = create_submission(test_ids, predictions, filename)

            # Check return value
            assert len(submission) == 3
            assert submission["Personality"].tolist() == ["Extrovert", "Introvert", "Extrovert"]

            # Check file was created and saved correctly
            assert os.path.exists(filename)
            saved_df = pd.read_csv(filename)
            assert len(saved_df) == 3
            assert saved_df["Personality"].tolist() == ["Extrovert", "Introvert", "Extrovert"]
