"""
Comprehensive test cases for data modules to achieve 95% coverage
"""

import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data import create_all_tables, quick_start
from src.data.bronze import basic_features, create_bronze_tables, load_bronze_data, load_data, quick_preprocess
from src.data.gold import (
    create_gold_tables,
    create_submission,
    encode_target,
    get_feature_names,
    get_ml_ready_data,
    load_gold_data,
    prepare_model_data,
)
from src.data.silver import advanced_features, create_silver_tables, load_silver_data, scaling_features


class TestDataInit:
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
        mock_data = (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [3, 4]}))
        mock_load.return_value = mock_data

        train, test = quick_start("bronze")
        assert train.equals(mock_data[0])
        assert test.equals(mock_data[1])

    @patch("src.data.silver.load_silver_data")
    def test_quick_start_silver(self, mock_load):
        """Test quick start with silver level"""
        mock_data = (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [3, 4]}))
        mock_load.return_value = mock_data

        train, test = quick_start("silver")
        assert train.equals(mock_data[0])
        assert test.equals(mock_data[1])

    @patch("src.data.gold.load_gold_data")
    def test_quick_start_gold(self, mock_load):
        """Test quick start with gold level"""
        mock_data = (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [3, 4]}))
        mock_load.return_value = mock_data

        train, test = quick_start("gold")
        assert train.equals(mock_data[0])
        assert test.equals(mock_data[1])

    def test_quick_start_invalid_level(self):
        """Test quick start with invalid level"""
        with pytest.raises(ValueError, match="Unsupported level"):
            quick_start("platinum")


class TestBronzeDataFull:
    """Comprehensive tests for bronze data layer"""

    @patch("duckdb.connect")
    def test_load_data(self, mock_connect):
        """Test raw data loading"""
        # Mock database connection and queries
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Mock train data
        train_data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "feature1": [0.1, 0.2, 0.3],
                "Personality": ["Introvert", "Extrovert", "Introvert"],
            }
        )

        # Mock test data
        test_data = pd.DataFrame(
            {
                "id": [4, 5, 6],
                "feature1": [0.4, 0.5, 0.6],
            }
        )

        mock_conn.execute.side_effect = [
            Mock(df=Mock(return_value=train_data)),
            Mock(df=Mock(return_value=test_data)),
        ]

        train, test = load_data()

        assert len(train) == 3
        assert len(test) == 3
        assert "Personality" in train.columns
        assert "Personality" not in test.columns

    @patch("duckdb.connect")
    def test_create_bronze_tables(self, mock_connect):
        """Test bronze table creation"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Mock source data
        train_data = pd.DataFrame(
            {
                "id": [1, 2],
                "Time_spent_Alone": [1, 2],
                "Social_event_attendance": [3, 4],
                "Going_outside": [1, 0],
                "Friends_circle_size": [10, 20],
                "Post_frequency": [5, 15],
                "Stage_fear": ["Yes", "No"],
                "Drained_after_socializing": ["Yes", "No"],
                "Personality": ["Introvert", "Extrovert"],
            }
        )

        test_data = pd.DataFrame(
            {
                "id": [3, 4],
                "Time_spent_Alone": [2, 3],
                "Social_event_attendance": [2, 5],
                "Going_outside": [0, 1],
                "Friends_circle_size": [15, 25],
                "Post_frequency": [10, 20],
                "Stage_fear": ["No", "Yes"],
                "Drained_after_socializing": ["No", "Yes"],
            }
        )

        mock_conn.execute.side_effect = [
            Mock(df=Mock(return_value=train_data)),
            Mock(df=Mock(return_value=test_data)),
            Mock(),  # CREATE SCHEMA
            Mock(),  # CREATE TABLE train
            Mock(),  # INSERT train
            Mock(),  # CREATE TABLE test
            Mock(),  # INSERT test
        ]

        create_bronze_tables()

        # Verify schema and table creation
        assert mock_conn.execute.call_count >= 5

    @patch("duckdb.connect")
    def test_load_bronze_data(self, mock_connect):
        """Test loading bronze data"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        train_data = pd.DataFrame({"id": [1, 2], "feature": [0.1, 0.2]})
        test_data = pd.DataFrame({"id": [3, 4], "feature": [0.3, 0.4]})

        mock_conn.execute.side_effect = [
            Mock(df=Mock(return_value=train_data)),
            Mock(df=Mock(return_value=test_data)),
        ]

        train, test = load_bronze_data()

        assert train.equals(train_data)
        assert test.equals(test_data)

    def test_quick_preprocess(self):
        """Test quick preprocessing"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "Time_spent_Alone": [1, 2, np.nan],
                "Social_event_attendance": [3, np.nan, 5],
                "Going_outside": [1, 0, 1],
                "Stage_fear": ["Yes", "No", "Yes"],
                "Drained_after_socializing": ["Yes", "No", "Yes"],
            }
        )

        processed = quick_preprocess(df)

        # Check NaN filling
        assert not processed["Time_spent_Alone"].isna().any()
        assert not processed["Social_event_attendance"].isna().any()

        # Check encoding
        assert processed["Stage_fear_Yes"].tolist() == [1, 0, 1]
        assert processed["Drained_after_socializing_Yes"].tolist() == [1, 0, 1]

    def test_quick_preprocess_missing_columns(self):
        """Test preprocessing with missing columns"""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "Time_spent_Alone": [1, 2],
                # Missing other columns
            }
        )

        processed = quick_preprocess(df)

        # Should handle gracefully
        assert "Time_spent_Alone" in processed.columns

    def test_basic_features(self):
        """Test basic feature engineering"""
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

        # Check new features
        assert "social_score" in features.columns
        assert "isolation_score" in features.columns
        assert "online_offline_ratio" in features.columns

        # Verify calculations
        expected_social = (10 + 2 + 50) + (5 + 1 + 30) + (1 + 0 + 10)
        assert features["social_score"].sum() == expected_social

    def test_basic_features_with_missing_columns(self):
        """Test basic features with missing columns"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1, 2, 3],
                # Missing other columns
            }
        )

        # Should return original dataframe
        features = basic_features(df)
        assert features.equals(df)


class TestSilverDataFull:
    """Comprehensive tests for silver data layer"""

    @patch("duckdb.connect")
    def test_create_silver_tables(self, mock_connect):
        """Test silver table creation"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Mock bronze data with all required columns
        bronze_train = pd.DataFrame(
            {
                "id": [1, 2],
                "Time_spent_Alone": [1, 5],
                "Social_event_attendance": [10, 2],
                "Going_outside": [2, 0],
                "Friends_circle_size": [50, 10],
                "Post_frequency": [20, 5],
                "Stage_fear": ["Yes", "No"],
                "Drained_after_socializing": ["Yes", "No"],
                "Personality": ["Extrovert", "Introvert"],
            }
        )

        bronze_test = pd.DataFrame(
            {
                "id": [3, 4],
                "Time_spent_Alone": [3, 7],
                "Social_event_attendance": [5, 1],
                "Going_outside": [1, 0],
                "Friends_circle_size": [25, 5],
                "Post_frequency": [10, 2],
                "Stage_fear": ["No", "Yes"],
                "Drained_after_socializing": ["No", "Yes"],
            }
        )

        mock_conn.execute.side_effect = [
            Mock(df=Mock(return_value=bronze_train)),
            Mock(df=Mock(return_value=bronze_test)),
            Mock(),  # CREATE SCHEMA
            Mock(),  # CREATE TABLE train
            Mock(),  # INSERT train
            Mock(),  # CREATE TABLE test
            Mock(),  # INSERT test
        ]

        create_silver_tables()

        assert mock_conn.execute.call_count >= 5

    @patch("duckdb.connect")
    def test_load_silver_data(self, mock_connect):
        """Test loading silver data"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        train_data = pd.DataFrame({"id": [1, 2], "feature": [0.1, 0.2]})
        test_data = pd.DataFrame({"id": [3, 4], "feature": [0.3, 0.4]})

        mock_conn.execute.side_effect = [
            Mock(df=Mock(return_value=train_data)),
            Mock(df=Mock(return_value=test_data)),
        ]

        train, test = load_silver_data()

        assert train.equals(train_data)
        assert test.equals(test_data)

    def test_advanced_features(self):
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

        # Check ratios
        assert "alone_to_social_ratio" in features.columns
        assert "post_per_friend" in features.columns

    def test_advanced_features_with_zeros(self):
        """Test advanced features with zero values"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1, 0, 10],
                "Social_event_attendance": [0, 5, 1],
                "Going_outside": [2, 1, 0],
                "Friends_circle_size": [0, 30, 10],
                "Post_frequency": [20, 10, 0],
            }
        )

        features = advanced_features(df)

        # Should handle division by zero
        assert not features["alone_to_social_ratio"].isna().any()
        assert not features["post_per_friend"].isna().any()

    def test_scaling_features(self):
        """Test feature scaling"""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "id": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        scaled = scaling_features(df)

        # Check that non-numeric columns are preserved
        assert "id" in scaled.columns
        assert "target" in scaled.columns

        # Check that numeric features are scaled
        assert scaled["feature1"].mean() == pytest.approx(0, abs=1e-10)
        assert scaled["feature2"].mean() == pytest.approx(0, abs=1e-10)
        assert scaled["feature1"].std() == pytest.approx(1, rel=1e-2)


class TestGoldDataFull:
    """Comprehensive tests for gold data layer"""

    @patch("duckdb.connect")
    def test_create_gold_tables(self, mock_connect):
        """Test gold table creation"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Mock silver data
        silver_train = pd.DataFrame(
            {
                "id": [1, 2],
                "Time_spent_Alone": [1, 5],
                "Personality": ["Introvert", "Extrovert"],
            }
        )

        silver_test = pd.DataFrame(
            {
                "id": [3, 4],
                "Time_spent_Alone": [3, 7],
            }
        )

        mock_conn.execute.side_effect = [
            Mock(df=Mock(return_value=silver_train)),
            Mock(df=Mock(return_value=silver_test)),
            Mock(),  # CREATE SCHEMA
            Mock(),  # CREATE TABLE train
            Mock(),  # INSERT train
            Mock(),  # CREATE TABLE test
            Mock(),  # INSERT test
        ]

        create_gold_tables()

        assert mock_conn.execute.call_count >= 5

    @patch("duckdb.connect")
    def test_load_gold_data(self, mock_connect):
        """Test loading gold data"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        train_data = pd.DataFrame(
            {
                "id": [1, 2],
                "feature": [0.1, 0.2],
                "Personality_encoded": [0, 1],
            }
        )
        test_data = pd.DataFrame(
            {
                "id": [3, 4],
                "feature": [0.3, 0.4],
            }
        )

        mock_conn.execute.side_effect = [
            Mock(df=Mock(return_value=train_data)),
            Mock(df=Mock(return_value=test_data)),
        ]

        train, test = load_gold_data()

        assert train.equals(train_data)
        assert test.equals(test_data)

    def test_prepare_model_data(self):
        """Test model data preparation"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [1, 2, 3],
                "Personality": ["Introvert", "Extrovert", "Introvert"],
            }
        )

        X, y, features = prepare_model_data(df)

        assert X.shape == (3, 2)
        assert len(y) == 3
        assert features == ["feature1", "feature2"]
        assert y.tolist() == [0, 1, 0]

    def test_prepare_model_data_no_target(self):
        """Test model data preparation without target"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [1, 2, 3],
            }
        )

        X, y, features = prepare_model_data(df, target_col=None)

        assert X.shape == (3, 2)
        assert y is None
        assert features == ["feature1", "feature2"]

    def test_encode_target(self):
        """Test target encoding"""
        df = pd.DataFrame(
            {
                "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert"],
            }
        )

        encoded = encode_target(df)

        assert "Personality_encoded" in encoded.columns
        assert encoded["Personality_encoded"].tolist() == [0, 1, 0, 1]

    def test_encode_target_custom_column(self):
        """Test target encoding with custom column"""
        df = pd.DataFrame(
            {
                "target": ["A", "B", "A", "B"],
            }
        )

        encoded = encode_target(df, target_col="target")

        assert "target_encoded" in encoded.columns

    def test_get_feature_names(self):
        """Test feature name extraction"""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "feature1": [0.1, 0.2],
                "feature2": [1, 2],
                "Personality": ["A", "B"],
                "Personality_encoded": [0, 1],
            }
        )

        features = get_feature_names(df)

        assert features == ["feature1", "feature2"]

    def test_get_ml_ready_data(self):
        """Test ML-ready data preparation"""
        train = pd.DataFrame(
            {
                "id": [1, 2],
                "feature1": [0.1, 0.2],
                "Personality": ["Introvert", "Extrovert"],
            }
        )
        test = pd.DataFrame(
            {
                "id": [3, 4],
                "feature1": [0.3, 0.4],
            }
        )

        X_train, X_test, y_train, feature_names = get_ml_ready_data(train, test)

        assert X_train.shape == (2, 1)
        assert X_test.shape == (2, 1)
        assert len(y_train) == 2
        assert feature_names == ["feature1"]

    def test_create_submission(self):
        """Test submission creation"""
        test_ids = np.array([1, 2, 3, 4, 5])
        predictions = np.array([0, 1, 1, 0, 1])

        submission = create_submission(test_ids, predictions)

        assert len(submission) == 5
        assert list(submission.columns) == ["id", "Personality"]
        assert submission["Personality"].tolist() == ["Introvert", "Extrovert", "Extrovert", "Introvert", "Extrovert"]

    def test_create_submission_with_filename(self):
        """Test submission creation with file save"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_ids = np.array([1, 2, 3])
            predictions = np.array([0, 1, 0])

            filename = f"{tmpdir}/submission.csv"
            create_submission(test_ids, predictions, filename)

            # Check file was created
            saved_df = pd.read_csv(filename)
            assert len(saved_df) == 3
            assert saved_df["Personality"].tolist() == ["Introvert", "Extrovert", "Introvert"]
