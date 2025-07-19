"""
Test for Bronze Level Data Management
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.bronze import (
    basic_features, 
    load_data, 
    quick_preprocess,
    validate_data_quality,
    advanced_missing_strategy,
    encode_categorical_robust,
    winsorize_outliers,
    create_bronze_tables
)


class TestBronzeData:
    """Bronze data management tests"""

    @patch("src.data.bronze.duckdb.connect")
    def test_load_data(self, mock_connect):
        """Test data loading from DuckDB"""
        # Mock setup
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_train = pd.DataFrame({"id": [1, 2, 3], "Personality": ["Introvert", "Extrovert", "Introvert"]})
        mock_test = pd.DataFrame({"id": [4, 5, 6]})

        mock_conn.execute.side_effect = [MagicMock(df=lambda: mock_train), MagicMock(df=lambda: mock_test)]

        # Execute
        train, test = load_data()

        # Assert
        assert len(train) == 3
        assert len(test) == 3
        assert "Personality" in train.columns
        mock_conn.close.assert_called_once()

    def test_quick_preprocess(self):
        """Test basic preprocessing"""
        # Setup test data
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1.0, None, 3.0],
                "Social_event_attendance": [2.0, 3.0, None],
                "Stage_fear": ["Yes", "No", "Yes"],
                "Drained_after_socializing": ["No", "Yes", "No"],
            }
        )

        # Execute
        result = quick_preprocess(df)

        # Assert
        assert result["Time_spent_Alone"].isna().sum() == 0  # No missing values
        assert result["Social_event_attendance"].isna().sum() == 0
        assert "Stage_fear_encoded" in result.columns
        assert "Drained_after_socializing_encoded" in result.columns
        assert result["Stage_fear_encoded"].tolist() == [1, 0, 1]
        assert result["Drained_after_socializing_encoded"].tolist() == [0, 1, 0]

    def test_basic_features(self):
        """Test basic feature engineering"""
        # Setup test data
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1.0, 2.0, 3.0],
                "Social_event_attendance": [2.0, 4.0, 6.0],
                "Going_outside": [1.0, 2.0, 3.0],
            }
        )

        # Execute
        result = basic_features(df)

        # Assert
        assert "social_ratio" in result.columns
        assert "activity_sum" in result.columns
        assert result["social_ratio"].tolist() == pytest.approx([1.0, 1.33333333, 1.5], rel=1e-6)
        assert result["activity_sum"].tolist() == [3.0, 6.0, 9.0]

    def test_quick_preprocess_missing_columns(self):
        """Test preprocessing with missing columns"""
        df = pd.DataFrame({"other_column": [1, 2, 3]})

        result = quick_preprocess(df)

        # Should not fail and return original data
        assert len(result) == 3
        assert "other_column" in result.columns

    def test_basic_features_missing_columns(self):
        """Test feature engineering with missing columns"""
        df = pd.DataFrame({"other_column": [1, 2, 3]})

        result = basic_features(df)

        # Should not fail and return original data
        assert len(result) == 3
        assert "other_column" in result.columns

    def test_validate_data_quality(self):
        """Test data quality validation"""
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 12.0, 25.0],  # One value > 24hrs
            "Social_event_attendance": [2.0, 3.0, -1.0],  # One negative
            "Stage_fear": ["Yes", "No", "Yes"],
            "Drained_after_socializing": ["No", "Yes", "No"]
        })
        
        result = validate_data_quality(df)
        
        assert "type_validation" in result
        assert "range_validation" in result
        assert result["type_validation"]["Time_spent_Alone"] == True
        assert result["range_validation"]["Time_spent_Alone"]["within_24hrs"] == False
        assert result["range_validation"]["Social_event_attendance"]["non_negative"] == False

    def test_encode_categorical_robust(self):
        """Test robust categorical encoding"""
        df = pd.DataFrame({
            "Stage_fear": ["YES", "no", "Yes", "NO", None],
            "Drained_after_socializing": ["yes", "NO", "Yes", "no", None]
        })
        
        result = encode_categorical_robust(df)
        
        # Check non-null values
        assert result["Stage_fear"].iloc[:4].tolist() == [1.0, 0.0, 1.0, 0.0]
        assert result["Drained_after_socializing"].iloc[:4].tolist() == [1.0, 0.0, 1.0, 0.0]
        
        # Check null handling (NaN != None in pandas)
        assert pd.isna(result["Stage_fear"].iloc[4])
        assert pd.isna(result["Drained_after_socializing"].iloc[4])
        
        assert result["Stage_fear"].dtype == "float64"

    def test_advanced_missing_strategy(self):
        """Test missing value strategy with flags"""
        df = pd.DataFrame({
            "Stage_fear": ["Yes", None, "No"],
            "Going_outside": [1.0, None, 3.0],
            "Time_spent_Alone": [2.0, 4.0, None]
        })
        
        result = advanced_missing_strategy(df)
        
        assert "Stage_fear_missing" in result.columns
        assert "Going_outside_missing" in result.columns
        assert "Time_spent_Alone_missing" in result.columns
        assert result["Stage_fear_missing"].tolist() == [0, 1, 0]
        assert result["Going_outside_missing"].tolist() == [0, 1, 0]
        assert result["Time_spent_Alone_missing"].tolist() == [0, 0, 1]

    def test_winsorize_outliers(self):
        """Test outlier winsorization"""
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 2.0, 3.0, 100.0],  # 100.0 is outlier
            "Social_event_attendance": [1.0, 2.0, 3.0, 4.0]
        })
        
        result = winsorize_outliers(df, percentile=0.25)  # Aggressive clipping for test
        
        # Check that extreme values are clipped
        assert result["Time_spent_Alone"].max() < 100.0
        assert result["Time_spent_Alone"].min() >= 1.0

    @patch("src.data.bronze.duckdb.connect")
    def test_create_bronze_tables(self, mock_connect):
        """Test bronze table creation"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Mock load_data to return test data
        with patch("src.data.bronze.load_data") as mock_load:
            mock_train = pd.DataFrame({
                "Time_spent_Alone": [1.0, 2.0, 3.0],
                "Stage_fear": ["Yes", "No", "Yes"]
            })
            mock_test = pd.DataFrame({
                "Time_spent_Alone": [4.0, 5.0, 6.0]
            })
            mock_load.return_value = (mock_train, mock_test)
            
            create_bronze_tables()
            
            # Verify schema creation
            mock_conn.execute.assert_any_call("CREATE SCHEMA IF NOT EXISTS bronze")
            
            # Verify table drops
            mock_conn.execute.assert_any_call("DROP TABLE IF EXISTS bronze.train")
            mock_conn.execute.assert_any_call("DROP TABLE IF EXISTS bronze.test")
            
            # Verify table creation
            mock_conn.execute.assert_any_call("CREATE TABLE bronze.train AS SELECT * FROM train_bronze_df")
            mock_conn.execute.assert_any_call("CREATE TABLE bronze.test AS SELECT * FROM test_bronze_df")
