"""
Test for Bronze Level Data Management
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.bronze import basic_features, load_data, quick_preprocess


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
