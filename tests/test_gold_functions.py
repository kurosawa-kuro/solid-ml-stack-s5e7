"""
Test for Gold Level Data Functions - Success Cases Only
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.gold import encode_target, prepare_model_data


class TestGoldFunctions:
    """Gold level function tests - minimal success cases"""

    def test_prepare_model_data_basic(self):
        """Test basic model data preparation"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "Personality": ["Introvert", "Extrovert", "Introvert"],
            }
        )

        result = prepare_model_data(df)

        # Should return processed data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_prepare_model_data_with_target(self):
        """Test model data preparation with specific target"""
        df = pd.DataFrame({"id": [1, 2, 3], "feature1": [1.0, 2.0, 3.0], "target_col": ["A", "B", "A"]})

        result = prepare_model_data(df, target_col="target_col")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_encode_target_basic(self):
        """Test basic target encoding"""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert"]})

        result = encode_target(df)

        # Should have encoded personality
        assert "Personality_encoded" in result.columns
        assert set(result["Personality_encoded"].values) == {0, 1}

    def test_encode_target_custom_column(self):
        """Test target encoding with custom column"""
        df = pd.DataFrame({"id": [1, 2, 3], "custom_target": ["A", "B", "A"]})

        result = encode_target(df, target_col="custom_target")

        assert "custom_target_encoded" in result.columns

    @patch("src.data.gold.duckdb.connect")
    def test_create_gold_tables_success(self, mock_connect):
        """Test gold table creation succeeds"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # Should not raise an error
        from src.data.gold import create_gold_tables

        create_gold_tables()

        # Verify connection was made and closed
        mock_connect.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.data.gold.duckdb.connect")
    def test_load_gold_data_success(self, mock_connect):
        """Test gold data loading succeeds"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_train = pd.DataFrame({"id": [1, 2], "feature": [1.0, 2.0]})
        mock_test = pd.DataFrame({"id": [3, 4], "feature": [3.0, 4.0]})

        mock_conn.execute.side_effect = [MagicMock(df=lambda: mock_train), MagicMock(df=lambda: mock_test)]

        from src.data.gold import load_gold_data

        train, test = load_gold_data()

        assert len(train) == 2
        assert len(test) == 2
        assert "feature" in train.columns
        assert "feature" in test.columns

    def test_empty_dataframe_handling(self):
        """Test functions handle empty DataFrames gracefully"""
        empty_df = pd.DataFrame()

        # Should not crash
        result1 = prepare_model_data(empty_df)
        result2 = encode_target(empty_df)

        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
