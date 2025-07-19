"""
Test for Silver Level Data Functions - Success Cases Only
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.silver import advanced_features, scaling_features


class TestSilverFunctions:
    """Silver level function tests - minimal success cases"""

    def test_advanced_features_basic(self):
        """Test advanced features creation"""
        df = pd.DataFrame({
            'Time_spent_Alone': [1.0, 2.0, 3.0],
            'Social_event_attendance': [2.0, 4.0, 6.0],
            'Going_outside': [1.0, 2.0, 3.0],
            'Stage_fear_encoded': [1, 0, 1],
            'Drained_after_socializing_encoded': [0, 1, 0]
        })
        
        result = advanced_features(df)
        
        # Should return processed DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_scaling_features_basic(self):
        """Test feature scaling"""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10.0, 20.0, 30.0],
            'feature3': [100.0, 200.0, 300.0]
        })
        
        result = scaling_features(df)
        
        # Should return scaled DataFrame with additional scaled columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # Original columns should still be present
        for col in df.columns:
            assert col in result.columns

    @patch('src.data.silver.duckdb.connect')
    def test_create_silver_tables_success(self, mock_connect):
        """Test silver table creation succeeds"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Should not raise an error
        from src.data.silver import create_silver_tables
        create_silver_tables()
        
        # Verify connection was made and closed
        mock_connect.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('src.data.silver.duckdb.connect')
    def test_load_silver_data_success(self, mock_connect):
        """Test silver data loading succeeds"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        mock_train = pd.DataFrame({
            'id': [1, 2], 
            'feature': [1.0, 2.0],
            'Stage_fear_encoded': [1, 0],
            'Drained_after_socializing_encoded': [0, 1]
        })
        mock_test = pd.DataFrame({
            'id': [3, 4], 
            'feature': [3.0, 4.0],
            'Stage_fear_encoded': [1, 0],
            'Drained_after_socializing_encoded': [1, 0]
        })
        
        mock_conn.execute.side_effect = [
            MagicMock(df=lambda: mock_train),
            MagicMock(df=lambda: mock_test)
        ]
        
        from src.data.silver import load_silver_data
        train, test = load_silver_data()
        
        assert len(train) == 2
        assert len(test) == 2
        assert 'Stage_fear_encoded' in train.columns
        assert 'Drained_after_socializing_encoded' in train.columns

    def test_advanced_features_with_missing_columns(self):
        """Test advanced features with missing columns"""
        df = pd.DataFrame({
            'other_column': [1, 2, 3]
        })
        
        result = advanced_features(df)
        
        # Should not crash and return data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_scaling_features_single_column(self):
        """Test scaling with single column"""
        df = pd.DataFrame({
            'single_feature': [1.0, 2.0, 3.0]
        })
        
        result = scaling_features(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'single_feature' in result.columns

    def test_empty_dataframe_handling(self):
        """Test functions handle empty DataFrames gracefully"""
        empty_df = pd.DataFrame()
        
        # Should not crash
        result1 = advanced_features(empty_df)
        result2 = scaling_features(empty_df)
        
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)