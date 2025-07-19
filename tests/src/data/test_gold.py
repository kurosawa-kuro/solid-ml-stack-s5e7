"""
Test for Gold Level Data Functions - Success Cases Only
Includes comprehensive enhanced test cases from test_silver_gold_enhanced.py
"""

import tempfile
from unittest.mock import Mock, patch, MagicMock
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PolynomialFeatures

from src.data.gold import encode_target, prepare_model_data

try:
    from src.data.gold import (
        clean_and_validate_features,
        select_best_features,
    )
except ImportError:
    # If gold module has issues, we'll skip those tests
    clean_and_validate_features = None
    select_best_features = None


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


@pytest.mark.skipif(clean_and_validate_features is None, reason="Gold module not available")
class TestGoldCleaning:
    """Test gold.py cleaning functionality"""

    def test_clean_and_validate_features_infinite_values(self):
        """Test cleaning of infinite values"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': [1, np.inf, 3],
            'feature2': [np.nan, -np.inf, 2]
        })
        
        result = clean_and_validate_features(df)
        
        # Infinite values should be handled
        assert not np.isinf(result['feature1']).any()
        assert not np.isinf(result['feature2']).any()
        
        # ID should be unchanged
        assert result['id'].tolist() == [1, 2, 3]

    def test_clean_and_validate_features_outliers(self):
        """Test outlier handling"""
        # Create data with clear outlier
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 1000]  # 1000 is clear outlier
        })
        
        result = clean_and_validate_features(df)
        
        # Outlier should be clipped
        assert result['feature'].max() < 1000

    def test_clean_and_validate_features_missing_values(self):
        """Test missing value handling"""
        df = pd.DataFrame({
            'feature1': [1, np.nan, 3, 4, 5],
            'feature2': [np.nan, 2, 3, np.nan, 5]
        })
        
        result = clean_and_validate_features(df)
        
        # Missing values should be filled
        assert not result['feature1'].isna().any()
        assert not result['feature2'].isna().any()


@pytest.mark.skipif(select_best_features is None, reason="Gold module not available")
class TestGoldFeatureSelection:
    """Test gold.py feature selection functionality"""

    def test_select_best_features_basic(self):
        """Test basic feature selection"""
        # Create data with clear feature importance pattern
        np.random.seed(42)
        df = pd.DataFrame({
            'id': range(100),
            'important_feature': np.random.randn(100),
            'noise_feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Make important_feature actually important
        df.loc[df['target'] == 1, 'important_feature'] += 2
        
        selected_features = select_best_features(df, 'target', k=1)
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 1
        assert 'id' not in selected_features
        assert 'target' not in selected_features

    def test_select_best_features_string_target(self):
        """Test feature selection with string target"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [4, 3, 2, 1],
            'target': ['Introvert', 'Extrovert', 'Introvert', 'Extrovert']
        })
        
        selected_features = select_best_features(df, 'target', k=2)
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 2

    def test_select_best_features_fewer_than_k(self):
        """Test feature selection when features < k"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        selected_features = select_best_features(df, 'target', k=10)
        
        # Should return all available features
        assert len(selected_features) == 1
        assert 'feature1' in selected_features
