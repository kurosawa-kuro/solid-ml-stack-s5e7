"""
Test for Silver Level Data Management
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.silver import DataPipeline, FeatureStore


class TestDataPipeline:
    """DataPipeline class tests"""

    def test_init(self):
        """Test DataPipeline initialization"""
        pipeline = DataPipeline("/path/to/db")
        assert pipeline.db_path == "/path/to/db"
        assert pipeline.config == {}
        assert pipeline.conn is None

    def test_init_with_config(self):
        """Test DataPipeline initialization with config"""
        config = {"missing_strategy": "mean"}
        pipeline = DataPipeline("/path/to/db", config)
        assert pipeline.config == config

    @patch('src.data.silver.duckdb.connect')
    def test_load_raw(self, mock_connect):
        """Test raw data loading"""
        # Mock setup
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        mock_train = pd.DataFrame({'id': [1, 2], 'target': [0, 1]})
        mock_test = pd.DataFrame({'id': [3, 4]})
        
        mock_conn.execute.side_effect = [
            MagicMock(df=lambda: mock_train),
            MagicMock(df=lambda: mock_test)
        ]
        
        pipeline = DataPipeline("/path/to/db")
        
        # Execute
        train, test = pipeline.load_raw()
        
        # Assert
        assert len(train) == 2
        assert len(test) == 2

    def test_preprocess_median_strategy(self):
        """Test preprocessing with median strategy"""
        df = pd.DataFrame({
            'numeric_col': [1.0, None, 3.0, 5.0],
            'Stage_fear': ['Yes', 'No', 'Yes', 'No'],
            'Drained_after_socializing': ['No', 'Yes', 'No', 'Yes']
        })
        
        pipeline = DataPipeline("/path/to/db", {"missing_strategy": "median"})
        result = pipeline.preprocess(df)
        
        # Assert
        assert result['numeric_col'].isna().sum() == 0
        assert result['numeric_col'].iloc[1] == 3.0  # median of [1, 3, 5]
        assert 'Stage_fear_encoded' in result.columns
        assert 'Drained_after_socializing_encoded' in result.columns

    def test_preprocess_mean_strategy(self):
        """Test preprocessing with mean strategy"""
        df = pd.DataFrame({
            'numeric_col': [1.0, None, 3.0, 5.0]
        })
        
        pipeline = DataPipeline("/path/to/db", {"missing_strategy": "mean"})
        result = pipeline.preprocess(df)
        
        # Assert
        assert result['numeric_col'].isna().sum() == 0
        assert result['numeric_col'].iloc[1] == 3.0  # mean of [1, 3, 5]

    def test_engineer_features_basic(self):
        """Test basic feature engineering"""
        df = pd.DataFrame({
            'Time_spent_Alone': [1.0, 2.0, 3.0],
            'Social_event_attendance': [2.0, 4.0, 6.0],
            'Going_outside': [1.0, 2.0, 3.0]
        })
        
        pipeline = DataPipeline("/path/to/db", {"features": ["basic"]})
        result = pipeline.engineer_features(df)
        
        # Assert
        assert 'social_ratio' in result.columns
        assert 'activity_sum' in result.columns

    def test_engineer_features_interaction(self):
        """Test interaction feature engineering"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        pipeline = DataPipeline("/path/to/db", {"features": ["interaction"]})
        result = pipeline.engineer_features(df)
        
        # Should return original df since interaction features not implemented
        assert len(result) == 3

    def test_close(self):
        """Test connection close"""
        pipeline = DataPipeline("/path/to/db")
        mock_conn = MagicMock()
        pipeline.conn = mock_conn
        
        pipeline.close()
        
        mock_conn.close.assert_called_once()
        assert pipeline.conn is None


class TestFeatureStore:
    """FeatureStore class tests"""

    def test_init(self):
        """Test FeatureStore initialization"""
        store = FeatureStore("/path/to/db")
        assert store.db_path == "/path/to/db"
        assert store.conn is None

    @patch('src.data.silver.duckdb.connect')
    def test_save_features(self, mock_connect):
        """Test feature saving"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        store = FeatureStore("/path/to/db")
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        # Should not raise error
        store.save_features(df, "test_features")

    @patch('src.data.silver.duckdb.connect')
    def test_load_features(self, mock_connect):
        """Test feature loading"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        store = FeatureStore("/path/to/db")
        result = store.load_features("test_features")
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_close(self):
        """Test connection close"""
        store = FeatureStore("/path/to/db")
        mock_conn = MagicMock()
        store.conn = mock_conn
        
        store.close()
        
        mock_conn.close.assert_called_once()
        assert store.conn is None