"""
Test for Gold Level Data Management
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from src.data.gold import DataManager


class TestDataManager:
    """DataManager class tests"""

    def test_init_default(self):
        """Test DataManager initialization with defaults"""
        manager = DataManager()
        assert manager.config["database"]["path"] == "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"
        assert manager.config["cache"]["directory"] == "/tmp/ml_cache"
        assert manager.conn is None

    @patch('pathlib.Path.mkdir')
    def test_init_with_config(self, mock_mkdir):
        """Test DataManager initialization with custom config"""
        custom_config = {
            "database": {"path": "/custom/path.db"},
            "cache": {"directory": "/custom/cache"}
        }
        manager = DataManager(custom_config)
        assert manager.config == custom_config
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.mkdir')
    @patch('src.data.gold.duckdb.connect')
    def test_default_config(self, mock_connect, mock_mkdir):
        """Test default configuration"""
        manager = DataManager()
        config = manager._default_config()
        
        assert "database" in config
        assert "cache" in config
        assert config["database"]["schema"] == "playground_series_s5e7"

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('pathlib.Path.mkdir')
    def test_get_data_with_cache(self, mock_mkdir, mock_pickle_load, mock_file, mock_exists):
        """Test data retrieval with cache hit"""
        mock_exists.return_value = True
        mock_data = (pd.DataFrame({'train': [1, 2]}), pd.DataFrame({'test': [3, 4]}))
        mock_pickle_load.return_value = mock_data
        
        manager = DataManager()
        train, test = manager.get_data(cache=True)
        
        assert len(train) == 2
        assert len(test) == 2
        mock_pickle_load.assert_called_once()

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    @patch('src.data.gold.duckdb.connect')
    @patch('pathlib.Path.mkdir')
    def test_get_data_without_cache(self, mock_mkdir, mock_connect, mock_pickle_dump, mock_file, mock_exists):
        """Test data retrieval without cache"""
        mock_exists.return_value = False
        
        # Mock database connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        mock_train = pd.DataFrame({'id': [1, 2], 'Personality': ['Introvert', 'Extrovert']})
        mock_test = pd.DataFrame({'id': [3, 4]})
        
        mock_conn.execute.side_effect = [
            MagicMock(df=lambda: mock_train),
            MagicMock(df=lambda: mock_test)
        ]
        
        manager = DataManager()
        train, test = manager.get_data(cache=True)
        
        assert len(train) == 2
        assert len(test) == 2
        mock_pickle_dump.assert_called_once()

    @patch('src.data.gold.duckdb.connect')
    @patch('pathlib.Path.mkdir')
    def test_process_data_basic_features(self, mock_mkdir, mock_connect):
        """Test data processing with basic features"""
        # Mock database connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        mock_train = pd.DataFrame({
            'Time_spent_Alone': [1.0, None, 3.0],
            'Social_event_attendance': [2.0, 3.0, 4.0],
            'Stage_fear': ['Yes', 'No', 'Yes'],
            'Drained_after_socializing': ['No', 'Yes', 'No']
        })
        mock_test = pd.DataFrame({
            'Time_spent_Alone': [2.0, 3.0],
            'Social_event_attendance': [1.0, 2.0]
        })
        
        mock_conn.execute.side_effect = [
            MagicMock(df=lambda: mock_train),
            MagicMock(df=lambda: mock_test)
        ]
        
        manager = DataManager()
        train, test = manager._process_data(["basic"])
        
        # Assert
        assert 'Stage_fear_encoded' in train.columns
        assert 'social_ratio' in train.columns
        assert train['Time_spent_Alone'].isna().sum() == 0  # Missing values filled

    @patch('src.data.gold.duckdb.connect')
    @patch('pathlib.Path.mkdir')
    def test_process_data_advanced_features(self, mock_mkdir, mock_connect):
        """Test data processing with advanced features"""
        # Mock database connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        mock_train = pd.DataFrame({
            'Time_spent_Alone': [1.0, 2.0, 3.0],
            'Social_event_attendance': [2.0, 3.0, 4.0],
            'Going_outside': [1.0, 2.0, 3.0],
            'Stage_fear': ['Yes', 'No', 'Yes'],
            'Drained_after_socializing': ['No', 'Yes', 'No']
        })
        mock_test = pd.DataFrame({
            'Time_spent_Alone': [2.0, 3.0],
            'Social_event_attendance': [1.0, 2.0]
        })
        
        mock_conn.execute.side_effect = [
            MagicMock(df=lambda: mock_train),
            MagicMock(df=lambda: mock_test)
        ]
        
        manager = DataManager()
        train, test = manager._process_data(["basic", "advanced"])
        
        # Assert
        assert 'fear_drained_interaction' in train.columns
        assert 'numeric_mean' in train.columns
        assert 'numeric_std' in train.columns

    def test_add_basic_features(self):
        """Test basic feature addition"""
        df = pd.DataFrame({
            'Time_spent_Alone': [1.0, None, 3.0],
            'Social_event_attendance': [2.0, 3.0, 4.0],
            'Going_outside': [1.0, 2.0, 3.0],
            'Stage_fear': ['Yes', 'No', 'Yes'],
            'Drained_after_socializing': ['No', 'Yes', 'No']
        })
        
        manager = DataManager()
        result = manager._add_basic_features(df)
        
        # Assert
        assert result['Time_spent_Alone'].isna().sum() == 0
        assert 'Stage_fear_encoded' in result.columns
        assert 'Drained_after_socializing_encoded' in result.columns
        assert 'social_ratio' in result.columns
        assert 'activity_sum' in result.columns

    def test_add_advanced_features(self):
        """Test advanced feature addition"""
        df = pd.DataFrame({
            'Time_spent_Alone': [1.0, 2.0, 3.0],
            'Social_event_attendance': [2.0, 3.0, 4.0],
            'Going_outside': [1.0, 2.0, 3.0],
            'Stage_fear_encoded': [1, 0, 1],
            'Drained_after_socializing_encoded': [0, 1, 0]
        })
        
        manager = DataManager()
        result = manager._add_advanced_features(df)
        
        # Assert
        assert 'fear_drained_interaction' in result.columns
        assert 'numeric_mean' in result.columns
        assert 'numeric_std' in result.columns

    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.mkdir')
    def test_clear_cache(self, mock_mkdir, mock_glob):
        """Test cache clearing"""
        mock_file = MagicMock()
        mock_glob.return_value = [mock_file]
        
        manager = DataManager()
        manager.clear_cache()
        
        mock_file.unlink.assert_called_once()

    @patch('pathlib.Path.mkdir')
    def test_close(self, mock_mkdir):
        """Test connection close"""
        manager = DataManager()
        mock_conn = MagicMock()
        manager.conn = mock_conn
        
        manager.close()
        
        mock_conn.close.assert_called_once()
        assert manager.conn is None