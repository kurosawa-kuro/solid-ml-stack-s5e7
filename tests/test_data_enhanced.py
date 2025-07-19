"""
Comprehensive test coverage for src/data/ modules
Targeting high-impact functions to improve coverage from 10-18% to 80%+
"""

import tempfile
from unittest.mock import Mock, patch, MagicMock
import os

import numpy as np
import pandas as pd
import pytest
import duckdb

from src.data.bronze import (
    load_data,
    quick_preprocess,
    basic_features,
    create_bronze_tables,
    load_bronze_data,
    DB_PATH,
)


class TestBronzeDataLoading:
    """Test bronze.py data loading functionality"""

    @patch('duckdb.connect')
    def test_load_data_success(self, mock_connect):
        """Test successful data loading from DuckDB"""
        # Setup mocks
        mock_conn = Mock()
        mock_train_df = pd.DataFrame({
            'id': [1, 2, 3],
            'Time_spent_Alone': [5, 3, 7],
            'Personality': ['Introvert', 'Extrovert', 'Introvert']
        })
        mock_test_df = pd.DataFrame({
            'id': [4, 5, 6],
            'Time_spent_Alone': [4, 6, 2]
        })
        
        mock_train_result = Mock()
        mock_train_result.df.return_value = mock_train_df
        mock_test_result = Mock()
        mock_test_result.df.return_value = mock_test_df
        
        mock_conn.execute.side_effect = [mock_train_result, mock_test_result]
        mock_connect.return_value = mock_conn
        
        # Test function
        train, test = load_data()
        
        # Assertions
        mock_connect.assert_called_once_with(DB_PATH)
        assert len(mock_conn.execute.call_args_list) == 2
        assert mock_conn.close.called
        
        pd.testing.assert_frame_equal(train, mock_train_df)
        pd.testing.assert_frame_equal(test, mock_test_df)

    @patch('duckdb.connect')
    def test_load_data_connection_error(self, mock_connect):
        """Test data loading with connection error"""
        mock_connect.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception, match="Database connection failed"):
            load_data()

    @patch('duckdb.connect')
    def test_load_bronze_data_success(self, mock_connect):
        """Test successful bronze data loading"""
        mock_conn = Mock()
        mock_train_df = pd.DataFrame({'id': [1, 2], 'processed': [True, True]})
        mock_test_df = pd.DataFrame({'id': [3, 4], 'processed': [True, True]})
        
        mock_train_result = Mock()
        mock_train_result.df.return_value = mock_train_df
        mock_test_result = Mock()
        mock_test_result.df.return_value = mock_test_df
        
        mock_conn.execute.side_effect = [mock_train_result, mock_test_result]
        mock_connect.return_value = mock_conn
        
        train, test = load_bronze_data()
        
        # Verify bronze table queries
        expected_calls = [
            "SELECT * FROM bronze.train",
            "SELECT * FROM bronze.test"
        ]
        actual_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert actual_calls == expected_calls


class TestDataPreprocessing:
    """Test data preprocessing functionality"""

    def test_quick_preprocess_numeric_fillna(self):
        """Test numeric column missing value handling"""
        df = pd.DataFrame({
            'Time_spent_Alone': [5, np.nan, 7, 3],
            'Social_event_attendance': [2, 4, np.nan, 6],
            'Going_outside': [1, 3, 5, np.nan],
            'Friends_circle_size': [10, np.nan, 15, 20],
            'Post_frequency': [5, 7, np.nan, 9],
            'other_col': [1, 2, 3, 4]  # Non-target column
        })
        
        result = quick_preprocess(df)
        
        # Check that NaNs are filled with median
        assert not result['Time_spent_Alone'].isna().any()
        assert not result['Social_event_attendance'].isna().any()
        assert not result['Going_outside'].isna().any()
        assert not result['Friends_circle_size'].isna().any()
        assert not result['Post_frequency'].isna().any()
        
        # Verify medians are correct
        assert result['Time_spent_Alone'].iloc[1] == 5.0  # median of [5, 7, 3]
        assert result['Going_outside'].iloc[3] == 3.0    # median of [1, 3, 5]

    def test_quick_preprocess_categorical_encoding(self):
        """Test categorical column encoding"""
        df = pd.DataFrame({
            'Stage_fear': ['Yes', 'No', 'Yes', 'No'],
            'Drained_after_socializing': ['Yes', 'Yes', 'No', 'No']
        })
        
        result = quick_preprocess(df)
        
        # Check encoded columns exist
        assert 'Stage_fear_encoded' in result.columns
        assert 'Drained_after_socializing_encoded' in result.columns
        
        # Check encoding correctness
        expected_stage_fear = [1, 0, 1, 0]
        expected_drained = [1, 1, 0, 0]
        
        assert result['Stage_fear_encoded'].tolist() == expected_stage_fear
        assert result['Drained_after_socializing_encoded'].tolist() == expected_drained

    def test_quick_preprocess_missing_columns(self):
        """Test preprocessing with missing columns"""
        df = pd.DataFrame({
            'Time_spent_Alone': [5, 3, 7],
            'other_column': [1, 2, 3]
        })
        
        result = quick_preprocess(df)
        
        # Should not fail with missing columns
        assert len(result) == 3
        assert 'Time_spent_Alone' in result.columns
        assert 'other_column' in result.columns

    def test_quick_preprocess_no_missing_values(self):
        """Test preprocessing with no missing values"""
        df = pd.DataFrame({
            'Time_spent_Alone': [5, 3, 7],
            'Social_event_attendance': [2, 4, 6],
            'Stage_fear': ['Yes', 'No', 'Yes']
        })
        
        result = quick_preprocess(df)
        
        # Data should remain unchanged except for encoding
        assert result['Time_spent_Alone'].tolist() == [5, 3, 7]
        assert result['Social_event_attendance'].tolist() == [2, 4, 6]
        assert 'Stage_fear_encoded' in result.columns

    def test_quick_preprocess_copy_behavior(self):
        """Test that preprocessing creates a copy"""
        df = pd.DataFrame({
            'Time_spent_Alone': [5, np.nan, 7],
            'Stage_fear': ['Yes', 'No', 'Yes']
        })
        
        original_has_nan = df['Time_spent_Alone'].isna().any()
        result = quick_preprocess(df)
        
        # Original should still have NaN
        assert original_has_nan
        assert df['Time_spent_Alone'].isna().any()
        
        # Result should not have NaN
        assert not result['Time_spent_Alone'].isna().any()


class TestFeatureEngineering:
    """Test feature engineering functionality"""

    def test_basic_features_social_ratio(self):
        """Test social ratio feature creation"""
        df = pd.DataFrame({
            'Social_event_attendance': [4, 6, 2],
            'Time_spent_Alone': [2, 0, 8]  # Note: +1 added to avoid division by zero
        })
        
        result = basic_features(df)
        
        assert 'social_ratio' in result.columns
        
        # Check calculations: Social_event_attendance / (Time_spent_Alone + 1)
        expected_ratios = [4/3, 6/1, 2/9]  # [4/(2+1), 6/(0+1), 2/(8+1)]
        
        np.testing.assert_array_almost_equal(
            result['social_ratio'].values, 
            expected_ratios, 
            decimal=6
        )

    def test_basic_features_activity_sum(self):
        """Test activity sum feature creation"""
        df = pd.DataFrame({
            'Going_outside': [3, 5, 1],
            'Social_event_attendance': [2, 4, 7]
        })
        
        result = basic_features(df)
        
        assert 'activity_sum' in result.columns
        expected_sums = [5, 9, 8]  # [3+2, 5+4, 1+7]
        
        assert result['activity_sum'].tolist() == expected_sums

    def test_basic_features_missing_columns(self):
        """Test feature engineering with missing required columns"""
        df = pd.DataFrame({
            'Going_outside': [3, 5, 1],
            'other_column': [1, 2, 3]
        })
        
        result = basic_features(df)
        
        # Should not create features when required columns are missing
        assert 'social_ratio' not in result.columns
        assert 'activity_sum' not in result.columns
        assert len(result) == 3

    def test_basic_features_copy_behavior(self):
        """Test that feature engineering creates a copy"""
        df = pd.DataFrame({
            'Social_event_attendance': [4, 6, 2],
            'Time_spent_Alone': [2, 0, 8],
            'Going_outside': [3, 5, 1]
        })
        
        original_columns = df.columns.tolist()
        result = basic_features(df)
        
        # Original should remain unchanged
        assert df.columns.tolist() == original_columns
        assert 'social_ratio' not in df.columns
        
        # Result should have new features
        assert 'social_ratio' in result.columns
        assert 'activity_sum' in result.columns


class TestBronzeTableCreation:
    """Test bronze table creation functionality"""

    @patch('duckdb.connect')
    @patch('src.data.bronze.quick_preprocess')
    def test_create_bronze_tables_success(self, mock_preprocess, mock_connect):
        """Test successful bronze table creation"""
        # Setup mocks
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Mock raw data
        raw_train = pd.DataFrame({'id': [1, 2], 'raw_col': ['a', 'b']})
        raw_test = pd.DataFrame({'id': [3, 4], 'raw_col': ['c', 'd']})
        
        # Mock processed data
        processed_train = pd.DataFrame({'id': [1, 2], 'processed_col': [1, 2]})
        processed_test = pd.DataFrame({'id': [3, 4], 'processed_col': [3, 4]})
        
        # Setup execute mock to return raw data
        mock_train_result = Mock()
        mock_train_result.df.return_value = raw_train
        mock_test_result = Mock()
        mock_test_result.df.return_value = raw_test
        
        mock_conn.execute.side_effect = [mock_train_result, mock_test_result] + [None] * 10
        
        # Setup preprocessing mock
        mock_preprocess.side_effect = [processed_train, processed_test]
        
        # Test function
        create_bronze_tables()
        
        # Verify database operations
        mock_connect.assert_called_with(DB_PATH)
        assert mock_conn.execute.call_count >= 6  # Schema + data loading + table operations
        assert mock_conn.register.call_count == 2
        assert mock_conn.close.called
        
        # Verify preprocessing was called
        assert mock_preprocess.call_count == 2

    @patch('duckdb.connect')
    def test_create_bronze_tables_connection_error(self, mock_connect):
        """Test bronze table creation with connection error"""
        mock_connect.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            create_bronze_tables()

    @patch('builtins.print')
    @patch('duckdb.connect')
    @patch('src.data.bronze.quick_preprocess')
    def test_create_bronze_tables_output(self, mock_preprocess, mock_connect, mock_print):
        """Test bronze table creation output messages"""
        # Setup mocks
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        processed_train = pd.DataFrame({'id': range(100)})  # 100 rows
        processed_test = pd.DataFrame({'id': range(50)})    # 50 rows
        
        # Mock execute results
        mock_result = Mock()
        mock_result.df.return_value = pd.DataFrame()
        mock_conn.execute.return_value = mock_result
        
        mock_preprocess.side_effect = [processed_train, processed_test]
        
        # Test function
        create_bronze_tables()
        
        # Verify output messages
        assert mock_print.call_count >= 3
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        
        # Check for expected output patterns
        assert any("Bronze tables created" in call for call in print_calls)
        assert any("100 rows" in call for call in print_calls)
        assert any("50 rows" in call for call in print_calls)


class TestDataIntegration:
    """Test data module integration scenarios"""

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        # Create sample raw data
        raw_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'Time_spent_Alone': [5, np.nan, 7, 3, 6],
            'Social_event_attendance': [2, 4, np.nan, 6, 8],
            'Going_outside': [1, 3, 5, np.nan, 4],
            'Friends_circle_size': [10, np.nan, 15, 20, 12],
            'Post_frequency': [5, 7, np.nan, 9, 6],
            'Stage_fear': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Drained_after_socializing': ['Yes', 'Yes', 'No', 'No', 'Yes']
        })
        
        # Apply preprocessing
        preprocessed = quick_preprocess(raw_data)
        
        # Apply feature engineering
        featured = basic_features(preprocessed)
        
        # Verify pipeline results
        assert not featured.select_dtypes(include=[np.number]).isna().any().any()
        assert 'Stage_fear_encoded' in featured.columns
        assert 'Drained_after_socializing_encoded' in featured.columns
        assert 'social_ratio' in featured.columns
        assert 'activity_sum' in featured.columns
        
        # Verify data integrity
        assert len(featured) == len(raw_data)
        assert all(featured['id'] == raw_data['id'])

    def test_edge_case_all_missing(self):
        """Test preprocessing with all missing values in a column"""
        df = pd.DataFrame({
            'Time_spent_Alone': [np.nan, np.nan, np.nan],
            'Social_event_attendance': [1, 2, 3]
        })
        
        result = quick_preprocess(df)
        
        # With all NaN, median should be NaN, but fillna should handle it
        # (The specific behavior depends on pandas version)
        assert 'Time_spent_Alone' in result.columns

    def test_edge_case_single_row(self):
        """Test preprocessing with single row"""
        df = pd.DataFrame({
            'Time_spent_Alone': [5],
            'Social_event_attendance': [3],
            'Going_outside': [2],
            'Stage_fear': ['Yes']
        })
        
        preprocessed = quick_preprocess(df)
        featured = basic_features(preprocessed)
        
        assert len(featured) == 1
        assert 'Stage_fear_encoded' in featured.columns
        assert featured['Stage_fear_encoded'].iloc[0] == 1
        assert 'social_ratio' in featured.columns
        assert 'activity_sum' in featured.columns

    def test_data_type_consistency(self):
        """Test data type consistency after processing"""
        df = pd.DataFrame({
            'Time_spent_Alone': [5.0, 3.0, 7.0],
            'Social_event_attendance': [2, 4, 6],  # int
            'Going_outside': [1.5, 3.2, 5.1],      # float
            'Stage_fear': ['Yes', 'No', 'Yes']
        })
        
        result = quick_preprocess(df)
        
        # Numeric columns should remain numeric
        assert pd.api.types.is_numeric_dtype(result['Time_spent_Alone'])
        assert pd.api.types.is_numeric_dtype(result['Social_event_attendance'])
        assert pd.api.types.is_numeric_dtype(result['Going_outside'])
        
        # Encoded categorical should be int
        assert pd.api.types.is_integer_dtype(result['Stage_fear_encoded'])


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_dataframe_input(self):
        """Test functions with invalid input"""
        # Test with None
        with pytest.raises(AttributeError):
            quick_preprocess(None)
        
        with pytest.raises(AttributeError):
            basic_features(None)

    def test_empty_dataframe(self):
        """Test functions with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        result_preprocess = quick_preprocess(empty_df)
        result_features = basic_features(empty_df)
        
        assert len(result_preprocess) == 0
        assert len(result_features) == 0

    def test_unexpected_categorical_values(self):
        """Test categorical encoding with unexpected values"""
        df = pd.DataFrame({
            'Stage_fear': ['Yes', 'No', 'Maybe', 'Unknown'],
            'Drained_after_socializing': ['Yes', 'No', 'Sometimes', None]
        })
        
        result = quick_preprocess(df)
        
        # Only 'Yes' should be encoded as 1, everything else as 0
        expected_stage_fear = [1, 0, 0, 0]
        expected_drained = [1, 0, 0, 0]
        
        assert result['Stage_fear_encoded'].tolist() == expected_stage_fear
        assert result['Drained_after_socializing_encoded'].tolist() == expected_drained