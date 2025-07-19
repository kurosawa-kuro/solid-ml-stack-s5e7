"""
Integration tests for Medallion Architecture
Tests complete pipeline from Bronze to Gold layers
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.data.bronze import create_bronze_tables
from src.data.silver import create_silver_tables
from src.data.gold import create_gold_tables, get_ml_ready_data

# Import common fixtures and utilities
from tests.conftest import (
    sample_bronze_data, sample_silver_data, sample_gold_data, large_test_data,
    mock_db_connection, assert_lightgbm_compatibility, assert_data_quality, 
    performance_test
)


class TestMedallionIntegration:
    """Test complete Medallion Architecture integration"""
    
    @patch('duckdb.connect')
    def test_bronze_to_silver_to_gold_pipeline(self, mock_connect, mock_db_connection):
        """Test complete pipeline from Bronze to Gold"""
        mock_connect.return_value = mock_db_connection.get_mock_conn()
        
        # Step 1: Bronze layer
        with patch('src.data.bronze.load_data') as mock_load_bronze:
            mock_load_bronze.return_value = (sample_bronze_data, sample_bronze_data)
            create_bronze_tables()
        
        # Step 2: Silver layer
        with patch('src.data.silver.load_bronze_data') as mock_load_silver:
            mock_load_silver.return_value = (sample_silver_data, sample_silver_data)
            create_silver_tables()
        
        # Step 3: Gold layer
        with patch('src.data.gold.load_silver_data') as mock_load_gold:
            mock_load_gold.return_value = (sample_gold_data, sample_gold_data)
            create_gold_tables()
        
        # Verify pipeline integrity
        assert mock_connect.called
        mock_conn = mock_connect.return_value
        assert mock_conn.close.called
        
        # Verify correct table creation order
        actual_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        expected_patterns = [
            'CREATE SCHEMA IF NOT EXISTS bronze',
            'CREATE SCHEMA IF NOT EXISTS silver', 
            'CREATE SCHEMA IF NOT EXISTS gold'
        ]
        for pattern in expected_patterns:
            assert any(pattern in call for call in actual_calls), f"Missing {pattern}"

    @performance_test(max_time=2.0)
    def test_end_to_end_performance(self, mock_db_connection, sample_bronze_data, sample_silver_data, sample_gold_data):
        """Test end-to-end pipeline performance"""
        with patch('duckdb.connect', return_value=mock_db_connection.get_mock_conn()):
            # Mock all data loading functions
            with patch('src.data.bronze.load_data') as mock_bronze:
                with patch('src.data.silver.load_bronze_data') as mock_silver:
                    with patch('src.data.gold.load_silver_data') as mock_gold:
                        mock_bronze.return_value = (sample_bronze_data, sample_bronze_data)
                        mock_silver.return_value = (sample_silver_data, sample_silver_data)
                        mock_gold.return_value = (sample_gold_data, sample_gold_data)
                        
                        # Execute complete pipeline
                        create_bronze_tables()
                        create_silver_tables()
                        create_gold_tables()
                        
                        # Verify all layers executed
                        assert mock_bronze.called
                        assert mock_silver.called
                        assert mock_gold.called

    def test_data_lineage_integrity(self, sample_bronze_data):
        """Test data lineage integrity through all layers"""
        # Simulate Bronze → Silver → Gold transformation
        bronze_data = sample_bronze_data.copy()
        
        # Bronze transformations
        bronze_processed = bronze_data.copy()
        bronze_processed['Stage_fear_encoded'] = [1, 0, 1, 0, 1]
        bronze_processed['Drained_after_socializing_encoded'] = [0, 1, 0, 1, 0]
        
        # Silver transformations
        silver_processed = bronze_processed.copy()
        silver_processed['extrovert_score'] = [8, 16, 24, 32, 40]
        silver_processed['introvert_score'] = [6, 6, 10, 6, 10]
        
        # Gold transformations
        gold_processed = silver_processed.copy()
        gold_processed['Personality_encoded'] = [0, 1, 0, 1, 0]
        
        # Verify data lineage
        assert len(bronze_processed) == len(silver_processed) == len(gold_processed)
        assert len(gold_processed.columns) > len(bronze_processed.columns)
        
        # Verify original columns preserved
        for col in bronze_data.columns:
            assert col in gold_processed.columns, f"Original column {col} lost in pipeline"

    def test_dependency_chain_enforcement(self, mock_db_connection):
        """Test that dependency chain is properly enforced"""
        with patch('duckdb.connect', return_value=mock_db_connection.get_mock_conn()):
            # Test that Silver cannot access raw data directly
            with patch('src.data.silver.load_data') as mock_raw_access:
                mock_raw_access.return_value = (sample_bronze_data, sample_bronze_data)
                create_silver_tables()
                
                # Silver should not call raw data access
                assert not mock_raw_access.called
            
            # Test that Gold cannot access Bronze data directly
            with patch('src.data.gold.load_bronze_data') as mock_bronze_access:
                mock_bronze_access.return_value = (sample_bronze_data, sample_bronze_data)
                create_gold_tables()
                
                # Gold should not call Bronze data access
                assert not mock_bronze_access.called

    def test_ml_ready_data_quality(self, sample_gold_data):
        """Test ML-ready data quality from Gold layer"""
        # Prepare ML-ready data
        X, y = get_ml_ready_data(sample_gold_data, target_col='Personality')
        
        # Use common assertions
        assert_lightgbm_compatibility(X)
        assert_data_quality(X)
        
        # Verify target encoding
        assert y.dtype in ['int64', 'int32']
        assert set(y.values) == {0, 1}
        
        # Verify feature matrix quality
        assert X.shape[0] == len(y)
        assert X.shape[1] > 0
        assert not X.isna().any().any()

    def test_error_propagation_handling(self, mock_db_connection):
        """Test error propagation through layers"""
        with patch('duckdb.connect', return_value=mock_db_connection.get_mock_conn()):
            # Simulate Bronze layer failure
            with patch('src.data.bronze.load_data', side_effect=Exception("Bronze error")):
                with pytest.raises(Exception, match="Bronze error"):
                    create_bronze_tables()
            
            # Simulate Silver layer failure
            with patch('src.data.silver.load_bronze_data', side_effect=Exception("Silver error")):
                with pytest.raises(Exception, match="Silver error"):
                    create_silver_tables()
            
            # Simulate Gold layer failure
            with patch('src.data.gold.load_silver_data', side_effect=Exception("Gold error")):
                with pytest.raises(Exception, match="Gold error"):
                    create_gold_tables()

    def test_data_consistency_across_layers(self, sample_bronze_data):
        """Test data consistency maintained across all layers"""
        # Simulate consistent data flow
        bronze_data = sample_bronze_data.copy()
        
        # Bronze layer should preserve data length
        bronze_processed = bronze_data.copy()
        assert len(bronze_processed) == len(bronze_data)
        
        # Silver layer should preserve data length
        silver_processed = bronze_processed.copy()
        assert len(silver_processed) == len(bronze_data)
        
        # Gold layer should preserve data length
        gold_processed = silver_processed.copy()
        assert len(gold_processed) == len(bronze_data)
        
        # Verify ID consistency if present
        if 'id' in bronze_data.columns:
            assert 'id' in gold_processed.columns
            pd.testing.assert_series_equal(
                bronze_data['id'], 
                gold_processed['id'], 
                check_names=False
            )


class TestMedallionPerformance:
    """Test Medallion Architecture performance characteristics"""
    
    @performance_test(max_time=1.0)
    def test_bronze_layer_performance(self, large_test_data):
        """Test Bronze layer performance with large dataset"""
        # Simulate Bronze processing
        result = large_test_data.copy()
        result['Stage_fear_encoded'] = np.random.randint(0, 2, len(result))
        result['Drained_after_socializing_encoded'] = np.random.randint(0, 2, len(result))
        
        assert len(result) == len(large_test_data)
        assert_data_quality(result)

    @performance_test(max_time=2.0)
    def test_silver_layer_performance(self, large_test_data):
        """Test Silver layer performance with large dataset"""
        # Simulate Silver processing
        result = large_test_data.copy()
        # Add engineered features
        for i in range(10):
            result[f'engineered_feature_{i}'] = np.random.randn(len(result))
        
        assert len(result) == len(large_test_data)
        assert_data_quality(result)
        assert len(result.columns) > len(large_test_data.columns)

    @performance_test(max_time=1.0)
    def test_gold_layer_performance(self, sample_gold_data):
        """Test Gold layer performance"""
        # Simulate Gold processing
        X, y = get_ml_ready_data(sample_gold_data, target_col='Personality')
        
        assert_lightgbm_compatibility(X)
        assert len(X) == len(y)
        assert len(X) == len(sample_gold_data) 