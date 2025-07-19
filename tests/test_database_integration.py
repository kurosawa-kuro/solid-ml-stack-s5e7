"""
Database Integration Tests
Tests for real DuckDB database interactions and data pipeline consistency
"""

import os
import tempfile
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.data.bronze import create_bronze_tables, load_bronze_data, load_data
from src.data.gold import create_gold_tables, get_ml_ready_data, load_gold_data
from src.data.silver import create_silver_tables, load_silver_data

DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"


class TestDatabaseIntegration:
    """Test database integration with real DuckDB"""

    def test_database_connectivity(self):
        """Test basic database connectivity"""
        conn = duckdb.connect(DB_PATH)
        
        # Test connection works
        result = conn.execute("SELECT 1 as test").fetchone()
        assert result[0] == 1
        
        # Test basic schema exists
        schemas = conn.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
        schema_names = [s[0] for s in schemas]
        assert "playground_series_s5e7" in schema_names
        
        conn.close()

    def test_raw_data_integrity(self):
        """Test raw data from playground_series_s5e7 schema"""
        conn = duckdb.connect(DB_PATH)
        
        # Test train data
        train_count = conn.execute("SELECT COUNT(*) FROM playground_series_s5e7.train").fetchone()[0]
        assert train_count > 0, "Train data should not be empty"
        
        # Test test data  
        test_count = conn.execute("SELECT COUNT(*) FROM playground_series_s5e7.test").fetchone()[0]
        assert test_count > 0, "Test data should not be empty"
        
        # Test columns exist
        train_cols = conn.execute("PRAGMA table_info('playground_series_s5e7.train')").fetchall()
        col_names = [col[1] for col in train_cols]
        
        expected_cols = [
            "id", "Time_spent_Alone", "Stage_fear", "Social_event_attendance",
            "Going_outside", "Drained_after_socializing", "Friends_circle_size", 
            "Post_frequency", "Personality"
        ]
        
        for col in expected_cols:
            assert col in col_names, f"Column {col} should exist in train data"
        
        # Test sample data types
        sample = conn.execute("SELECT * FROM playground_series_s5e7.train LIMIT 1").fetchone()
        assert sample is not None
        assert len(sample) == len(col_names)
        
        conn.close()

    def test_bronze_layer_operations(self):
        """Test bronze layer data operations"""
        # Test loading raw data
        train_raw, test_raw = load_data()
        
        assert len(train_raw) > 0
        assert len(test_raw) > 0
        assert "Personality" in train_raw.columns
        assert "id" in train_raw.columns and "id" in test_raw.columns
        
        # Test bronze table creation
        create_bronze_tables()
        
        # Test loading bronze data
        train_bronze, test_bronze = load_bronze_data()
        
        assert len(train_bronze) == len(train_raw)
        assert len(test_bronze) == len(test_raw)
        
        # Check that encoded columns were added
        assert "Stage_fear_encoded" in train_bronze.columns
        assert "Drained_after_socializing_encoded" in train_bronze.columns

    def test_silver_layer_operations(self):
        """Test silver layer data operations"""
        # Ensure bronze exists first
        create_bronze_tables()
        
        # Create silver tables
        create_silver_tables()
        
        # Load silver data
        train_silver, test_silver = load_silver_data()
        
        assert len(train_silver) > 0
        assert len(test_silver) > 0
        
        # Check for advanced features
        expected_features = [
            "extrovert_score", "social_ratio", "activity_sum"
        ]
        
        for feature in expected_features:
            assert feature in train_silver.columns, f"Feature {feature} should exist in silver layer"

    def test_gold_layer_operations(self):
        """Test gold layer data operations"""
        # Ensure previous layers exist
        create_bronze_tables()
        create_silver_tables()
        
        # Create gold tables
        create_gold_tables()
        
        # Load gold data
        train_gold, test_gold = load_gold_data()
        
        assert len(train_gold) > 0
        assert len(test_gold) > 0
        
        # Check for encoded target
        assert "Personality_encoded" in train_gold.columns
        assert "Personality" in train_gold.columns
        
        # Test ML-ready data extraction
        X_train, y_train, X_test, test_ids = get_ml_ready_data()
        
        assert X_train.shape[0] == len(train_gold)
        assert y_train.shape[0] == len(train_gold)
        assert X_test.shape[0] == len(test_gold)
        assert len(test_ids) == len(test_gold)
        
        # Check that features are numeric
        assert X_train.dtype in [np.float64, np.float32]
        assert y_train.dtype in [np.int64, np.int32]

    def test_end_to_end_data_pipeline(self):
        """Test complete data pipeline from raw to model-ready"""
        # Full pipeline
        create_bronze_tables()
        create_silver_tables() 
        create_gold_tables()
        
        # Get final ML data
        X_train, y_train, X_test, test_ids = get_ml_ready_data(scale_features=True)
        
        # Verify data quality
        assert not np.isnan(X_train).any(), "Training features should not have NaN"
        assert not np.isnan(y_train).any(), "Training target should not have NaN"
        assert not np.isnan(X_test).any(), "Test features should not have NaN"
        
        assert not np.isinf(X_train).any(), "Training features should not have inf"
        assert not np.isinf(X_test).any(), "Test features should not have inf"
        
        # Verify target is binary
        assert set(np.unique(y_train)) == {0, 1}, "Target should be binary 0/1"
        
        # Verify feature scaling worked (mean ~0, std ~1)
        feature_means = np.mean(X_train, axis=0)
        feature_stds = np.std(X_train, axis=0)
        
        assert np.allclose(feature_means, 0, atol=1e-10), "Scaled features should have mean ~0"
        assert np.allclose(feature_stds, 1, atol=1e-10), "Scaled features should have std ~1"

    def test_database_consistency_across_layers(self):
        """Test data consistency across bronze/silver/gold layers"""
        # Create all layers
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        # Load data from each layer
        train_bronze, _ = load_bronze_data()
        train_silver, _ = load_silver_data()
        train_gold, _ = load_gold_data()
        
        # Check row counts are consistent
        assert len(train_bronze) == len(train_silver) == len(train_gold), \
            "Row counts should be consistent across all layers"
        
        # Check ID consistency
        assert train_bronze["id"].equals(train_silver["id"]), \
            "IDs should be consistent between bronze and silver"
        assert train_silver["id"].equals(train_gold["id"]), \
            "IDs should be consistent between silver and gold"
        
        # Check original columns preserved
        original_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside"]
        for col in original_cols:
            if col in train_bronze.columns and col in train_gold.columns:
                pd.testing.assert_series_equal(
                    train_bronze[col].fillna(0), 
                    train_gold[col].fillna(0),
                    check_names=False
                )

    def test_error_handling_database_issues(self):
        """Test error handling for database connectivity issues"""
        # Test with non-existent database
        with pytest.raises(Exception):
            conn = duckdb.connect("/non/existent/path.duckdb")
            conn.execute("SELECT 1").fetchone()
        
        # Test with corrupted SQL
        conn = duckdb.connect(DB_PATH)
        with pytest.raises(Exception):
            conn.execute("INVALID SQL QUERY").fetchone()
        conn.close()

    def test_concurrent_database_access(self):
        """Test handling of concurrent database access"""
        connections = []
        try:
            # Create multiple connections
            for _ in range(3):
                conn = duckdb.connect(DB_PATH)
                connections.append(conn)
            
            # Test concurrent reads
            results = []
            for conn in connections:
                result = conn.execute("SELECT COUNT(*) FROM playground_series_s5e7.train").fetchone()
                results.append(result[0])
            
            # All should return same count
            assert len(set(results)) == 1, "Concurrent reads should return consistent results"
            
        finally:
            # Clean up connections
            for conn in connections:
                conn.close()


class TestDatabasePerformance:
    """Test database performance and optimization"""
    
    def test_query_performance(self):
        """Test query performance for typical operations"""
        import time
        
        conn = duckdb.connect(DB_PATH)
        
        # Time a typical aggregation query
        start_time = time.time()
        result = conn.execute("""
            SELECT Personality, 
                   AVG(Time_spent_Alone) as avg_alone,
                   COUNT(*) as count
            FROM playground_series_s5e7.train 
            GROUP BY Personality
        """).fetchall()
        query_time = time.time() - start_time
        
        assert query_time < 1.0, "Basic aggregation should complete quickly"
        assert len(result) == 2, "Should have 2 personality types"
        
        conn.close()

    def test_data_loading_performance(self):
        """Test performance of data loading operations"""
        import time
        
        # Time the data loading
        start_time = time.time()
        train, test = load_data()
        load_time = time.time() - start_time
        
        assert load_time < 5.0, "Data loading should complete within 5 seconds"
        assert len(train) > 1000, "Should load substantial amount of data"


class TestDatabaseMaintenance:
    """Test database maintenance and cleanup operations"""
    
    def test_table_cleanup_recreation(self):
        """Test cleaning up and recreating tables"""
        conn = duckdb.connect(DB_PATH)
        
        # Create a test table
        conn.execute("CREATE SCHEMA IF NOT EXISTS test_schema")
        conn.execute("DROP TABLE IF EXISTS test_schema.test_table")
        conn.execute("CREATE TABLE test_schema.test_table (id INT, value TEXT)")
        conn.execute("INSERT INTO test_schema.test_table VALUES (1, 'test')")
        
        # Verify it exists
        result = conn.execute("SELECT COUNT(*) FROM test_schema.test_table").fetchone()
        assert result[0] == 1
        
        # Clean up
        conn.execute("DROP TABLE IF EXISTS test_schema.test_table")
        conn.execute("DROP SCHEMA IF EXISTS test_schema")
        
        conn.close()

    def test_bronze_silver_gold_recreation(self):
        """Test recreating all data layers"""
        # This tests the robustness of the pipeline recreation
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        
        # Load data to verify everything works
        train_gold, test_gold = load_gold_data()
        assert len(train_gold) > 0
        assert len(test_gold) > 0
        
        # Recreate again (should handle existing tables)
        create_bronze_tables()  # Should drop and recreate
        create_silver_tables()  # Should drop and recreate
        create_gold_tables()    # Should drop and recreate
        
        # Verify still works
        train_gold2, test_gold2 = load_gold_data()
        assert len(train_gold2) == len(train_gold)
        assert len(test_gold2) == len(test_gold)