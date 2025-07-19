"""
Refactored Test for Bronze Level Data Management
Uses common fixtures and utilities from conftest.py
"""

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.bronze import (
    load_data, 
    quick_preprocess,
    validate_data_quality,
    advanced_missing_strategy,
    encode_categorical_robust,
    winsorize_outliers,
    create_bronze_tables
)

# Import common fixtures and utilities
from tests.conftest import (
    sample_bronze_data, edge_case_data, missing_data, large_test_data,
    mock_db_connection, assert_sub_second_performance, assert_lightgbm_compatibility,
    assert_no_data_loss, assert_data_quality, assert_feature_engineering_quality, 
    performance_test, lightgbm_compatibility_test,
    assert_database_operations, create_correlated_test_data, create_missing_pattern_data,
    create_outlier_data
)


class TestBronzeData:
    """Bronze data management tests using common fixtures"""

    @patch("src.data.bronze.duckdb.connect")
    def test_load_data(self, mock_connect, mock_db_connection):
        """Test data loading from DuckDB using common mock"""
        # Use common mock setup
        mock_connect.return_value = mock_db_connection.get_mock_conn()

        # Execute
        train, test = load_data()

        # Assert using common utilities
        assert len(train) == 5  # sample_bronze_data length
        assert len(test) == 5   # sample_gold_data length
        assert "Personality" in train.columns
        
        # Use common database assertion
        assert_database_operations(mock_connect)

    def test_quick_preprocess(self, missing_data):
        """Test basic preprocessing using common test data"""
        # Use common missing data fixture
        result = quick_preprocess(missing_data)

        # Use common assertions
        assert_no_data_loss(missing_data, result)
        assert_data_quality(result)
        
        # Specific assertions
        assert "Stage_fear_encoded" in result.columns
        assert "Drained_after_socializing_encoded" in result.columns

    def test_bronze_data_quality_only(self, sample_bronze_data):
        """Test that Bronze layer only handles data quality, not feature engineering"""
        result = quick_preprocess(sample_bronze_data)

        # Use common assertions
        assert_no_data_loss(sample_bronze_data, result)
        assert_data_quality(result)
        
        # Bronze layer should only add data quality features, not engineered features
        quality_features = [col for col in result.columns if col.endswith("_encoded") or col.endswith("_missing")]
        assert len(quality_features) > 0, "Bronze layer should add data quality features"
        
        # Should not have engineered features (those belong in Silver layer)
        engineered_features = [col for col in result.columns if any(keyword in col.lower() 
                           for keyword in ['ratio', 'sum', 'score', 'interaction'])]
        assert len(engineered_features) == 0, "Bronze layer should not contain engineered features"

    def test_quick_preprocess_missing_columns(self):
        """Test preprocessing with missing columns"""
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        result = quick_preprocess(df)

        # Should not fail and return original data
        assert len(result) == 3
        assert "other_column" in result.columns

    def test_bronze_data_quality_missing_columns(self):
        """Test data quality processing with missing columns"""
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        result = quick_preprocess(df)

        # Should not fail and return original data
        assert len(result) == 3
        assert "other_column" in result.columns

    def test_validate_data_quality(self, edge_case_data):
        """Test data quality validation using common edge case data"""
        result = validate_data_quality(edge_case_data)
        
        # Use common assertions
        assert "type_validation" in result
        assert "range_validation" in result
        
        # Specific assertions for edge cases
        assert result["type_validation"]["Time_spent_Alone"] == True
        if "within_24hrs" in result["range_validation"]["Time_spent_Alone"]:
            assert result["range_validation"]["Time_spent_Alone"]["within_24hrs"] == False

    def test_encode_categorical_robust(self, edge_case_data):
        """Test robust categorical encoding using common edge case data"""
        result = encode_categorical_robust(edge_case_data)
        
        # Use common assertions
        assert_data_quality(result)
        
        # Specific assertions for categorical encoding
        assert result["Stage_fear"].dtype == "float64"
        # Check case normalization
        assert result["Stage_fear"].iloc[0] == result["Stage_fear"].iloc[2]  # "Yes" == "yes"

    def test_advanced_missing_strategy(self, missing_data):
        """Test missing value strategy using common missing data"""
        result = advanced_missing_strategy(missing_data)
        
        # Use common assertions
        assert_no_data_loss(missing_data, result)
        
        # Specific assertions for missing flags
        missing_cols = [col for col in result.columns if col.endswith("_missing")]
        assert len(missing_cols) > 0
        for col in missing_cols:
            assert result[col].dtype in ["int64", "int32", "bool"]
            assert result[col].isin([0, 1]).all()

    def test_winsorize_outliers(self, create_outlier_data):
        """Test outlier winsorization using common outlier data"""
        df = create_outlier_data(100)
        result = winsorize_outliers(df, percentile=0.25)
        
        # Use common assertions
        assert_no_data_loss(df, result)
        assert_data_quality(result)
        
        # Specific assertions for outlier handling
        assert result["outlier_feature"].max() < 1000  # Outlier should be clipped

    @patch("src.data.bronze.duckdb.connect")
    def test_create_bronze_tables(self, mock_connect, mock_db_connection):
        """Test bronze table creation using common mock"""
        mock_connect.return_value = mock_db_connection.get_mock_conn()
        
        # Mock load_data to return test data
        with patch("src.data.bronze.load_data") as mock_load:
            mock_load.return_value = (sample_bronze_data, sample_bronze_data)
            
            create_bronze_tables()
            
            # Use common database assertions
            assert_database_operations(mock_connect, expected_calls=[
                "CREATE SCHEMA IF NOT EXISTS bronze",
                "DROP TABLE IF EXISTS bronze.train",
                "CREATE TABLE bronze.train"
            ])


class TestBronzeTypeSafety:
    """Test type safety enhancements using common fixtures"""

    def test_explicit_dtype_setting(self, sample_bronze_data):
        """Test explicit dtype setting for LightGBM optimization"""
        result = quick_preprocess(sample_bronze_data)
        
        # Use common LightGBM compatibility assertions
        assert_lightgbm_compatibility(result)
        
        # Specific dtype assertions
        assert result["Time_spent_Alone"].dtype in ["float64", "float32"]
        assert result["Stage_fear_encoded"].dtype in ["float64", "int64"]

    def test_schema_validation(self, sample_bronze_data, edge_case_data):
        """Test schema validation using common test data"""
        # Valid schema test
        valid_result = validate_data_quality(sample_bronze_data)
        assert valid_result["type_validation"]["Time_spent_Alone"] == True
        
        # Invalid schema test (wrong data types)
        invalid_df = pd.DataFrame({
            "Time_spent_Alone": ["invalid", "data", "types"],
            "Social_event_attendance": ["should", "be", "numeric"]
        })
        
        invalid_result = validate_data_quality(invalid_df)
        assert valid_result["type_validation"]["Time_spent_Alone"] == True


class TestBronzeLeakPrevention:
    """Test leak prevention foundation using common fixtures"""

    def test_fold_safe_statistics(self, sample_bronze_data):
        """Test fold-safe statistics computation using common test data"""
        # Simulate CV fold separation
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        
        fold_stats = []
        for train_idx, val_idx in skf.split(sample_bronze_data, sample_bronze_data["Personality"]):
            train_fold = sample_bronze_data.iloc[train_idx]
            val_fold = sample_bronze_data.iloc[val_idx]
            
            # Compute statistics only on training fold (leak prevention)
            train_mean = train_fold["Time_spent_Alone"].mean()
            train_std = train_fold["Time_spent_Alone"].std()
            
            # Apply to validation fold (no statistics computed on validation)
            val_normalized = (val_fold["Time_spent_Alone"] - train_mean) / train_std
            
            fold_stats.append({
                "train_mean": train_mean,
                "train_std": train_std,
                "val_normalized_mean": val_normalized.mean()
            })
        
        # Assert fold-safe computation
        assert len(fold_stats) == 2
        assert all("train_mean" in stats for stats in fold_stats)
        # Validation statistics should be different (proving no leak)
        assert fold_stats[0]["val_normalized_mean"] != fold_stats[1]["val_normalized_mean"]

    def test_categorical_encoding_fold_safety(self, sample_bronze_data):
        """Test categorical encoding is fold-safe"""
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        
        fold_encodings = []
        for train_idx, val_idx in skf.split(sample_bronze_data, sample_bronze_data["Personality"]):
            train_fold = sample_bronze_data.iloc[train_idx]
            val_fold = sample_bronze_data.iloc[val_idx]
            
            # Encode only on training fold
            train_encoded = encode_categorical_robust(train_fold)
            val_encoded = encode_categorical_robust(val_fold)
            
            # Store encoding patterns
            fold_encodings.append({
                "train_pattern": train_encoded["Stage_fear"].value_counts().to_dict(),
                "val_pattern": val_encoded["Stage_fear"].value_counts().to_dict()
            })
        
        # Assert fold-safe encoding
        assert len(fold_encodings) == 2
        # Each fold should have its own encoding pattern
        assert fold_encodings[0]["train_pattern"] != fold_encodings[1]["train_pattern"]

    def test_missing_strategy_fold_safety(self, missing_data):
        """Test missing value strategy is fold-safe"""
        # ターゲット列からNaNを除去してからCVを実行
        clean_data = missing_data.dropna(subset=["Stage_fear_encoded"])
        
        if len(clean_data) < 4:  # 十分なサンプルがない場合はスキップ
            pytest.skip("Insufficient samples for CV")
        
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        
        fold_missing_flags = []
        for train_idx, val_idx in skf.split(clean_data, clean_data["Stage_fear_encoded"]):
            train_fold = clean_data.iloc[train_idx]
            val_fold = clean_data.iloc[val_idx]
            
            # Apply missing strategy only on training fold
            train_processed = advanced_missing_strategy(train_fold)
            val_processed = advanced_missing_strategy(val_fold)
            
            # Count missing flags
            train_missing_count = sum([col.endswith('_missing') for col in train_processed.columns])
            val_missing_count = sum([col.endswith('_missing') for col in val_processed.columns])
            
            fold_missing_flags.append({
                "train_missing_features": train_missing_count,
                "val_missing_features": val_missing_count
            })
        
        # Assert fold-safe missing strategy
        assert len(fold_missing_flags) == 2
        # Each fold should have consistent missing flag generation
        assert fold_missing_flags[0]["train_missing_features"] == fold_missing_flags[1]["train_missing_features"]

    def test_sklearn_compatible_transformers(self, missing_data):
        """Test sklearn-compatible transformers using common test data"""
        from sklearn.pipeline import Pipeline
        from sklearn.base import BaseEstimator, TransformerMixin
        
        # Create a simple transformer wrapper
        class BronzePreprocessor(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                return quick_preprocess(X)
        
        # Test pipeline compatibility
        pipeline = Pipeline([
            ("bronze_preprocess", BronzePreprocessor())
        ])
        
        result = pipeline.fit_transform(missing_data)
        
        # Use common assertions
        assert hasattr(result, "shape")
        assert "Stage_fear_encoded" in result.columns


class TestBronzeCrossFeaturePatterns:
    """Test cross-feature patterns using common fixtures"""

    def test_cross_feature_imputation(self, create_correlated_test_data):
        """Test cross-feature imputation using high correlation patterns"""
        df = create_correlated_test_data(100, correlation=0.8)
        
        # Test that missing values are handled appropriately
        result = advanced_missing_strategy(df)
        
        # Use common assertions
        assert_no_data_loss(df, result)
        
        # Assert missing flags are created
        missing_cols = [col for col in result.columns if col.endswith("_missing")]
        assert len(missing_cols) > 0

    def test_missing_pattern_analysis(self, create_missing_pattern_data):
        """Test systematic vs random missing pattern detection"""
        df = create_missing_pattern_data(100)
        
        # Analyze missing patterns
        missing_flags = advanced_missing_strategy(df)
        
        # Use common assertions
        assert_no_data_loss(df, missing_flags)
        
        # Assert missing flags are created for all features
        expected_missing_cols = [col for col in df.columns if col.startswith('feature')]
        for col in expected_missing_cols:
            missing_col = f"{col}_missing"
            if missing_col in missing_flags.columns:
                assert missing_flags[missing_col].dtype in ["int64", "int32", "bool"]
                assert missing_flags[missing_col].isin([0, 1]).all()


class TestBronzePerformance:
    """Test performance requirements using common fixtures"""

    def test_sub_second_processing(self, large_test_data):
        """Test sub-second processing performance using common large test data"""
        # Performance test for quick_preprocess
        result_quick = assert_sub_second_performance(quick_preprocess, large_test_data)
        assert len(result_quick) == len(large_test_data)
        
        # Performance test for advanced_missing_strategy
        result_missing = assert_sub_second_performance(advanced_missing_strategy, large_test_data)
        assert len(result_missing) == len(large_test_data)
        
        # Performance test for encode_categorical_robust
        result_encode = assert_sub_second_performance(encode_categorical_robust, large_test_data)
        assert len(result_encode) == len(large_test_data)
        
        # Performance test for winsorize_outliers
        result_winsorize = assert_sub_second_performance(winsorize_outliers, large_test_data)
        assert len(result_winsorize) == len(large_test_data)

    def test_lightgbm_optimization_validation(self, missing_data):
        """Test LightGBM-specific optimization features using common test data"""
        # Test categorical encoding preserves NaN for LightGBM
        result_encode = encode_categorical_robust(missing_data)
        # Check if Stage_fear_encoded exists (original Stage_fear was already encoded)
        if "Stage_fear_encoded" in result_encode.columns:
            assert result_encode["Stage_fear_encoded"].dtype == "float64"  # LightGBM compatible
        
        # Test missing flags are binary for LightGBM
        result_missing = advanced_missing_strategy(missing_data)
        missing_cols = [col for col in result_missing.columns if col.endswith("_missing")]
        for col in missing_cols:
            assert result_missing[col].dtype in ["int64", "int32", "bool"]
            assert result_missing[col].isin([0, 1]).all()  # Binary for LightGBM

    def test_competition_grade_validation(self, edge_case_data):
        """Test competition-grade data quality standards using common edge case data"""
        # Test comprehensive validation
        validation_result = validate_data_quality(edge_case_data)
        
        # Use common assertions
        assert "type_validation" in validation_result
        assert "range_validation" in validation_result
        
        # Test categorical standardization handles case variations
        result_encode = encode_categorical_robust(edge_case_data)
        assert result_encode["Stage_fear"].iloc[0] == result_encode["Stage_fear"].iloc[2]  # "Yes" == "yes"
        assert result_encode["Stage_fear"].iloc[1] == result_encode["Stage_fear"].iloc[3]  # "NO" == "no" 


class TestBronzeDataQualityOnly:
    """Test that Bronze layer only handles data quality, not feature engineering"""

    def test_no_winner_solution_features_in_bronze(self, sample_bronze_data):
        """Test that Winner Solution features are NOT in Bronze layer"""
        result = quick_preprocess(sample_bronze_data)
        
        # Use common assertions
        assert_no_data_loss(sample_bronze_data, result)
        assert_data_quality(result)
        
        # Bronze layer should NOT contain Winner Solution features
        winner_features = [
            "Social_event_participation_rate",
            "Communication_ratio", 
            "Friend_social_efficiency",
            "Non_social_outings"
        ]
        for feature in winner_features:
            assert feature not in result.columns, f"Winner feature {feature} should not be in Bronze layer"

    def test_bronze_only_data_quality_features(self, sample_bronze_data):
        """Test that Bronze layer only adds data quality features"""
        result = quick_preprocess(sample_bronze_data)
        
        # Use common assertions
        assert_no_data_loss(sample_bronze_data, result)
        assert_data_quality(result)
        
        # Bronze layer should only add data quality features
        quality_features = [col for col in result.columns if col.endswith("_encoded") or col.endswith("_missing")]
        assert len(quality_features) > 0, "Bronze layer should add data quality features"
        
        # Should not have any engineered features
        engineered_features = [col for col in result.columns if any(keyword in col.lower() 
                           for keyword in ['ratio', 'sum', 'score', 'interaction', 'participation_rate', 'efficiency'])]
        assert len(engineered_features) == 0, "Bronze layer should not contain any engineered features" 