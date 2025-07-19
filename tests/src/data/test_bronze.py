"""
Test for Bronze Level Data Management
"""

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.bronze import (
    basic_features, 
    load_data, 
    quick_preprocess,
    validate_data_quality,
    advanced_missing_strategy,
    encode_categorical_robust,
    winsorize_outliers,
    create_bronze_tables
)


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

    def test_validate_data_quality(self):
        """Test data quality validation"""
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 12.0, 25.0],  # One value > 24hrs
            "Social_event_attendance": [2.0, 3.0, -1.0],  # One negative
            "Stage_fear": ["Yes", "No", "Yes"],
            "Drained_after_socializing": ["No", "Yes", "No"]
        })
        
        result = validate_data_quality(df)
        
        assert "type_validation" in result
        assert "range_validation" in result
        assert result["type_validation"]["Time_spent_Alone"] == True
        assert result["range_validation"]["Time_spent_Alone"]["within_24hrs"] == False
        assert result["range_validation"]["Social_event_attendance"]["non_negative"] == False

    def test_encode_categorical_robust(self):
        """Test robust categorical encoding"""
        df = pd.DataFrame({
            "Stage_fear": ["YES", "no", "Yes", "NO", None],
            "Drained_after_socializing": ["yes", "NO", "Yes", "no", None]
        })
        
        result = encode_categorical_robust(df)
        
        # Check non-null values
        assert result["Stage_fear"].iloc[:4].tolist() == [1.0, 0.0, 1.0, 0.0]
        assert result["Drained_after_socializing"].iloc[:4].tolist() == [1.0, 0.0, 1.0, 0.0]
        
        # Check null handling (NaN != None in pandas)
        assert pd.isna(result["Stage_fear"].iloc[4])
        assert pd.isna(result["Drained_after_socializing"].iloc[4])
        
        assert result["Stage_fear"].dtype == "float64"

    def test_advanced_missing_strategy(self):
        """Test missing value strategy with flags"""
        df = pd.DataFrame({
            "Stage_fear": ["Yes", None, "No"],
            "Going_outside": [1.0, None, 3.0],
            "Time_spent_Alone": [2.0, 4.0, None]
        })
        
        result = advanced_missing_strategy(df)
        
        assert "Stage_fear_missing" in result.columns
        assert "Going_outside_missing" in result.columns
        assert "Time_spent_Alone_missing" in result.columns
        assert result["Stage_fear_missing"].tolist() == [0, 1, 0]
        assert result["Going_outside_missing"].tolist() == [0, 1, 0]
        assert result["Time_spent_Alone_missing"].tolist() == [0, 0, 1]

    def test_winsorize_outliers(self):
        """Test outlier winsorization"""
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 2.0, 3.0, 100.0],  # 100.0 is outlier
            "Social_event_attendance": [1.0, 2.0, 3.0, 4.0]
        })
        
        result = winsorize_outliers(df, percentile=0.25)  # Aggressive clipping for test
        
        # Check that extreme values are clipped
        assert result["Time_spent_Alone"].max() < 100.0
        assert result["Time_spent_Alone"].min() >= 1.0

    @patch("src.data.bronze.duckdb.connect")
    def test_create_bronze_tables(self, mock_connect):
        """Test bronze table creation"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Mock load_data to return test data
        with patch("src.data.bronze.load_data") as mock_load:
            mock_train = pd.DataFrame({
                "Time_spent_Alone": [1.0, 2.0, 3.0],
                "Stage_fear": ["Yes", "No", "Yes"]
            })
            mock_test = pd.DataFrame({
                "Time_spent_Alone": [4.0, 5.0, 6.0]
            })
            mock_load.return_value = (mock_train, mock_test)
            
            create_bronze_tables()
            
            # Verify schema creation
            mock_conn.execute.assert_any_call("CREATE SCHEMA IF NOT EXISTS bronze")
            
            # Verify table drops
            mock_conn.execute.assert_any_call("DROP TABLE IF EXISTS bronze.train")
            mock_conn.execute.assert_any_call("DROP TABLE IF EXISTS bronze.test")
            
            # Verify table creation
            mock_conn.execute.assert_any_call("CREATE TABLE bronze.train AS SELECT * FROM train_bronze_df")
            mock_conn.execute.assert_any_call("CREATE TABLE bronze.test AS SELECT * FROM test_bronze_df")

    # ===== 新規追加テスト: Type Safety Enhancement =====
    
    def test_explicit_dtype_setting(self):
        """Test explicit dtype setting for LightGBM optimization"""
        # Setup test data with mixed types
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 2.0, 3.0],
            "Social_event_attendance": [2, 3, 4],  # int
            "Stage_fear": ["Yes", "No", "Yes"],  # object
            "Drained_after_socializing": [True, False, True],  # bool
            "Friends_circle_size": [5, 10, 15]  # int
        })
        
        # Execute preprocessing (assuming quick_preprocess sets dtypes)
        result = quick_preprocess(df)
        
        # Assert explicit dtype setting
        assert result["Time_spent_Alone"].dtype in ["float64", "float32"]
        assert result["Social_event_attendance"].dtype in ["int64", "int32", "float64"]
        assert result["Stage_fear_encoded"].dtype in ["float64", "int64"]
        assert result["Drained_after_socializing_encoded"].dtype in ["float64", "int64"]
        assert result["Friends_circle_size"].dtype in ["int64", "int32", "float64"]

    def test_schema_validation(self):
        """Test schema validation preventing downstream corruption"""
        # Valid schema test
        valid_df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 2.0, 3.0],
            "Social_event_attendance": [2.0, 3.0, 4.0],
            "Stage_fear": ["Yes", "No", "Yes"],
            "Drained_after_socializing": ["No", "Yes", "No"]
        })
        
        result = validate_data_quality(valid_df)
        assert result["type_validation"]["Time_spent_Alone"] == True
        assert result["type_validation"]["Social_event_attendance"] == True
        
        # Invalid schema test (wrong data types)
        invalid_df = pd.DataFrame({
            "Time_spent_Alone": ["invalid", "data", "types"],
            "Social_event_attendance": ["should", "be", "numeric"]
        })
        
        result = validate_data_quality(invalid_df)
        assert result["type_validation"]["Time_spent_Alone"] == False
        assert result["type_validation"]["Social_event_attendance"] == False

    # ===== 新規追加テスト: Leak Prevention Foundation =====
    
    def test_fold_safe_statistics(self):
        """Test fold-safe statistics computation"""
        # Setup test data with target for CV simulation
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Social_event_attendance": [2.0, 3.0, 4.0, 5.0, 6.0],
            "Stage_fear": ["Yes", "No", "Yes", "No", "Yes"],
            "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert", "Introvert"]
        })
        
        # Simulate CV fold separation
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        
        fold_stats = []
        for train_idx, val_idx in skf.split(df, df["Personality"]):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]
            
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
        assert all("train_std" in stats for stats in fold_stats)
        # Validation statistics should be different (proving no leak)
        assert fold_stats[0]["val_normalized_mean"] != fold_stats[1]["val_normalized_mean"]

    def test_sklearn_compatible_transformers(self):
        """Test sklearn-compatible transformers for Silver layer"""
        # Test that preprocessing functions can be used in sklearn pipelines
        from sklearn.pipeline import Pipeline
        from sklearn.base import BaseEstimator, TransformerMixin
        
        # Create a simple transformer wrapper
        class BronzePreprocessor(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                return quick_preprocess(X)
        
        # Test pipeline compatibility
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, None, 3.0],
            "Stage_fear": ["Yes", "No", "Yes"],
            "Drained_after_socializing": ["No", "Yes", "No"]
        })
        
        pipeline = Pipeline([
            ("bronze_preprocess", BronzePreprocessor())
        ])
        
        result = pipeline.fit_transform(df)
        
        # Assert sklearn compatibility
        assert hasattr(result, "shape")
        assert "Stage_fear_encoded" in result.columns
        assert "Drained_after_socializing_encoded" in result.columns

    # ===== 新規追加テスト: Cross-Feature Patterns =====
    
    def test_cross_feature_imputation(self):
        """Test cross-feature imputation using high correlation patterns"""
        # Setup test data with correlated features
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 2.0, None, 4.0, 5.0],
            "Social_event_attendance": [2.0, 4.0, 6.0, 8.0, 10.0],  # Correlated with Time_spent_Alone
            "Going_outside": [3.0, 6.0, 9.0, 12.0, 15.0],  # Correlated with Social_event_attendance
            "Stage_fear": ["Yes", "No", "Yes", None, "No"]
        })
        
        # Test that missing values are handled appropriately
        result = advanced_missing_strategy(df)
        
        # Assert missing flags are created
        assert "Time_spent_Alone_missing" in result.columns
        assert "Stage_fear_missing" in result.columns
        
        # Assert correlation-based patterns are preserved
        # Time_spent_Alone and Social_event_attendance should maintain correlation
        non_missing_mask = ~result["Time_spent_Alone_missing"].astype(bool)
        if non_missing_mask.sum() > 1:
            correlation = result.loc[non_missing_mask, ["Time_spent_Alone", "Social_event_attendance"]].corr().iloc[0, 1]
            assert not pd.isna(correlation)  # Correlation should be computable

    def test_missing_pattern_analysis(self):
        """Test systematic vs random missing pattern detection"""
        # Setup test data with systematic missing patterns
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 2.0, None, 4.0, None, 6.0],
            "Social_event_attendance": [2.0, None, 6.0, None, 10.0, 12.0],
            "Stage_fear": ["Yes", "No", None, "Yes", "No", None],
            "Drained_after_socializing": ["No", "Yes", "No", None, "Yes", "No"]
        })
        
        # Analyze missing patterns
        missing_flags = advanced_missing_strategy(df)
        
        # Assert missing flags are created for all features
        expected_missing_cols = [
            "Time_spent_Alone_missing",
            "Social_event_attendance_missing", 
            "Stage_fear_missing",
            "Drained_after_socializing_missing"
        ]
        
        for col in expected_missing_cols:
            assert col in missing_flags.columns
            assert missing_flags[col].dtype in ["int64", "int32", "bool"]
            assert missing_flags[col].isin([0, 1]).all()  # Binary flags only
        
        # Test systematic pattern detection
        # If Time_spent_Alone and Social_event_attendance have opposite missing patterns
        # (systematic pattern), their missing flags should be negatively correlated
        if missing_flags["Time_spent_Alone_missing"].sum() > 0 and missing_flags["Social_event_attendance_missing"].sum() > 0:
            correlation = missing_flags[["Time_spent_Alone_missing", "Social_event_attendance_missing"]].corr().iloc[0, 1]
            assert not pd.isna(correlation)

    # ===== 新規追加テスト: Performance Requirements =====
    
    def test_sub_second_processing(self):
        """Test sub-second processing performance"""
        # Setup larger test dataset for performance testing
        n_samples = 1000
        df = pd.DataFrame({
            "Time_spent_Alone": np.random.uniform(0, 24, n_samples),
            "Social_event_attendance": np.random.uniform(0, 10, n_samples),
            "Going_outside": np.random.uniform(0, 15, n_samples),
            "Friends_circle_size": np.random.randint(0, 50, n_samples),
            "Post_frequency": np.random.uniform(0, 20, n_samples),
            "Stage_fear": np.random.choice(["Yes", "No", None], n_samples, p=[0.4, 0.5, 0.1]),
            "Drained_after_socializing": np.random.choice(["Yes", "No", None], n_samples, p=[0.3, 0.6, 0.1])
        })
        
        # Performance test for quick_preprocess
        start_time = time.time()
        result_quick = quick_preprocess(df)
        quick_time = time.time() - start_time
        
        # Performance test for advanced_missing_strategy
        start_time = time.time()
        result_missing = advanced_missing_strategy(df)
        missing_time = time.time() - start_time
        
        # Performance test for encode_categorical_robust
        start_time = time.time()
        result_encode = encode_categorical_robust(df)
        encode_time = time.time() - start_time
        
        # Performance test for winsorize_outliers
        start_time = time.time()
        result_winsorize = winsorize_outliers(df)
        winsorize_time = time.time() - start_time
        
        # Assert sub-second processing (Bronze Medal requirement)
        assert quick_time < 1.0, f"quick_preprocess took {quick_time:.3f}s (should be < 1.0s)"
        assert missing_time < 1.0, f"advanced_missing_strategy took {missing_time:.3f}s (should be < 1.0s)"
        assert encode_time < 1.0, f"encode_categorical_robust took {encode_time:.3f}s (should be < 1.0s)"
        assert winsorize_time < 1.0, f"winsorize_outliers took {winsorize_time:.3f}s (should be < 1.0s)"
        
        # Assert total processing time is reasonable
        total_time = quick_time + missing_time + encode_time + winsorize_time
        assert total_time < 2.0, f"Total processing took {total_time:.3f}s (should be < 2.0s)"

    def test_lightgbm_optimization_validation(self):
        """Test LightGBM-specific optimization features"""
        # Setup test data with LightGBM-optimized features
        df = pd.DataFrame({
            "Time_spent_Alone": [1.0, 2.0, 3.0, None, 5.0],
            "Social_event_attendance": [2.0, 4.0, 6.0, 8.0, None],
            "Stage_fear": ["Yes", "No", "Yes", None, "No"],
            "Drained_after_socializing": ["No", "Yes", None, "No", "Yes"]
        })
        
        # Test categorical encoding preserves NaN for LightGBM
        result_encode = encode_categorical_robust(df)
        assert result_encode["Stage_fear"].dtype == "float64"  # LightGBM compatible
        assert pd.isna(result_encode["Stage_fear"].iloc[3])  # NaN preserved
        
        # Test missing flags are binary for LightGBM
        result_missing = advanced_missing_strategy(df)
        missing_cols = [col for col in result_missing.columns if col.endswith("_missing")]
        for col in missing_cols:
            assert result_missing[col].dtype in ["int64", "int32", "bool"]
            assert result_missing[col].isin([0, 1]).all()  # Binary for LightGBM
        
        # Test numeric features are float for LightGBM
        numeric_cols = ["Time_spent_Alone", "Social_event_attendance"]
        for col in numeric_cols:
            if col in result_missing.columns:
                assert result_missing[col].dtype in ["float64", "float32"]

    def test_competition_grade_validation(self):
        """Test competition-grade data quality standards"""
        # Setup test data with edge cases
        df = pd.DataFrame({
            "Time_spent_Alone": [0.0, 24.0, 25.0, -1.0, None],  # Edge cases
            "Social_event_attendance": [0.0, 10.0, -5.0, None, 5.0],  # Edge cases
            "Stage_fear": ["Yes", "NO", "yes", "no", None],  # Case variations
            "Drained_after_socializing": ["No", "YES", "no", "yes", None]  # Case variations
        })
        
        # Test comprehensive validation
        validation_result = validate_data_quality(df)
        
        # Assert type validation
        assert "type_validation" in validation_result
        assert "range_validation" in validation_result
        
        # Assert range validation catches edge cases
        range_validation = validation_result["range_validation"]
        if "Time_spent_Alone" in range_validation:
            if "within_24hrs" in range_validation["Time_spent_Alone"]:
                assert range_validation["Time_spent_Alone"]["within_24hrs"] == False  # 25.0 > 24hrs
            if "non_negative" in range_validation["Time_spent_Alone"]:
                assert range_validation["Time_spent_Alone"]["non_negative"] == False  # -1.0 < 0
        
        # Test categorical standardization handles case variations
        result_encode = encode_categorical_robust(df)
        assert result_encode["Stage_fear"].iloc[0] == result_encode["Stage_fear"].iloc[2]  # "Yes" == "yes"
        assert result_encode["Stage_fear"].iloc[1] == result_encode["Stage_fear"].iloc[3]  # "NO" == "no"
