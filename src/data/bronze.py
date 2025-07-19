"""
Bronze Level Data Management
Raw Data Standardization & Quality Assurance (Entry Point to Medallion Pipeline)
"""

from typing import Tuple, Dict, Any, Optional
import warnings

import duckdb
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Raw data access point - Single source entry to Medallion pipeline"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM playground_series_s5e7.train").df()
    test = conn.execute("SELECT * FROM playground_series_s5e7.test").df()
    conn.close()
    
    # Explicit dtype setting for LightGBM optimization
    train = _set_optimal_dtypes(train)
    test = _set_optimal_dtypes(test)
    
    return train, test


def _set_optimal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set optimal dtypes for LightGBM compatibility and performance"""
    df = df.copy()
    
    # Numeric features - use float32 for memory efficiency when possible
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']
    
    for col in numeric_cols:
        if col in df.columns:
            # Check if values fit in float32 range
            if df[col].dtype in ['float64', 'float32']:
                col_min, col_max = df[col].min(), df[col].max()
                if col_min >= -3.4e38 and col_max <= 3.4e38:
                    df[col] = df[col].astype('float32')
                else:
                    df[col] = df[col].astype('float64')
            elif df[col].dtype in ['int64', 'int32']:
                # Convert integers to float for LightGBM
                df[col] = df[col].astype('float32')
    
    # Categorical features - ensure object type for processing
    categorical_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')
    
    return df


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Type validation and range guards for data quality assurance"""
    validation_results = {
        'type_validation': {},
        'range_validation': {},
        'schema_validation': {},
        'quality_metrics': {}
    }
    
    # Type validation
    expected_numeric = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                       'Friends_circle_size', 'Post_frequency']
    expected_categorical = ['Stage_fear', 'Drained_after_socializing']
    
    for col in expected_numeric:
        if col in df.columns:
            validation_results['type_validation'][col] = pd.api.types.is_numeric_dtype(df[col])
    
    for col in expected_categorical:
        if col in df.columns:
            validation_results['type_validation'][col] = df[col].dtype == 'object'
    
    # Range validation with more comprehensive checks
    if 'Time_spent_Alone' in df.columns:
        validation_results['range_validation']['Time_spent_Alone'] = {
            'within_24hrs': (df['Time_spent_Alone'] <= 24).all(),
            'non_negative': (df['Time_spent_Alone'] >= 0).all(),
            'finite_values': np.isfinite(df['Time_spent_Alone']).all()
        }
    
    for col in ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
        if col in df.columns:
            validation_results['range_validation'][col] = {
                'non_negative': (df[col] >= 0).all(),
                'finite_values': np.isfinite(df[col]).all()
            }
    
    # Quality metrics
    validation_results['quality_metrics'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return validation_results


def encode_categorical_robust(df: pd.DataFrame) -> pd.DataFrame:
    """Yes/No normalization with case-insensitive unified mapping → {0,1}"""
    df = df.copy()
    
    categorical_columns = ['Stage_fear', 'Drained_after_socializing']
    
    for col in categorical_columns:
        if col in df.columns:
            # Handle NaN values first
            df[col] = df[col].fillna('Unknown')
            
            # Case-insensitive Yes/No → 1/0 mapping
            df[col] = df[col].astype(str).str.lower().str.strip()
            
            # More robust mapping
            yes_values = ['yes', 'y', '1', 'true']
            no_values = ['no', 'n', '0', 'false']
            
            # Create mapping dictionary
            mapping_dict = {}
            for yes_val in yes_values:
                mapping_dict[yes_val] = 1.0
            for no_val in no_values:
                mapping_dict[no_val] = 0.0
            
            # Apply mapping, keep NaN for unknown values (LightGBM will handle)
            df[col] = df[col].map(mapping_dict)
            
            # Convert to float64 for LightGBM compatibility
            df[col] = df[col].astype('float64')
            
            # Create encoded column for compatibility
            encoded_col = f"{col}_encoded"
            df[encoded_col] = df[col]
    
    return df


def advanced_missing_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Missing value intelligence with LightGBM native handling"""
    df = df.copy()
    
    # Create missing flags for high-impact features (Winner Solution pattern)
    high_impact_features = ['Stage_fear', 'Going_outside', 'Time_spent_Alone', 
                           'Drained_after_socializing', 'Social_event_attendance']
    
    for col in high_impact_features:
        if col in df.columns:
            missing_flag_col = f"{col}_missing"
            df[missing_flag_col] = df[col].isna().astype('int32')  # LightGBM optimized
    
    # Cross-feature missing pattern analysis
    if 'Stage_fear' in df.columns and 'Drained_after_socializing' in df.columns:
        # Create interaction missing flag
        df['social_fatigue_missing'] = (
            df['Stage_fear'].isna() & df['Drained_after_socializing'].isna()
        ).astype('int32')
    
    # Preserve NaN for LightGBM native handling (don't impute)
    # LightGBM will handle NaN values automatically in tree splits
    
    return df


def winsorize_outliers(df: pd.DataFrame, percentile: float = 0.01) -> pd.DataFrame:
    """IQR-based outlier clipping for numeric stability"""
    df = df.copy()
    
    numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                      'Friends_circle_size', 'Post_frequency']
    
    for col in numeric_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # Skip if too many NaN values
            if df[col].isna().sum() > len(df) * 0.5:
                continue
                
            # Calculate bounds using quantiles
            lower_bound = df[col].quantile(percentile)
            upper_bound = df[col].quantile(1 - percentile)
            
            # Apply clipping
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df





def create_bronze_tables() -> None:
    """Creates standardized bronze.train, bronze.test tables"""
    conn = duckdb.connect(DB_PATH)
    
    # Create bronze schema
    conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    
    # Load raw data
    train_raw, test_raw = load_data()
    
    # Apply bronze layer processing pipeline (modern robust functions)
    train_bronze = encode_categorical_robust(train_raw)
    test_bronze = encode_categorical_robust(test_raw)
    
    train_bronze = advanced_missing_strategy(train_bronze)
    test_bronze = advanced_missing_strategy(test_bronze)
    
    train_bronze = winsorize_outliers(train_bronze)
    test_bronze = winsorize_outliers(test_bronze)
    
    # Validate data quality
    train_validation = validate_data_quality(train_bronze)
    test_validation = validate_data_quality(test_bronze)
    
    # Create bronze tables
    conn.execute("DROP TABLE IF EXISTS bronze.train")
    conn.execute("DROP TABLE IF EXISTS bronze.test")
    
    conn.register("train_bronze_df", train_bronze)
    conn.register("test_bronze_df", test_bronze)
    
    conn.execute("CREATE TABLE bronze.train AS SELECT * FROM train_bronze_df")
    conn.execute("CREATE TABLE bronze.test AS SELECT * FROM test_bronze_df")
    
    print("Bronze tables created:")
    print(f"- bronze.train: {len(train_bronze)} rows, {len(train_bronze.columns)} columns")
    print(f"- bronze.test: {len(test_bronze)} rows, {len(test_bronze.columns)} columns")
    print(f"- Data quality validation: {len([k for k, v in train_validation['type_validation'].items() if v])} types passed")
    print(f"- Bronze layer features: {len(train_bronze.columns)} columns (quality assured)")
    
    conn.close()


def quick_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy preprocessing function - USE create_bronze_tables() for production pipeline"""
    df = df.copy()

    # 欠損値処理
    numeric_cols = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # カテゴリ変換
    if "Stage_fear" in df.columns:
        df["Stage_fear_encoded"] = (df["Stage_fear"] == "Yes").astype(int)
    if "Drained_after_socializing" in df.columns:
        df["Drained_after_socializing_encoded"] = (df["Drained_after_socializing"] == "Yes").astype(int)

    return df


def load_bronze_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """bronze層データ読み込み"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM bronze.train").df()
    test = conn.execute("SELECT * FROM bronze.test").df()
    conn.close()
    return train, test


# ===== Sklearn-Compatible Transformers for Pipeline Integration =====

class BronzePreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for Bronze layer processing"""
    
    def __init__(self, add_features: bool = True, winsorize: bool = True):
        self.add_features = add_features
        self.winsorize = winsorize
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Fit the transformer (no fitting required for Bronze layer)"""
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply Bronze layer transformations"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        # Apply Bronze pipeline
        X_transformed = encode_categorical_robust(X)
        X_transformed = advanced_missing_strategy(X_transformed)
        
        if self.winsorize:
            X_transformed = winsorize_outliers(X_transformed)
        
        if self.add_features:
            X_transformed = basic_features(X_transformed)
        
        return X_transformed


class FoldSafeBronzePreprocessor(BaseEstimator, TransformerMixin):
    """Fold-safe Bronze preprocessor for CV integration"""
    
    def __init__(self):
        self.categorical_mappings = {}
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Learn categorical mappings from training data only"""
        self.categorical_mappings = {}
        
        categorical_columns = ['Stage_fear', 'Drained_after_socializing']
        
        for col in categorical_columns:
            if col in X.columns:
                # Learn unique values from training data
                unique_values = X[col].dropna().unique()
                self.categorical_mappings[col] = set(unique_values)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Apply fold-safe transformations"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Apply fold-safe categorical encoding
        categorical_columns = ['Stage_fear', 'Drained_after_socializing']
        
        for col in categorical_columns:
            if col in X_transformed.columns:
                # Handle NaN values
                X_transformed[col] = X_transformed[col].fillna('Unknown')
                
                # Case-insensitive mapping
                X_transformed[col] = X_transformed[col].astype(str).str.lower().str.strip()
                
                # Apply mapping only for known values
                yes_values = ['yes', 'y', '1', 'true']
                no_values = ['no', 'n', '0', 'false']
                
                mapping_dict = {}
                for yes_val in yes_values:
                    mapping_dict[yes_val] = 1.0
                for no_val in no_values:
                    mapping_dict[no_val] = 0.0
                
                X_transformed[col] = X_transformed[col].map(mapping_dict)
                X_transformed[col] = X_transformed[col].astype('float64')
        
        # Add missing flags
        high_impact_features = ['Stage_fear', 'Going_outside', 'Time_spent_Alone']
        for col in high_impact_features:
            if col in X_transformed.columns:
                missing_flag_col = f"{col}_missing"
                X_transformed[missing_flag_col] = X_transformed[col].isna().astype('int32')
        
        return X_transformed
