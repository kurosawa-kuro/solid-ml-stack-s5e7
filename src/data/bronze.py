"""
Bronze Level Data Management
Raw Data Standardization & Quality Assurance (Entry Point to Medallion Pipeline)
"""

from typing import Tuple, Dict, Any

import duckdb
import pandas as pd
import numpy as np

DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Raw data access point - Single source entry to Medallion pipeline"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM playground_series_s5e7.train").df()
    test = conn.execute("SELECT * FROM playground_series_s5e7.test").df()
    conn.close()
    return train, test


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Type validation and range guards for data quality assurance"""
    validation_results = {
        'type_validation': {},
        'range_validation': {},
        'schema_validation': {}
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
    
    # Range validation
    if 'Time_spent_Alone' in df.columns:
        validation_results['range_validation']['Time_spent_Alone'] = {
            'within_24hrs': (df['Time_spent_Alone'] <= 24).all(),
            'non_negative': (df['Time_spent_Alone'] >= 0).all()
        }
    
    for col in ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
        if col in df.columns:
            validation_results['range_validation'][col] = {
                'non_negative': (df[col] >= 0).all()
            }
    
    return validation_results


def encode_categorical_robust(df: pd.DataFrame) -> pd.DataFrame:
    """Yes/No normalization with case-insensitive unified mapping → {0,1}"""
    df = df.copy()
    
    categorical_columns = ['Stage_fear', 'Drained_after_socializing']
    
    for col in categorical_columns:
        if col in df.columns:
            # Case-insensitive Yes/No → 1/0 mapping
            df[col] = df[col].astype(str).str.lower()
            df[col] = df[col].map({'yes': 1, 'no': 0})
            df[col] = df[col].astype('float64')  # LightGBM-friendly type
    
    return df


def advanced_missing_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Missing value intelligence with LightGBM native handling"""
    df = df.copy()
    
    # Create missing flags for high-impact features
    high_impact_features = ['Stage_fear', 'Going_outside', 'Time_spent_Alone']
    
    for col in high_impact_features:
        if col in df.columns:
            missing_flag_col = f"{col}_missing"
            df[missing_flag_col] = df[col].isna().astype(int)
    
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
            lower_bound = df[col].quantile(percentile)
            upper_bound = df[col].quantile(1 - percentile)
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


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """基本的な特徴量生成"""
    df = df.copy()

    # 比率特徴量
    if "Social_event_attendance" in df.columns and "Time_spent_Alone" in df.columns:
        df["social_ratio"] = df["Social_event_attendance"] / (df["Time_spent_Alone"] + 1)

    # 合計特徴量
    if "Going_outside" in df.columns and "Social_event_attendance" in df.columns:
        df["activity_sum"] = df["Going_outside"] + df["Social_event_attendance"]

    return df




def load_bronze_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """bronze層データ読み込み"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM bronze.train").df()
    test = conn.execute("SELECT * FROM bronze.test").df()
    conn.close()
    return train, test
