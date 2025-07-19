"""
Bronze Level Data Management
シンプル・直接的・高速プロトタイピング用
"""
from typing import Tuple
import pandas as pd
import duckdb

DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """DuckDBから直接データ読み込み"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM playground_series_s5e7.train").df()
    test = conn.execute("SELECT * FROM playground_series_s5e7.test").df()
    conn.close()
    return train, test


def quick_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """最小限の前処理（インライン）"""
    df = df.copy()
    
    # 欠損値処理
    numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                   'Friends_circle_size', 'Post_frequency']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # カテゴリ変換
    if 'Stage_fear' in df.columns:
        df['Stage_fear_encoded'] = (df['Stage_fear'] == 'Yes').astype(int)
    if 'Drained_after_socializing' in df.columns:
        df['Drained_after_socializing_encoded'] = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    return df


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """基本的な特徴量生成"""
    df = df.copy()
    
    # 比率特徴量
    if 'Social_event_attendance' in df.columns and 'Time_spent_Alone' in df.columns:
        df['social_ratio'] = df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1)
    
    # 合計特徴量
    if 'Going_outside' in df.columns and 'Social_event_attendance' in df.columns:
        df['activity_sum'] = df['Going_outside'] + df['Social_event_attendance']
    
    return df