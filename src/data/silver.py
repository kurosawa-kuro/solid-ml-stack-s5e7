"""
Silver Level Data Management
Feature Engineering & Advanced Preprocessing
"""

from typing import Tuple
import pandas as pd
import numpy as np
import duckdb

DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"


def advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """高度な特徴量エンジニアリング"""
    df = df.copy()

    # 既存の基本特徴量
    if "Social_event_attendance" in df.columns and "Time_spent_Alone" in df.columns:
        df["social_ratio"] = df["Social_event_attendance"] / (df["Time_spent_Alone"] + 1)

    if "Going_outside" in df.columns and "Social_event_attendance" in df.columns:
        df["activity_sum"] = df["Going_outside"] + df["Social_event_attendance"]

    # 新しい特徴量
    numeric_cols = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]

    # 統計的特徴量
    if all(col in df.columns for col in numeric_cols):
        df["total_activity"] = df[numeric_cols].sum(axis=1)
        df["avg_activity"] = df[numeric_cols].mean(axis=1)
        df["activity_std"] = df[numeric_cols].std(axis=1)

    # 比率特徴量
    if "Friends_circle_size" in df.columns and "Post_frequency" in df.columns:
        df["post_per_friend"] = df["Post_frequency"] / (df["Friends_circle_size"] + 1)

    # 二項交互作用
    if "Stage_fear_encoded" in df.columns and "Drained_after_socializing_encoded" in df.columns:
        df["fear_drain_interaction"] = df["Stage_fear_encoded"] * df["Drained_after_socializing_encoded"]

    # 外向性スコア（仮説ベース）
    extrovert_features = []
    if "Social_event_attendance" in df.columns:
        extrovert_features.append("Social_event_attendance")
    if "Going_outside" in df.columns:
        extrovert_features.append("Going_outside")
    if "Friends_circle_size" in df.columns:
        extrovert_features.append("Friends_circle_size")

    if extrovert_features:
        df["extrovert_score"] = df[extrovert_features].sum(axis=1)

    # 内向性スコア
    if "Time_spent_Alone" in df.columns:
        df["introvert_score"] = df["Time_spent_Alone"]
        if "Stage_fear_encoded" in df.columns:
            df["introvert_score"] += df["Stage_fear_encoded"] * 2
        if "Drained_after_socializing_encoded" in df.columns:
            df["introvert_score"] += df["Drained_after_socializing_encoded"] * 2

    return df


def scaling_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量スケーリング（標準化）"""
    df = df.copy()

    # 数値特徴量を標準化
    numeric_features = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ["id"]  # IDカラムは除外
    numeric_features = [col for col in numeric_features if col not in exclude_cols]

    for col in numeric_features:
        if df[col].std() > 0:  # 分散が0でない場合のみ
            df[f"{col}_scaled"] = (df[col] - df[col].mean()) / df[col].std()

    return df


def create_silver_tables() -> None:
    """silver層テーブルをDuckDBに作成"""
    conn = duckdb.connect(DB_PATH)

    # silverスキーマ作成
    conn.execute("CREATE SCHEMA IF NOT EXISTS silver")

    # bronzeデータ読み込み
    try:
        train_bronze = conn.execute("SELECT * FROM bronze.train").df()
        test_bronze = conn.execute("SELECT * FROM bronze.test").df()
    except Exception:
        print("Bronze tables not found. Creating bronze tables first...")
        from .bronze import create_bronze_tables

        create_bronze_tables()
        train_bronze = conn.execute("SELECT * FROM bronze.train").df()
        test_bronze = conn.execute("SELECT * FROM bronze.test").df()

    # 特徴量エンジニアリング適用
    train_silver = advanced_features(train_bronze)
    test_silver = advanced_features(test_bronze)

    # スケーリング適用
    train_silver = scaling_features(train_silver)
    test_silver = scaling_features(test_silver)

    # silverテーブル作成・挿入
    conn.execute("DROP TABLE IF EXISTS silver.train")
    conn.execute("DROP TABLE IF EXISTS silver.test")

    conn.register("train_silver_df", train_silver)
    conn.register("test_silver_df", test_silver)

    conn.execute("CREATE TABLE silver.train AS SELECT * FROM train_silver_df")
    conn.execute("CREATE TABLE silver.test AS SELECT * FROM test_silver_df")

    print("Silver tables created:")
    print(f"- silver.train: {len(train_silver)} rows, {len(train_silver.columns)} columns")
    print(f"- silver.test: {len(test_silver)} rows, {len(test_silver.columns)} columns")

    conn.close()


def load_silver_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """silver層データ読み込み"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM silver.train").df()
    test = conn.execute("SELECT * FROM silver.test").df()
    conn.close()
    return train, test


def get_feature_importance_order() -> list:
    """特徴量重要度順リスト（経験的順序）"""
    return [
        "extrovert_score",
        "introvert_score",
        "Social_event_attendance",
        "Time_spent_Alone",
        "Drained_after_socializing_encoded",
        "Stage_fear_encoded",
        "social_ratio",
        "Friends_circle_size",
        "Going_outside",
        "Post_frequency",
        "activity_sum",
        "post_per_friend",
        "fear_drain_interaction",
        "total_activity",
        "avg_activity",
    ]
