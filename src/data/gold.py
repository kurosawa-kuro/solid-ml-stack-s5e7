"""
Gold Level Data Management
Model-Ready Data for ML Training
"""

# type: ignore

from typing import Any, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"


def prepare_model_data(df: pd.DataFrame, target_col: str = None, feature_cols: List[str] = None) -> pd.DataFrame:
    """モデル学習用データ準備"""
    df = df.copy()

    # 特徴量選択
    if feature_cols is None:
        # デフォルト特徴量セット
        feature_cols = [
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
        ]

    # 使用可能な特徴量のみ選択
    available_features = [col for col in feature_cols if col in df.columns]

    # 基本カラム追加
    model_cols = ["id"] if "id" in df.columns else []
    if target_col and target_col in df.columns:
        model_cols.append(target_col)
        # エンコードされたターゲットも追加
        encoded_target = f"{target_col}_encoded"
        if encoded_target in df.columns:
            model_cols.append(encoded_target)
    model_cols.extend(available_features)

    return df[model_cols]


def encode_target(df: pd.DataFrame, target_col: str = "Personality") -> pd.DataFrame:
    """ターゲット変数エンコーディング"""
    df = df.copy()
    if target_col in df.columns:
        df[f"{target_col}_encoded"] = (df[target_col] == "Extrovert").astype(int)
    return df


def create_gold_tables() -> None:
    """gold層テーブルをDuckDBに作成"""
    conn = duckdb.connect(DB_PATH)

    # goldスキーマ作成
    conn.execute("CREATE SCHEMA IF NOT EXISTS gold")

    # silverデータ読み込み
    try:
        train_silver = conn.execute("SELECT * FROM silver.train").df()
        test_silver = conn.execute("SELECT * FROM silver.test").df()
    except Exception:
        print("Silver tables not found. Creating silver tables first...")
        from .silver import create_silver_tables

        create_silver_tables()
        train_silver = conn.execute("SELECT * FROM silver.train").df()
        test_silver = conn.execute("SELECT * FROM silver.test").df()

    # ターゲットエンコーディング
    train_gold = encode_target(train_silver)
    test_gold = test_silver.copy()

    # モデル用データ準備（エンコード後に特徴量選択）
    train_gold = prepare_model_data(train_gold, target_col="Personality")
    test_gold = prepare_model_data(test_gold)

    # goldテーブル作成・挿入
    conn.execute("DROP TABLE IF EXISTS gold.train")
    conn.execute("DROP TABLE IF EXISTS gold.test")

    conn.register("train_gold_df", train_gold)
    conn.register("test_gold_df", test_gold)

    conn.execute("CREATE TABLE gold.train AS SELECT * FROM train_gold_df")
    conn.execute("CREATE TABLE gold.test AS SELECT * FROM test_gold_df")

    print("Gold tables created:")
    print(f"- gold.train: {len(train_gold)} rows, {len(train_gold.columns)} columns")
    print(f"- gold.test: {len(test_gold)} rows, {len(test_gold.columns)} columns")

    conn.close()


def load_gold_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """gold層データ読み込み"""
    conn = duckdb.connect(DB_PATH)
    train = conn.execute("SELECT * FROM gold.train").df()
    test = conn.execute("SELECT * FROM gold.test").df()
    conn.close()
    return train, test


def get_ml_ready_data(scale_features: bool = False) -> Tuple[Any, Optional[Any], Any, Optional[Any]]:
    """機械学習用データ準備（X, y分割）"""
    train, test = load_gold_data()

    # 特徴量とターゲット分離
    feature_cols = [col for col in train.columns if col not in ["id", "Personality", "Personality_encoded"]]

    X_train = train[feature_cols].values
    y_train = train["Personality_encoded"].values if "Personality_encoded" in train.columns else None
    X_test = test[feature_cols].values
    test_ids = test["id"].values if "id" in test.columns else None

    # スケーリング
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, test_ids


def create_submission(predictions: np.ndarray, filename: str = "submission.csv") -> None:
    """提出ファイル作成"""
    _, test = load_gold_data()

    submission = pd.DataFrame(
        {"id": test["id"], "Personality": ["Extrovert" if pred == 1 else "Introvert" for pred in predictions]}
    )

    submission.to_csv(filename, index=False)
    print(f"Submission file created: {filename}")
    print(f"Predictions: {submission['Personality'].value_counts()}")


def get_feature_names() -> List[str]:
    """使用特徴量名取得"""
    train, _ = load_gold_data()
    return [col for col in train.columns if col not in ["id", "Personality", "Personality_encoded"]]
