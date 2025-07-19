"""
Gold Level Data Management
Kaggle精度向上に特化した高度データ管理（不要機能削除済み）
"""

from typing import Tuple, Dict, List, Optional
import pandas as pd
import duckdb
import pickle
from pathlib import Path


class DataManager:
    """Kaggle精度向上特化データ管理システム"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.db_path = self.config["database"]["path"]
        self.conn = None
        self.cache_dir = Path(self.config["cache"]["directory"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        """データベース接続"""
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
        return self.conn

    def _default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            "database": {
                "path": "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb",
                "schema": "playground_series_s5e7",
            },
            "cache": {"directory": "/tmp/ml_cache"},
        }

    def get_data(self, features: Optional[List[str]] = None, cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """特徴量指定でデータ取得（キャッシュ付き）"""

        features = features or ["basic"]
        cache_key = f"data_{hash(str(sorted(features)))}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # キャッシュチェック
        if cache and cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # データ処理
        train, test = self._process_data(features)

        # キャッシュ保存
        if cache:
            with open(cache_file, "wb") as f:
                pickle.dump((train, test), f)

        return train, test

    def _process_data(self, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """特徴量ベースのデータ処理"""
        conn = self._connect()
        train = conn.execute("SELECT * FROM playground_series_s5e7.train").df()
        test = conn.execute("SELECT * FROM playground_series_s5e7.test").df()

        # 特徴量エンジニアリング
        if "basic" in features:
            train = self._add_basic_features(train)
            test = self._add_basic_features(test)

        if "advanced" in features:
            train = self._add_advanced_features(train)
            test = self._add_advanced_features(test)

        return train, test

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量追加"""
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

        # カテゴリエンコーディング
        if "Stage_fear" in df.columns:
            df["Stage_fear_encoded"] = (df["Stage_fear"] == "Yes").astype(int)
        if "Drained_after_socializing" in df.columns:
            df["Drained_after_socializing_encoded"] = (df["Drained_after_socializing"] == "Yes").astype(int)

        # 基本特徴量
        if "Social_event_attendance" in df.columns and "Time_spent_Alone" in df.columns:
            df["social_ratio"] = df["Social_event_attendance"] / (df["Time_spent_Alone"] + 1)

        if "Going_outside" in df.columns and "Social_event_attendance" in df.columns:
            df["activity_sum"] = df["Going_outside"] + df["Social_event_attendance"]

        return df

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度特徴量追加"""
        df = df.copy()

        # 交互作用特徴量
        if "Stage_fear_encoded" in df.columns and "Drained_after_socializing_encoded" in df.columns:
            df["fear_drained_interaction"] = df["Stage_fear_encoded"] * df["Drained_after_socializing_encoded"]

        # 統計特徴量
        numeric_cols = [
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
            "Post_frequency",
        ]
        existing_cols = [col for col in numeric_cols if col in df.columns]

        if len(existing_cols) >= 2:
            df["numeric_mean"] = df[existing_cols].mean(axis=1)
            df["numeric_std"] = df[existing_cols].std(axis=1)

        return df

    def clear_cache(self):
        """キャッシュクリア"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def close(self):
        """接続終了"""
        if self.conn:
            self.conn.close()
            self.conn = None
