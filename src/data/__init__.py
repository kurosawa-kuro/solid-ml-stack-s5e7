"""
段階的データ管理パターン統合インターフェース
Bronze/Silver/Gold レベルを選択可能
"""
from typing import Tuple, Dict, Any
import pandas as pd

from .bronze import load_data as bronze_load, quick_preprocess, basic_features
from .silver import DataPipeline, FeatureStore
from .gold import DataManager


def get_data_loader(level: str = "bronze", **kwargs):
    """レベル選択可能なデータローダー
    
    Args:
        level: "bronze", "silver", "gold" のいずれか
        **kwargs: 各レベル固有の引数
    
    Returns:
        選択されたレベルのデータローダー
    """
    
    if level == "bronze":
        return {
            'load': bronze_load,
            'preprocess': quick_preprocess,
            'features': basic_features
        }
    elif level == "silver":
        return DataPipeline(**kwargs)
    elif level == "gold":
        return DataManager(**kwargs)
    else:
        raise ValueError(f"Unsupported level: {level}. Choose from 'bronze', 'silver', 'gold'")


def quick_start(level: str = "bronze", **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """レベル別クイックスタート
    
    Args:
        level: "bronze", "silver", "gold" のいずれか
        **kwargs: 各レベル固有の引数
    
    Returns:
        (train_df, test_df): 前処理済みデータフレーム
    """
    
    if level == "bronze":
        # Bronze: 最速開始
        train, test = bronze_load()
        train = quick_preprocess(train)
        train = basic_features(train)
        test = quick_preprocess(test)
        test = basic_features(test)
        return train, test
    
    elif level == "silver":
        # Silver: 構造化パイプライン
        config = kwargs.get('config', {})
        db_path = kwargs.get('db_path', '/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb')
        
        pipeline = DataPipeline(db_path, config)
        try:
            train, test = pipeline.load_raw()
            train = pipeline.preprocess(train)
            train = pipeline.engineer_features(train)
            test = pipeline.preprocess(test) 
            test = pipeline.engineer_features(test)
            return train, test
        finally:
            pipeline.close()
    
    elif level == "gold":
        # Gold: フル機能
        config_path = kwargs.get('config_path')
        version = kwargs.get('version', 'latest')
        features = kwargs.get('features', ['basic'])
        
        manager = DataManager(config_path)
        try:
            return manager.get_data(version=version, features=features)
        finally:
            manager.close()
    
    else:
        raise ValueError(f"Unsupported level: {level}. Choose from 'bronze', 'silver', 'gold'")


# 便利なエイリアス
Bronze = bronze_load
Silver = DataPipeline
Gold = DataManager

__all__ = [
    'get_data_loader',
    'quick_start', 
    'Bronze',
    'Silver',
    'Gold',
    'bronze_load',
    'quick_preprocess',
    'basic_features',
    'DataPipeline',
    'FeatureStore',
    'DataManager'
]