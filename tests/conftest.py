"""
pytestの共通設定とフィクスチャ定義
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_classification_data():
    """分類用テストデータ生成"""
    n_samples, n_features = 200, 10
    X = np.random.RandomState(42).random((n_samples, n_features))
    y = np.random.RandomState(42).randint(0, 2, n_samples)
    return X, y


@pytest.fixture
def feature_names():
    """標準的な特徴量名生成"""
    return [f"feature_{i:02d}" for i in range(10)]


@pytest.fixture
def sample_train_data():
    """標準的な訓練データ（DataFrame形式）"""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'id': range(n_samples),
        'Time_spent_Alone': np.random.random(n_samples),
        'Social_event_attendance': np.random.random(n_samples),
        'Going_outside': np.random.random(n_samples),
        'Friends_circle_size': np.random.random(n_samples),
        'Post_frequency': np.random.random(n_samples),
        'Stage_fear': np.random.choice(['Yes', 'No'], n_samples),
        'Drained_after_socializing': np.random.choice(['Yes', 'No'], n_samples),
        'Personality': np.random.choice(['Extrovert', 'Introvert'], n_samples),
        'Personality_encoded': np.random.randint(0, 2, n_samples)
    })
    
    return data


@pytest.fixture
def sample_test_data():
    """標準的なテストデータ（DataFrame形式）"""
    np.random.seed(43)
    n_samples = 50
    
    data = pd.DataFrame({
        'id': range(100, 100 + n_samples),
        'Time_spent_Alone': np.random.random(n_samples),
        'Social_event_attendance': np.random.random(n_samples),
        'Going_outside': np.random.random(n_samples),
        'Friends_circle_size': np.random.random(n_samples),
        'Post_frequency': np.random.random(n_samples),
        'Stage_fear': np.random.choice(['Yes', 'No'], n_samples),
        'Drained_after_socializing': np.random.choice(['Yes', 'No'], n_samples)
    })
    
    return data


@pytest.fixture
def cv_results_mock():
    """標準的なCV結果のモック（推奨: MockHelpers.create_sample_cv_results()を使用）"""
    return MockHelpers.create_sample_cv_results()


class MockHelpers:
    """モックオブジェクト生成ヘルパー"""
    
    @staticmethod
    def create_duckdb_mock(train_data, test_data):
        """DuckDB接続の標準モック"""
        mock_conn = Mock()
        
        # execute()がDataFrameを返すようなモック設定
        mock_train_result = Mock()
        mock_train_result.df.return_value = train_data
        
        mock_test_result = Mock()
        mock_test_result.df.return_value = test_data
        
        # side_effectで順番に返す
        mock_conn.execute.side_effect = [mock_train_result, mock_test_result]
        
        return mock_conn
    
    @staticmethod
    def create_lightgbm_mock(cv_score=0.975):
        """LightGBMモデルの標準モック"""
        mock_model = Mock()
        
        # 予測メソッド
        mock_model.predict.return_value = np.array([0, 1, 1, 0])
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.9, 0.1]
        ])
        
        # 特徴量重要度
        mock_model.feature_importances_ = np.random.random(10)
        
        return mock_model
    
    @staticmethod
    def create_cv_trainer_mock(cv_results):
        """CrossValidationTrainerの標準モック"""
        mock_trainer = Mock()
        mock_trainer.train_cv.return_value = cv_results
        return mock_trainer
    
    @staticmethod
    def create_simple_model_mock():
        """簡素なモデルモック（実装不要のテスト用）"""
        mock_model = Mock()
        mock_model.is_fitted = False
        mock_model.params = {"learning_rate": 0.1, "num_leaves": 31}
        mock_model.feature_names = None
        return mock_model
    
    @staticmethod
    def create_sample_cv_results():
        """サンプルCV結果データ"""
        return {
            "fold_scores": [0.97, 0.96, 0.98, 0.95, 0.97],
            "mean_score": 0.966,
            "std_score": 0.011,
            "oof_predictions": np.random.RandomState(42).random(100),
            "feature_importance": pd.DataFrame({
                "feature": [f"feature_{i}" for i in range(5)],
                "importance": np.random.RandomState(42).random(5)
            }),
            "training_time": 180.0,
            "models": []
        }
    
    @staticmethod
    def skip_implementation_test(reason="Implementation not ready"):
        """実装未完了のテストをスキップ"""
        import pytest
        return pytest.mark.skip(reason=reason)


# テスト用の定数
TEST_DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"
TEST_SCHEMA = "playground_series_s5e7"
BRONZE_MEDAL_THRESHOLD = 0.976518
TARGET_COLUMN = "Personality_encoded"
FEATURE_COLUMNS = [
    "Time_spent_Alone", 
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size", 
    "Post_frequency",
    "Stage_fear",
    "Drained_after_socializing"
]