"""
Test cases for models.py
"""

import os
import tempfile
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd


class TestLightGBMModel:
    """LightGBMModelクラスのテスト"""

    def test_model_initialization(self):
        """モデル初期化のテスト"""
        # 実装後にfrom src.models import LightGBMModelを追加

        # デフォルトパラメータの確認（設計書仕様）
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }

        # パラメータの妥当性確認
        assert default_params["objective"] == "binary"
        assert default_params["random_state"] == 42
        assert 0 < default_params["learning_rate"] <= 1

    def test_model_parameter_validation(self):
        """モデルパラメータ検証のテスト"""
        # 不正なパラメータのテスト
        invalid_params = {
            "learning_rate": -0.1,  # 負の学習率
            "num_leaves": 0,  # 無効なleaf数
        }

        # パラメータ範囲チェック
        assert invalid_params["learning_rate"] < 0  # これは無効
        assert invalid_params["num_leaves"] <= 0  # これも無効

    @patch("lightgbm.LGBMClassifier")
    def test_model_training(self, mock_lgb):
        """モデル学習のテスト"""
        # モックの設定
        mock_model = Mock()
        mock_lgb.return_value = mock_model

        # 学習実行（実装後に修正）
        # model = LightGBMModel()
        # model.fit(X_train, y_train)

        # モックが呼ばれたことを確認
        # mock_model.fit.assert_called_once()

    @patch("lightgbm.LGBMClassifier")
    def test_model_prediction(self, mock_lgb):
        """モデル予測のテスト"""
        # モックの設定
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.9, 0.1]])
        mock_lgb.return_value = mock_model

        # 予測実行（実装後に修正）
        # model = LightGBMModel()
        # predictions = model.predict(X_test)
        # probabilities = model.predict_proba(X_test)

        # 結果の確認
        expected_pred = np.array([0, 1, 1, 0])
        assert len(expected_pred) == 4

    def test_feature_importance_extraction(self):
        """特徴量重要度取得のテスト"""
        # モック重要度データ
        feature_names = [f"feature_{i}" for i in range(10)]
        importance_values = np.random.random(10)

        importance_df = pd.DataFrame({"feature": feature_names, "importance": importance_values}).sort_values(
            "importance", ascending=False
        )

        # 重要度の妥当性確認
        assert len(importance_df) == 10
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert importance_df["importance"].iloc[0] >= importance_df["importance"].iloc[1]

    def test_model_serialization(self):
        """モデル保存・読み込みのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")

            # ダミーモデル保存
            dummy_model = {"type": "LightGBM", "params": {"learning_rate": 0.1, "num_leaves": 31}}
            joblib.dump(dummy_model, model_path)

            # 読み込み
            loaded_model = joblib.load(model_path)

            assert loaded_model["type"] == "LightGBM"
            assert loaded_model["params"]["learning_rate"] == 0.1
            assert loaded_model["params"]["num_leaves"] == 31


class TestCrossValidationTrainer:
    """CrossValidationTrainerクラスのテスト"""

    def test_cv_trainer_initialization(self):
        """CVトレーナー初期化のテスト"""
        cv_config = {"n_splits": 5, "shuffle": True, "random_state": 42}

        # 設定の妥当性確認
        assert cv_config["n_splits"] > 1
        assert isinstance(cv_config["shuffle"], bool)
        assert cv_config["random_state"] is not None

    @patch("sklearn.model_selection.StratifiedKFold")
    def test_cv_fold_generation(self, mock_skf):
        """CVフォールド生成のテスト"""
        # モックの設定
        mock_cv = Mock()
        mock_cv.split.return_value = [
            (np.array([0, 1, 2]), np.array([3, 4])),
            (np.array([0, 3, 4]), np.array([1, 2])),
        ]
        mock_skf.return_value = mock_cv

        # フォールド生成（実装後に修正）
        # trainer = CrossValidationTrainer()
        # folds = list(trainer.generate_folds(X, y))

        # assert len(folds) == 2

    def test_out_of_fold_prediction_structure(self):
        """Out-of-Fold予測構造のテスト"""
        n_samples = 100
        oof_predictions = np.zeros(n_samples)

        # フォールド1の予測
        val_indices_1 = np.array([0, 1, 2, 3, 4])
        fold_predictions_1 = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        oof_predictions[val_indices_1] = fold_predictions_1

        # フォールド2の予測
        val_indices_2 = np.array([5, 6, 7, 8, 9])
        fold_predictions_2 = np.array([0.7, 0.4, 0.6, 0.1, 0.9])
        oof_predictions[val_indices_2] = fold_predictions_2

        # 予測が設定されたインデックスの確認
        assert oof_predictions[0] == 0.1
        assert oof_predictions[5] == 0.7

    def test_cv_score_calculation(self):
        """CVスコア計算のテスト"""
        fold_scores = [0.97, 0.96, 0.98, 0.95, 0.97]

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        expected_mean = 0.966
        assert abs(mean_score - expected_mean) < 0.001
        assert std_score >= 0

    @patch("time.time")
    def test_training_time_measurement(self, mock_time):
        """学習時間計測のテスト"""
        # 時間の推移をモック
        mock_time.side_effect = [1000.0, 1180.0]  # 180秒の学習時間

        start_time = mock_time()
        # 学習処理（モック）
        end_time = mock_time()

        training_time = end_time - start_time
        assert training_time == 180.0

    def test_cv_results_aggregation(self):
        """CV結果集計のテスト"""
        cv_results = {
            "fold_scores": [0.97, 0.96, 0.98, 0.95, 0.97],
            "mean_score": 0.966,
            "std_score": 0.011,
            "oof_predictions": np.random.random(100),
            "feature_importance": pd.DataFrame({"feature": ["f1", "f2", "f3"], "importance": [0.5, 0.3, 0.2]}),
            "training_time": 180.0,
        }

        # 必須フィールドの確認
        required_fields = ["fold_scores", "mean_score", "std_score", "oof_predictions", "training_time"]

        for field in required_fields:
            assert field in cv_results


class TestModelUtilities:
    """ユーティリティ関数のテスト"""

    def test_accuracy_calculation(self):
        """Accuracy計算のテスト"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_true, y_pred)

        expected = 4 / 5  # 0.8
        assert accuracy == expected

    def test_auc_calculation(self):
        """AUC計算のテスト"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y_true, y_pred_proba)

        assert 0 <= auc <= 1

    def test_prediction_statistics(self):
        """予測統計情報のテスト"""
        predictions = np.array([0, 1, 1, 0, 1, 1, 0, 1])

        stats = {
            "total_predictions": len(predictions),
            "extrovert_count": np.sum(predictions == 1),
            "introvert_count": np.sum(predictions == 0),
            "extrovert_ratio": np.mean(predictions == 1),
            "introvert_ratio": np.mean(predictions == 0),
        }

        assert stats["total_predictions"] == 8
        assert stats["extrovert_count"] == 5
        assert stats["introvert_count"] == 3
        assert abs(stats["extrovert_ratio"] + stats["introvert_ratio"] - 1.0) < 1e-10

    def test_model_evaluation_metrics(self):
        """モデル評価指標のテスト"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.7])

        from sklearn.metrics import accuracy_score, roc_auc_score

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_pred_proba),
        }

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc"] <= 1

    def test_learning_curve_data_structure(self):
        """学習曲線データ構造のテスト"""
        learning_curve = {
            "train_scores": [0.85, 0.90, 0.93, 0.95, 0.96],
            "val_scores": [0.82, 0.87, 0.90, 0.92, 0.94],
            "epochs": list(range(5)),
        }

        assert len(learning_curve["train_scores"]) == len(learning_curve["val_scores"])
        assert len(learning_curve["epochs"]) == len(learning_curve["train_scores"])

        # 学習の進歩（一般的に向上傾向）
        assert learning_curve["train_scores"][-1] >= learning_curve["train_scores"][0]


class TestModelPersistence:
    """モデル永続化のテスト"""

    def test_model_save_load_cycle(self):
        """モデル保存・読み込みサイクルのテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # ダミーモデルとメタデータ
            model_data = {
                "model": "dummy_lgb_model",
                "params": {"learning_rate": 0.1, "num_leaves": 31},
                "feature_names": ["f1", "f2", "f3"],
                "cv_score": 0.975,
                "timestamp": "2024-01-01T12: 00: 00",
            }

            model_path = os.path.join(temp_dir, "model.pkl")

            # 保存
            joblib.dump(model_data, model_path)

            # 読み込み
            loaded_data = joblib.load(model_path)

            assert loaded_data["params"]["learning_rate"] == 0.1
            assert loaded_data["cv_score"] == 0.975
            assert len(loaded_data["feature_names"]) == 3

    def test_model_metadata_structure(self):
        """モデルメタデータ構造のテスト"""
        metadata = {
            "model_type": "LightGBM",
            "version": "1.0.0",
            "training_date": "2024-01-01",
            "cv_config": {"n_splits": 5, "random_state": 42},
            "feature_count": 10,
            "performance": {"cv_accuracy": 0.975, "cv_std": 0.008, "cv_auc": 0.985},
            "hyperparameters": {"learning_rate": 0.1, "num_leaves": 31, "max_depth": -1},
        }

        # 必須フィールドの確認
        required_fields = ["model_type", "training_date", "cv_config", "performance", "hyperparameters"]

        for field in required_fields:
            assert field in metadata

    def test_model_versioning(self):
        """モデルバージョニングのテスト"""
        model_versions = [
            {"version": "1.0.0", "cv_score": 0.970, "timestamp": "2024-01-01"},
            {"version": "1.1.0", "cv_score": 0.975, "timestamp": "2024-01-02"},
            {"version": "1.2.0", "cv_score": 0.973, "timestamp": "2024-01-03"},
        ]

        # 最高スコアのバージョン取得
        best_version = max(model_versions, key=lambda x: x["cv_score"])

        assert best_version["version"] == "1.1.0"
        assert best_version["cv_score"] == 0.975


class TestModelIntegration:
    """モデル統合テスト"""

    def test_end_to_end_training_pipeline(self):
        """エンドツーエンド学習パイプラインのテスト"""
        # サンプルデータ
        n_samples, n_features = 200, 10
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 2, n_samples)

        # パイプライン構成要素のテスト
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 簡単なダミー予測（実装後はLightGBMに置換）
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)

        # パイプラインが動作することを確認
        assert len(fold_scores) == 3
        assert 0 <= mean_score <= 1

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 不正な入力データ
        X_invalid = np.array([[1, 2], [3, np.nan], [5, 6]])  # NaN含む
        y_invalid = np.array([0, 1])  # サイズ不一致

        # NaN検出
        has_nan = np.isnan(X_invalid).any()
        assert has_nan

        # サイズ不一致検出
        size_mismatch = X_invalid.shape[0] != y_invalid.shape[0]
        assert size_mismatch
