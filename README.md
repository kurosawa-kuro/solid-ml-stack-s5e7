# solid-ml-stack-s5e7# Solid ML Stack

Kaggle S5E7 性格予測コンペティション用の高速でスケーラブルな機械学習パイプライン

## 特徴

  2. 高性能モデルの実行:
    - make model-lgb (DARTモード)
    - make model-cat
    - make ensemble-average
  3. 最高性能を目指す: python3 scripts/enhanced_ensemble_workflow.py

### 🎯 Kaggle コンペティション最適化
- **エンドツーエンドワークフロー最適化**: 前処理 → 特徴量エンジニアリング → モデル学習 → アンサンブル → 提出の自動化パイプライン
- **再利用可能な設計**: 関数・クラスベースのアーキテクチャで異なるコンペティションにも容易に適用可能
- **CPU特化**: ツリーベースモデル（XGBoost/LightGBM/CatBoost）による高速学習

### 🔧 モジュール化設計
- **前処理**: 欠損値処理、外れ値除去、スケーリング、エンコーディング
- **特徴量エンジニアリング**: 数値変換、カテゴリエンコーディング、交互作用特徴量、日時特徴量
- **モデル学習**: XGBoost、LightGBM、CatBoost、線形モデル
- **パラメータ最適化**: Grid Search、Random Search、Bayesian Optimization、Optuna
- **アンサンブル手法**: 平均化、重み付き平均、スタッキング、投票

### 📊 タブラーデータ特化
- CSV等の構造化データ形式に特化
- pandas → scikit-learn パイプラインベース
- 画像・テキスト・時系列深層学習は対象外

## インストール

```bash
# 基本依存関係のインストール
pip install -e .

# オプション: 最適化ライブラリのインストール
pip install -e .[optimization]

# オプション: 可視化ライブラリのインストール
pip install -e .[visualization]

# 開発ツールのインストール
pip install -e .[dev]
```

## クイックスタート

### 基本的な使用方法

```python
import pandas as pd
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.modeling.factory import create_kaggle_models
from src.submission.submission_generator import SubmissionGenerator

# データの読み込み
loader = DataLoader()
train_data, test_data = loader.load_train_test()

# 前処理
preprocessor = DataPreprocessor()
X_train = train_data.drop(['Personality', 'id'], axis=1)
y_train = train_data['Personality']
X_test = test_data.drop('id', axis=1)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# モデル学習
models = create_kaggle_models(target_type='regression')
trained_models = {}

for model in models:
    model.fit(X_train_processed, y_train)
    trained_models[model.config.name] = model

# 予測と提出ファイルの生成
test_predictions = {}
for name, model in trained_models.items():
    test_predictions[name] = model.predict(X_test_processed)

submission_gen = SubmissionGenerator()
submission_path = submission_gen.create_submission(
    test_predictions[best_model_name], 
    test_data['id'], 
    filename='submission.csv'
)
```

### コマンドライン実行

```bash
# フルワークフローの実行
python3 scripts/kaggle_workflow.py \
    --target-col Personality \
    --problem-type classification \
    --optimize \
    --ensemble

# Makefileを使用した実行
make personality-prediction
make notebook-run  # 統合Kaggleノートブックの実行
```

### Jupyter Notebook の使用

```bash
# 統合Kaggle提出ノートブックの起動
jupyter notebook notebooks/kaggle_submission_notebook.ipynb

# または個別の分析ノートブックの実行
jupyter notebook notebooks/01_data_exploration_preprocessing.ipynb
jupyter notebook notebooks/02_model_training_evaluation.ipynb
jupyter notebook notebooks/03_ensemble_hyperparameter_tuning.ipynb
jupyter notebook notebooks/04_results_analysis_feature_importance.ipynb
```

## プロジェクト構造

```
src/
├── analysis/               # 分析とレポート
│   ├── comprehensive_analysis.py  # 包括的データ分析
│   ├── data_processor.py          # データ処理ユーティリティ
│   └── feature_importance.py      # 特徴量重要度分析
├── data/                  # データ読み込みと管理
│   └── data_loader.py     # データ読み込みユーティリティ
├── preprocessing/         # データ前処理
│   ├── preprocessor.py    # メイン前処理クラス
│   ├── pipeline.py        # 前処理パイプライン
│   └── transformers.py    # 個別変換器
├── features/              # 特徴量エンジニアリング
│   └── engineering/       # 特徴量生成
│       ├── base.py        # ベース特徴量生成器
│       ├── numeric.py     # 数値特徴量
│       ├── categorical.py # カテゴリ特徴量
│       ├── interaction.py # 交互作用特徴量
│       ├── datetime.py    # 日時特徴量
│       ├── aggregation.py # 集約特徴量
│       └── pipeline.py    # 特徴量パイプライン
├── modeling/              # モデル学習
│   ├── base.py           # ベースモデルクラス
│   ├── tree_models.py    # ツリーモデル（XGBoost, LightGBM, CatBoost）
│   ├── linear_models.py  # 線形モデル
│   ├── ensemble.py       # アンサンブル手法
│   └── factory.py        # モデルファクトリ
├── optimization/          # パラメータ最適化
│   ├── base.py           # ベース最適化クラス
│   ├── grid_search.py    # グリッドサーチ
│   ├── random_search.py  # ランダムサーチ
│   ├── bayesian_optimization.py # ベイジアン最適化
│   ├── optuna_optimizer.py # Optuna最適化
│   └── factory.py        # 最適化ファクトリ
├── evaluation/           # モデル評価
│   ├── metrics.py        # 評価指標
│   └── validation.py     # クロスバリデーションユーティリティ
├── submission/           # 提出ファイル生成
│   └── submission_generator.py
├── config/               # 設定管理
│   └── kaggle_config.py  # Kaggleコンペティション設定
└── utils/                # ユーティリティ
    ├── base.py           # ベースユーティリティ
    ├── config.py         # 設定管理
    └── io.py             # ファイル入出力操作

notebooks/                # Jupyter ノートブック
├── kaggle_submission_notebook.ipynb  # 統合提出ノートブック
├── 01_data_exploration_preprocessing.ipynb
├── 02_model_training_evaluation.ipynb
├── 03_ensemble_hyperparameter_tuning.ipynb
└── 04_results_analysis_feature_importance.ipynb
```

## 使用例

### 1. カスタムデータ処理パイプライン

```python
from src.preprocessing.preprocessor import DataPreprocessor
from src.data.data_loader import DataLoader

# データの読み込み
loader = DataLoader()
train_data, test_data = loader.load_train_test()

# カスタム前処理パイプライン
preprocessor = DataPreprocessor()
X_train = train_data.drop(['Personality', 'id'], axis=1)
y_train = train_data['Personality']

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(test_data.drop('id', axis=1))
```

### 2. モデル学習と評価

```python
from src.modeling.factory import create_kaggle_models
from src.evaluation.validation import CompetitionValidator

# モデルの作成
models = create_kaggle_models(target_type='regression')

# クロスバリデーション評価
validator = CompetitionValidator()
cv_results = {}

for model in models:
    cv_result = validator.cross_validate_model(model, X_train_processed, y_train, cv=5)
    cv_results[model.config.name] = cv_result
    print(f"{model.config.name}: CV RMSE = {cv_result['mean_rmse']:.4f}")
```

### 3. アンサンブル手法

```python
from src.modeling.ensemble import create_ensemble_from_models, create_optimized_ensemble

# 個別モデルの学習
trained_models = {}
for model in models:
    model.fit(X_train_processed, y_train)
    trained_models[model.config.name] = model

# アンサンブルの作成
ensemble = create_ensemble_from_models(list(trained_models.values()), 'average')
ensemble.fit(X_train_processed, y_train)

# 予測の生成
predictions = ensemble.predict(X_test_processed)
```

### 4. 分析と特徴量重要度

```python
from src.analysis.feature_importance import FeatureImportanceAnalyzer

# 特徴量重要度の分析
analyzer = FeatureImportanceAnalyzer()
importance_results = analyzer.analyze_models(trained_models, X_train_processed, y_train)

# 包括的分析レポートの生成
from src.analysis.comprehensive_analysis import ComprehensiveAnalysis

analysis = ComprehensiveAnalysis()
report = analysis.generate_report(trained_models, X_train_processed, y_train, X_test_processed)
```

## 設定

```python
from src.config.kaggle_config import KaggleConfig, ConfigPresets

# 回帰コンペティション用設定
config = ConfigPresets.regression_competition()

# 分類コンペティション用設定
config = ConfigPresets.classification_competition()

# カスタム設定
config = KaggleConfig(
    problem_type='regression',
    preprocessing={
        'handle_missing': True,
        'handle_outliers': True,
        'outlier_threshold': 2.0
    },
    feature_engineering={
        'numeric_features': True,
        'polynomial_features': True,
        'max_interactions': 150
    }
)
```

## テスト実行

```bash
# 全テストの実行
make test

# 高速テストの実行（スローマーカーを除外）
make test-fast

# ユニットテストのみの実行
make test-unit

# 統合テストのみの実行
make test-integration

# カバレッジ付きテストの実行
make test-coverage

# スモークテストの実行
make test-smoke
```

## 開発ガイドライン

### コード品質
- **型ヒント**: すべての関数とメソッドに型アノテーション
- **ドキュメント**: 主要クラスと関数にGoogleスタイルのdocstring
- **テスト**: pytestを使用したユニットテスト
- **フォーマット**: blackによる自動フォーマット

### パフォーマンス
- **CPU最適化**: GPU不要のローカル環境での高速実行
- **メモリ効率**: 大規模データセットの効率的な処理
- **並列処理**: 適用可能な箇所での並列化

### セキュリティ
- **機密情報**: APIキーや認証情報のハードコード禁止
- **入力検証**: 外部データの適切な検証とサニタイゼーション

## 利用可能なコマンド

```bash
# セットアップとインストール
make install              # 基本依存関係のインストール
make dev-install         # 開発ツール付きのインストール
make setup               # 必要なディレクトリの作成

# データ処理とモデリング
make preprocess          # 前処理モジュールのテスト
make model-xgb           # XGBoostモデルのテスト
make model-lgb           # LightGBMモデルのテスト
make ensemble-stacking   # スタッキングアンサンブルのテスト

# ノートブック実行
make notebook-run        # Kaggle提出ノートブックの実行
make notebook-clean      # ノートブック出力のクリーン

# Kaggleワークフロー
make personality-prediction
make personality-prediction  # 性格予測ワークフローの実行

# 開発とテスト
make test               # テストの実行
make lint               # コード品質チェック
make format             # コードフォーマット
make clean              # 生成ファイルのクリーン
```

## ライセンス

MIT License

## 貢献

1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. Pull Requestを作成

## サポート

- Issues: GitHub Issues でバグ報告・質問
- Discussions: GitHub Discussions で一般的な議論