# CLAUDE.md (日本語版)

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを提供します。

## 【プロジェクト概要】Kaggle S5E7 性格予測
- **コンペティション**: https://www.kaggle.com/competitions/playground-series-s5e7/overview
- **問題**: 2値分類（内向的 vs 外向的）
- **評価指標**: 精度（Accuracy）
- **現在の順位**: 2749チーム中1182位（上位43.0%）
- **ベストスコア**: 0.974898
- **ブロンズメダル目標**: 0.976518（+0.00162の改善が必要）

## 【重要 - 現在のプロジェクト状態】

### プロジェクトはリセットされました
- **以前の実装は削除済み**: 過度に複雑な76ファイル/15,000行を削除
- **まっさらな状態**: インフラファイルのみ残存（Makefile、pyproject.toml、docs/）
- **アクティブなコードなし**: src/、scripts/、tests/ディレクトリは空で実装が必要
- **データは準備完了**: DuckDBのコンペティションデータは `/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb` に準備済み

### 以前の問題点（リセットにより解決済み）
- 過度な複雑性、データリーケージ、ファクトリパターンの乱用、アーキテクチャの問題

## 【実装計画】シンプルで効果的なアプローチ

```
目標とする構造（これから構築）：
├── src/
│   ├── data.py          # DuckDBデータ読み込み
│   ├── features.py      # 特徴量エンジニアリング
│   ├── models.py        # LightGBM、XGBoost、CatBoost
│   ├── validation.py    # クロスバリデーション
│   ├── ensemble.py      # モデル結合
│   └── submission.py    # 提出ファイル生成
├── scripts/
│   └── workflow.py      # メインパイプライン
└── tests/
    └── test_*.py        # ユニットテスト
```

### 実装戦略
1. **フェーズ1**: シンプルなLightGBMベースライン（目標: 0.975+）
2. **フェーズ2**: XGBoost/CatBoost追加 + 特徴量エンジニアリング
3. **フェーズ3**: アンサンブル最適化（目標: ブロンズメダルの0.976518+）

## 【データ管理】DuckDB準備完了
- **データベースパス**: `/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb`
- **スキーマ**: `playground_series_s5e7`
- **テーブル**: `train`、`test`、`sample_submission`
- **ターゲット列**: `Personality`（内向的/外向的）
- **ID列**: `id`
- **特徴量**: 合計7個（数値5個 + カテゴリ2個）

### 特徴量の概要
- **数値特徴量**: Time_spent_Alone、Social_event_attendance、Going_outside、Friends_circle_size、Post_frequency
- **カテゴリ特徴量**: Stage_fear（Yes/No）、Drained_after_socializing（Yes/No）

### データアクセスパターン
```python
import duckdb
conn = duckdb.connect('/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb')
train = conn.execute("SELECT * FROM playground_series_s5e7.train").df()
test = conn.execute("SELECT * FROM playground_series_s5e7.test").df()
```

## 【開発コマンド】

### 現在利用可能（Makefile）
```bash
make install              # 依存関係のインストール
make dev-install         # 開発ツール込みでインストール
make setup               # ディレクトリ構造の作成
make quick-test          # 単一モデルでのクイックテスト
make personality-prediction  # フルワークフロー（実装後）
make test                # テスト実行（テスト作成後）
make clean               # 出力のクリーンアップ
make help                # 利用可能なコマンドを表示
```

### 目標コマンド（実装後）
```bash
# 開発ワークフロー
make data-explore        # 初期データ探索
make baseline           # LightGBMベースライン実行
make models             # 全モデル訓練（LGB、XGB、CatBoost）
make ensemble           # アンサンブル予測作成
make submit             # 提出ファイル生成

# 個別モデルテスト
make model-lgb          # LightGBMのみ
make model-xgb          # XGBoostのみ
make model-cat          # CatBoostのみ
```

## 【依存関係と環境】

### インストール（pyproject.toml設定済み）
```bash
pip install -e .                    # 基本的なML依存関係
pip install -e .[dev]              # + 開発ツール
pip install -e .[optimization]     # + Optunaでチューニング
pip install -e .[visualization]    # + 可視化ライブラリ
```

### コア依存関係
- **データ**: pandas、numpy、duckdb
- **モデル**: scikit-learn、xgboost、lightgbm、catboost
- **最適化**: optuna
- **開発**: pytest、black、flake8、mypy
- **Python**: 3.8+

## 【実証済みベンチマーク】以前のパフォーマンス
- **LightGBM**: 96.90%（±0.24%）← ベスト単一モデル
- **XGBoost**: 96.86%（±0.23%）
- **Random Forest**: 96.77%（±0.21%）
- **予測分布**: 外向的 74.7%、内向的 25.3%

## 【実装ガイドライン】

### 設計原則
- **シンプルに保つ**: 単一ファイルモジュール、過度なエンジニアリングを避ける
- **リーク防止**: 適切なCV対応の前処理
- **CVを信頼する**: StratifiedKFoldバリデーション
- **データ駆動**: 効果的な特徴量のみに焦点を当てる

### 主要な実装メモ
- **CSVファイルなし**: すべてのデータアクセスはDuckDB経由のみ
- **システムPython**: 仮想環境なし（プロジェクト履歴による）
- **分類設定**: `Personality`ターゲットの2値分類
- **精度指標**: 主要な評価基準

### 開発ワークフロー
1. **シンプルに始める**: 最小限の特徴量で基本的なLightGBM
2. **高速に反復**: 小さな改善、頻繁な検証
3. **段階的に複雑性を追加**: 有益であることが証明された場合のみ
4. **すべてをテスト**: 全コンポーネントのユニットテスト
5. **CVを信頼**: パブリックリーダーボードに過剰適合しない

## 【成功基準】
- **パフォーマンス**: 0.976518+の精度（ブロンズメダル閾値）
- **コード品質**: クリーンでシンプル、メンテナブルな実装
- **信頼性**: 適切なCVバリデーションによる再現可能な結果
- **効率性**: 高速な開発サイクル、迅速な反復