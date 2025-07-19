# CLAUDE.md
このファイルは、このリポジトリでコードを扱う際のClaude Code (claude.ai/code) への指針を提供します。

## 【プロジェクト概要】Kaggle S5E7 性格予測
- **コンペティション**: https://www.kaggle.com/competitions/playground-series-s5e7/overview
- **問題**: 二値分類（内向的 vs 外向的）
- **評価指標**: 精度（Accuracy）
- **現在の順位**: 2749チーム中1182位（上位43.0%）
- **現在のベストスコア**: 0.9684（CV スコア）
- **ブロンズメダル目標**: 0.976518（+0.008の改善が必要）

## 【重要 - 現在のプロジェクト状態】
### 高度な実装段階
- **成熟した実装**: 700行以上の本番環境対応コードを含むプロフェッショナルなMLパイプライン
- **ブロンズメダル目標**: 0.8%の改善が必要（現在: 0.9684、目標: 0.976518）
- **堅牢なアーキテクチャ**: メダリオンデータ管理、パイプライン統合、包括的なCVフレームワーク
- **高いテストカバレッジ**: 18個のテストファイルで475個のテスト、73%のカバレッジ
- **データ準備完了**: DuckDBにコンペティションデータ準備済み `/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb`

### 現在のパフォーマンス状況
- **最新CVスコア**: 0.9684 ± 0.0020（96.84%の精度）
- **ブロンズまでのギャップ**: +0.008の改善が必要
- **アーキテクチャ品質**: 将来の過度な複雑化を防ぐ拡張可能な設計

## 【現在のアーキテクチャ】メダリオン設計とパイプライン統合
```
実装済み構造:
├── src/
│   ├── data/
│   │   ├── bronze.py     # ✅ 生データ処理
│   │   ├── silver.py     # ✅ 特徴量エンジニアリング（30以上の特徴量）
│   │   └── gold.py       # ✅ ML対応データパイプライン
│   ├── models.py         # ✅ パイプライン統合を含むLightGBM（696行）
│   ├── validation.py     # ✅ リーク防止を含むCVフレームワーク（316行）
│   └── util/
│       ├── time_tracker.py   # ✅ 開発効率追跡
│       └── notifications.py  # ✅ ワークフロー通知
├── scripts/
│   ├── train.py          # ✅ 基本的な訓練
│   ├── train_light.py    # ✅ 高速イテレーション（322行）
│   ├── train_enhanced.py # ✅ 高度なパイプライン
│   └── train_heavy.py    # ✅ 完全最適化
└── tests/                # ✅ 73%カバレッジ、475テスト
```

### 現在の開発戦略
1. **ブロンズ最適化**（アクティブ）: ハイパーパラメータ調整、特徴量選択 → 0.976518
2. **シルバー拡張**（準備完了）: XGBoost/CatBoostアンサンブル、高度な特徴量
3. **ゴールド進化**（準備済み）: 複数コンペ再利用性、本番環境デプロイ

## 【データ管理】DuckDB準備完了
- **データベースパス**: `/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb`
- **スキーマ**: `playground_series_s5e7`
- **テーブル**: `train`、`test`、`sample_submission`
- **ターゲット列**: `Personality`（内向的/外向的）
- **ID列**: `id`
- **特徴量**: 合計7個（数値5個 + カテゴリカル2個）

### 特徴量概要
- **数値特徴量**: Time_spent_Alone、Social_event_attendance、Going_outside、Friends_circle_size、Post_frequency
- **カテゴリカル特徴量**: Stage_fear（Yes/No）、Drained_after_socializing（Yes/No）

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
make dev-install         # 開発ツール込みのインストール
make setup               # ディレクトリ構造の作成
make quick-test          # 単一モデルのクイックテスト
make personality-prediction  # フルワークフロー（実装時）
make test                # テストの実行（テスト存在時）
make clean               # 出力のクリーンアップ
make help                # 利用可能なコマンドを表示
```

### 利用可能なコマンド（実装済み）
```bash
# コアワークフロー
make install              # ✅ 依存関係のインストール
make dev-install         # ✅ 開発ツール込みのインストール
make test                # ✅ 475テストの実行（73%カバレッジ）
make quick-test          # ✅ 高速モデル検証
make personality-prediction  # ✅ フル訓練パイプライン
make clean               # ✅ 出力のクリーンアップ

# 訓練バリエーション
python scripts/train_light.py    # ✅ 高速イテレーション（0.5秒）
python scripts/train.py          # ✅ 標準訓練
python scripts/train_enhanced.py # ✅ 高度な特徴量
python scripts/train_heavy.py    # ✅ 完全最適化
```

## 【依存関係と環境】
### インストール（pyproject.toml設定済み）
```bash
pip install -e .                    # 基本的なML依存関係
pip install -e .[dev]              # + 開発ツール
pip install -e .[optimization]     # + 調整用Optuna
pip install -e .[visualization]    # + プロットライブラリ
```

### コア依存関係
- **データ**: pandas、numpy、duckdb
- **モデル**: scikit-learn、xgboost、lightgbm、catboost
- **最適化**: optuna
- **開発**: pytest、black、flake8、mypy
- **Python**: 3.8+

## 【現在のパフォーマンス】最近の訓練結果
- **Light Enhanced モデル**: 96.79% ± 0.22%（最新実行、30特徴量）
- **ベースラインモデル**: 96.84% ± 0.20%（最高CVスコア、10特徴量）
- **訓練効率**: 0.5秒（light）、0.39秒（ベースライン）
- **特徴量重要度**: poly_extrovert_score_Post_frequency（257.6）が最上位
- **ブロンズギャップ**: +0.8%必要（最適化で十分達成可能）

### パフォーマンス分析
- **一貫した結果**: 低い標準偏差は安定したモデルを示す
- **高速イテレーション**: 1秒未満の訓練により迅速な実験が可能
- **特徴量品質**: 多項式特徴量が強い予測力を示す

## 【実装ガイドライン】
### 設計原則（バランスの取れたアプローチ）
- **拡張可能なシンプルさ**: 複雑さを増やさずに成長をサポートするクリーンな抽象化
- **リーク防止**: パイプライン統合によりCV対応の前処理を確保（実装済み）
- **CVを信頼**: 整合性検証を含むStratifiedKFold（実装済み）
- **エビデンスベース**: 重要度分析に基づく特徴量エンジニアリング
- **段階的開発**: メダリオンアーキテクチャが段階的な拡張をサポート

### 主要な実装メモ
- **CSVファイル不使用**: すべてのデータアクセスはDuckDB経由のみ
- **システムPython**: 仮想環境なし（プロジェクト履歴による）
- **分類設定**: `Personality`ターゲットの二値分類
- **精度指標**: 主要な評価基準

### 開発ワークフロー（最適化済み）
1. **現状を最適化**: ブロンズ向けのハイパーパラメータ調整と特徴量選択
2. **高速イテレーション**: 0.5秒の訓練サイクルで迅速な実験が可能
3. **厳密な検証**: データリーク防止とCV整合性チェック（実装済み）
4. **包括的なテスト**: 統合テストを含む73%カバレッジ（実装済み）
5. **思慮深いスケーリング**: メダリオンアーキテクチャが制御された拡張をサポート

## 【成功基準】
- **ブロンズメダル**: 0.976518+の精度（現在の0.9684から+0.8%）
- **アーキテクチャ品質**: 複雑さを制御した拡張可能な設計
- **信頼性**: データリーク防止、再現可能なCV結果（実装済み）
- **開発効率**: 1秒未満の訓練、包括的なテスト（実装済み）
- **長期的価値**: 将来のコンペで再利用可能なパターン

## 【ブロンズメダルロードマップ】
### 即座の機会（1-2週間）
1. **ハイパーパラメータ最適化**: 既存のOptuna統合を活用
2. **特徴量選択**: トップ重要度特徴量に焦点（poly_extrovert_score_*）
3. **モデルアンサンブル**: 予測の安定性のためにCVフォールドを組み合わせ
4. **閾値調整**: 精度向上のための分類閾値の最適化

### 準備済み技術資産
- ✅ **データリーク防止**: パイプライン統合実装済み
- ✅ **CVフレームワーク**: 整合性検証と層別サンプリング
- ✅ **特徴量エンジニアリング**: 重要度ランキング付き30以上の特徴量
- ✅ **最適化インフラ**: ハイパーパラメータ調整用Optuna統合
- ✅ **パフォーマンス監視**: 時間追跡と包括的なロギング