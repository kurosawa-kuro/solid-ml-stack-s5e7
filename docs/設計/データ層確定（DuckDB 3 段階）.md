# データ層確定（DuckDB 3 段階）

## 概要
Kaggle S5E7 Personality Prediction コンペのデータ管理を3段階のDuckDBテーブル構成で行う。

## DuckDB テーブル構成 

### 1. Raw データ層
```sql
-- playground_series_s5e7.train_raw
-- playground_series_s5e7.test_raw  
-- playground_series_s5e7.sample_submission_raw
```
- **目的**: Kaggle からダウンロードした CSV データの保存
- **内容**: CSV ファイルそのままの形式
- **利用場面**: データの初期確認時

### 2. Preprocessed データ層
```sql
-- playground_series_s5e7.train_preprocessed
-- playground_series_s5e7.test_preprocessed
```
- **目的**: 基本的な前処理済みデータ
- **内容**: 
  - 欠損値補完: median, カテゴリ: mode
  - データ型変換
  - 基本的な値の正規化
- **利用場面**: CV fold 作成時の基準データ

### 3. Engineered データ層（特徴量エンジニアリング）
```sql
-- playground_series_s5e7.train_engineered  
-- playground_series_s5e7.test_engineered
```
- **目的**: 機械学習用の最終特徴量セット
- **内容**:
  - カテゴリ変数のエンコーディング
  - 交互作用項の作成
  - その他の特徴量生成

## 実装方針

### データアクセス（src/data.py）

```python
class DataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
    
    def load_raw(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Raw データ読み込み"""
        
    def load_preprocessed(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocessed データ読み込み"""
        
    def load_engineered(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Engineered データ読み込み"""

class DataPreprocessor:
    def preprocess_raw_to_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw から Preprocessed への変換"""
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessed から Engineered への変換"""
```

### 特徴量詳細

#### 基本特徴量（7項目）
- **数値**: Time_spent_Alone, Social_event_attendance, Going_outside, Friends_circle_size, Post_frequency
- **カテゴリ**: Stage_fear (Yes/No), Drained_after_socializing (Yes/No)
- **ターゲット**: Personality (Introvert/Extrovert)

#### エンジニアリング例
- **比率特徴量**: Social_event_attendance / Time_spent_Alone
- **合計特徴量**: Going_outside + Social_event_attendance  
- **カテゴリ交互作用**: Stage_fear と Drained_after_socializing
- **エンコーディング**: Label/Target encoding for categorical features

## Make コマンド

### make setup-data
```bash
# CSV から DuckDB Raw 層への変換
python scripts/setup_data.py
```

### make preprocess-data  
```bash
# Raw から Preprocessed への変換
python scripts/preprocess.py
```

### make engineer-features
```bash
# Preprocessed から Engineered への変換  
python scripts/feature_engineering.py
```

## データ利用パターン

### 基本利用
```python
from src.data import DataLoader

loader = DataLoader('/path/to/kaggle_datasets.duckdb')
train, test = loader.load_engineered()
```

### CV 設定注意点
- Preprocessed レベルでの分割を基準とする
- Engineered での Target encoding は CV fold を考慮
- リークを防ぐため fold ごとに encoding

## 実装チェックリスト

### データ変換
- [ ] Raw 層への CSV 読み込み
- [ ] Preprocessed 層での欠損値処理
- [ ] Engineered 層でのカテゴリ変数と数値変換
- [ ] Train/Test での一貫性確保 

### パフォーマンス  
- [ ] DuckDB クエリ最適化（インデックス等）
- [ ] メモリ効率の改善
- [ ] 数値計算の高速化

## ディレクトリ構成 
```
data/
   kaggle_datasets.duckdb    # 3層すべて格納
   submissions/              # 提出ファイル保存

src/  
   data.py                   # DataLoader, DataPreprocessor
   features.py               # 特徴量エンジニアリング関数

scripts/
   setup_data.py            # CSV→Raw変換
   preprocess.py            # Raw→Preprocessed変換  
   feature_engineering.py   # Preprocessed→Engineered変換
```

## 開発手順
1. **Raw層構築**: CSVファイルの取り込み
2. **前処理層**: 欠損値処理とデータ型変換 
3. **特徴量層**: 機械学習用の最終データセット作成 < 現在
4. **検証**: CVでの性能確認