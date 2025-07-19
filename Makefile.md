### 1. **基本セットアップ**
- `install` - 基本的な依存関係のインストール
- `dev-install` - 開発用の追加パッケージを含むインストール
- `setup` - 必要なディレクトリ構造の作成

### 2. **メインワークフロー**
- `quick-test` - 単一モデルでの高速テスト実行
- `personality-prediction` - 最適化とアンサンブルを含む完全なワークフロー

### 3. **メンテナンス**
- `test` - 基本的なテスト実行
- `clean` - 出力ファイルのクリーンアップ
- `help` - 使い方の表示

## 🚀 使い方

```bash
# 初回セットアップ
make install
make setup

# テスト実行
make quick-test

# 本番実行
make personality-prediction

# クリーンアップ
make clean
```
