# token_class

ChatAi などのサービスでユーザーが入力した日本語テキストを `ja_ginza` を利用してトークン化し、`scikit-learn` を用いて分類するサンプルコードです。

## 使い方

1. 依存パッケージをインストールします。
   ```bash
   pip install spacy ja_ginza scikit-learn
   ```
   `ja_ginza` モデルがインストールされていない場合は以下を実行してください。
   ```bash
   python -m spacy download ja_ginza
   ```
2. `classify.py` を実行し、表示されるプロンプトに文章を入力します。
   ```bash
   python classify.py
   ```
   簡易的な学習データを用いて学習した分類器が入力された文章を
   `質問`, `相談`, `創作`, `業務`, `雑談` のいずれかに分類します。

このコードはデモ用であり、本番環境で利用する際は十分な学習データと評価が必要です。
