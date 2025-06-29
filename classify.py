import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load Japanese spaCy model
def load_tokenizer():
    try:
        nlp = spacy.load('ja_ginza')
    except OSError:
        raise RuntimeError("ja_ginza model is not installed. Run 'python -m spacy download ja_ginza'.")
    return nlp

# Tokenizer using spaCy
def spacy_tokenizer(text, nlp=None):
    if nlp is None:
        nlp = load_tokenizer()
    doc = nlp(text)
    return [token.text for token in doc]

# Build training pipeline
def train_model(train_texts, train_labels):
    nlp = load_tokenizer()
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=lambda text: spacy_tokenizer(text, nlp))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(train_texts, train_labels)
    return pipeline

if __name__ == '__main__':
    # Sample training data for five categories
    train_texts = [
        '今何時ですか？',
        '明日の天気はどうなるでしょうか。',
        'このエラーをどうやって解決すればいいでしょうか。',
        '進学か就職か迷っています。',
        '風の声が聞こえる丘に私は立っていた。',
        '彼は深い森の奥へと歩みを進めた。',
        '今週の売上レポートを提出してください。',
        '会議は15時から開始します。',
        '昨日は友達と映画を見ました。',
        '今日はどこに食べに行こうか。'
    ]
    train_labels = [
        '質問', '質問',
        '相談', '相談',
        '創作', '創作',
        '業務', '業務',
        '雑談', '雑談'
    ]

    model = train_model(train_texts, train_labels)

    text = input('文章を入力してください: ')
    prediction = model.predict([text])[0]
    print('予測結果:', prediction)
