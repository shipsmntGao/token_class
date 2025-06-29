import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import classify

class DummyToken:
    def __init__(self, text):
        self.text = text

class DummyNLP:
    def __call__(self, text):
        return [DummyToken(t) for t in text.split()]

def test_spacy_tokenizer_with_stub():
    nlp = DummyNLP()
    tokens = classify.spacy_tokenizer('foo bar', nlp)
    assert tokens == ['foo', 'bar']

def test_train_model_basic(monkeypatch):
    def dummy_tokenizer(text, nlp=None):
        return text.split()
    monkeypatch.setattr(classify, 'load_tokenizer', lambda: None)
    monkeypatch.setattr(classify, 'spacy_tokenizer', dummy_tokenizer)
    model = classify.train_model(['apple', 'banana'], ['A', 'B'])
    assert model.predict(['apple'])[0] == 'A'
    assert model.predict(['banana'])[0] == 'B'

def test_load_tokenizer_failure(monkeypatch):
    def raise_oserror(name):
        raise OSError('missing')
    monkeypatch.setattr(classify.spacy, 'load', raise_oserror)
    with pytest.raises(RuntimeError):
        classify.load_tokenizer()
