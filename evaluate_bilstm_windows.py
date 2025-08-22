#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

# Suppress sklearn version mismatch warnings when unpickling
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# Also suppress torch weights_only-related warnings on older checkpoints
warnings.filterwarnings("ignore", category=UserWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Detected pickle protocol.*")

# Base directory inferred from script location by default
BASE_DIR: Path = Path(__file__).resolve().parent


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        w = torch.softmax(self.attention(out), dim=1)
        attended = torch.sum(w * out, dim=1)
        return self.classifier(attended)


class BiLSTMTokenClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        super(BiLSTMTokenClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        w = torch.softmax(self.attention(out), dim=1)
        attended = torch.sum(w * out, dim=1)
        return self.classifier(attended)


# SimpleTokenizer class to match training-time pickled objects
class SimpleTokenizer:
    def __init__(self, max_vocab_size=20000, pad_token='<PAD>', unk_token='<UNK>', start_token='<START>', end_token='<END>'):
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

    def _clean(self, text: str) -> str:
        import re
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize(self, text: str):
        return text.split()

    def fit(self, texts):
        from collections import Counter
        counter = Counter()
        for t in texts:
            toks = self._tokenize(self._clean(t))
            counter.update(toks)
        specials = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        self.word_to_idx = {tok: idx for idx, tok in enumerate(specials)}
        self.idx_to_word = {idx: tok for idx, tok in enumerate(specials)}
        limit = max(0, self.max_vocab_size - len(specials))
        for i, (w, _) in enumerate(counter.most_common(limit)):
            idx = len(specials) + i
            self.word_to_idx[w] = idx
            self.idx_to_word[idx] = w
        self.vocab_size = len(self.word_to_idx)
        return self

    def text_to_ids(self, text: str, max_length: int):
        toks = self._tokenize(self._clean(text))
        ids = [self.word_to_idx[self.start_token]]
        for w in toks:
            ids.append(self.word_to_idx.get(w, self.word_to_idx[self.unk_token]))
        ids.append(self.word_to_idx[self.end_token])
        if len(ids) > max_length:
            ids = ids[:max_length]
        if len(ids) < max_length:
            ids.extend([self.word_to_idx[self.pad_token]] * (max_length - len(ids)))
        return ids

    def texts_to_ids(self, texts, max_length: int):
        return [self.text_to_ids(t, max_length) for t in texts]


def _load_artifacts(base_dir_path: Path, level: int):
    model_pkl = base_dir_path / f"models/saved_models/level{level}_classifier/bilstm_level{level}/bilstm_level{level}_model.pkl"
    if not model_pkl.exists():
        raise FileNotFoundError(f"Model pickle not found at: {model_pkl}")
    try:
        # Prefer torch.load with explicit weights_only=False and CPU remap
        data = torch.load(model_pkl, map_location=torch.device("cpu"), weights_only=False)
    except Exception:
        try:
            with model_pkl.open("rb") as f:
                data = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
        except Exception:
            # Fallback to plain pickle if it wasn't saved via torch.save
            # Monkey-patch torch.load used internally by torch picklers to force CPU mapping
            original_torch_load = torch.load
            def _cpu_torch_load_wrapper(*args, **kwargs):
                kwargs.setdefault("map_location", torch.device("cpu"))
                kwargs.setdefault("weights_only", False)
                return original_torch_load(*args, **kwargs)
            torch.load = _cpu_torch_load_wrapper
            try:
                with model_pkl.open("rb") as f:
                    data = pickle.load(f)
            finally:
                # Restore original torch.load
                torch.load = original_torch_load
    # Ensure all tensors in state_dict are on CPU
    if isinstance(data, dict) and 'model_state_dict' in data and isinstance(data['model_state_dict'], dict):
        cpu_state_dict = {}
        for key, value in data['model_state_dict'].items():
            if torch.is_tensor(value):
                cpu_state_dict[key] = value.cpu()
            else:
                cpu_state_dict[key] = value
        data['model_state_dict'] = cpu_state_dict
    return data


def _rebuild_model_and_inputs(data, texts):
    cfg = data['config']
    lbl = data['label_encoder']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer-based pipeline (preferred)
    tokenizer = data.get('tokenizer', None)
    tokenizer_state = data.get('tokenizer_state', None)
    if tokenizer_state is not None and tokenizer is None:
        class _Tok:
            pass
        tokenizer = _Tok()
        tokenizer.word_to_idx = tokenizer_state['word_to_idx']
        tokenizer.pad_token = tokenizer_state['pad_token']
        tokenizer.unk_token = tokenizer_state['unk_token']
        tokenizer.start_token = tokenizer_state['start_token']
        tokenizer.end_token = tokenizer_state['end_token']
        tokenizer.vocab_size = tokenizer_state['vocab_size']

        def _clean(text):
            import re
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def _tokenize(text):
            return _clean(text).split()

        def text_to_ids(text, max_length: int):
            toks = _tokenize(text)
            ids = [tokenizer.word_to_idx.get(tokenizer.start_token, 0)]
            unk_id = tokenizer.word_to_idx.get(tokenizer.unk_token, 1)
            pad_id = tokenizer.word_to_idx.get(tokenizer.pad_token, 0)
            end_id = tokenizer.word_to_idx.get(tokenizer.end_token, 0)
            for w in toks:
                ids.append(tokenizer.word_to_idx.get(w, unk_id))
            ids.append(end_id)
            if len(ids) > cfg.get('max_length', 256):
                ids = ids[:cfg.get('max_length', 256)]
            if len(ids) < cfg.get('max_length', 256):
                ids.extend([pad_id] * (cfg.get('max_length', 256) - len(ids)))
            return ids

        tokenizer.texts_to_ids = lambda arr, ml: [text_to_ids(t, ml) for t in arr]

    if tokenizer is not None:
        seq = tokenizer.texts_to_ids(texts, cfg.get('max_length', 256))
        X_t = torch.tensor(seq, dtype=torch.long).to(device)
        model = BiLSTMTokenClassifier(
            vocab_size=getattr(tokenizer, 'vocab_size', cfg.get('max_features', 20000)),
            embedding_dim=cfg.get('embedding_dim', 256),
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            num_classes=len(lbl.classes_),
            dropout=cfg['dropout'],
        ).to(device)
        state = data['model_state_dict']
        model.load_state_dict(state, strict=True)
        if device.type == 'cpu' and next(model.parameters()).dtype == torch.float16:
            model = model.float()
        model.eval()
        print(f"Loaded Embedding BiLSTM | vocab_size={getattr(tokenizer, 'vocab_size', 'NA')} | max_len={cfg.get('max_length', 256)}")
        print(f"Label classes[{len(lbl.classes_)}]: {list(lbl.classes_)}")
        return model, X_t, lbl, device

    # Fallback: TF-IDF based evaluation (backward compatibility)
    vec = data['vectorizer']
    X = vec.transform(texts).toarray()
    vocab_size = X.shape[1]
    max_len = cfg.get('max_length', 1000)

    X_a = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    model_a = BiLSTMClassifier(
        input_size=vocab_size,
        hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'],
        num_classes=len(lbl.classes_),
        dropout=cfg['dropout'],
    ).to(device)
    state = data['model_state_dict']
    try:
        model_a.load_state_dict(state, strict=True)
        model = model_a
        X_t = X_a.to(device)
        loaded_variant = 'A(seq_len=1,input_size=vocab)'
    except Exception:
        x1d = []
        for row in X:
            if len(row) > max_len:
                x1d.append(row[:max_len])
            else:
                p = np.zeros(max_len, dtype=np.float32)
                p[:len(row)] = row
                x1d.append(p)
        X_b = torch.tensor(np.stack(x1d, axis=0), dtype=torch.float32).unsqueeze(-1)
        model_b = BiLSTMClassifier(
            input_size=1,
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            num_classes=len(lbl.classes_),
            dropout=cfg['dropout'],
        ).to(device)
        model_b.load_state_dict(state, strict=False)
        model = model_b
        X_t = X_b.to(device)
        loaded_variant = 'B(seq_len=max_len,input_size=1)'

    model.eval()
    print(f"Loaded BiLSTM variant: {loaded_variant} | dtype: {next(model.parameters()).dtype}")
    try:
        print(f"Vectorizer features: {vec.vocabulary_ and len(vec.vocabulary_)}")
    except Exception:
        pass
    print(f"Label classes[{len(lbl.classes_)}]: {list(lbl.classes_)}")
    return model, X_t, lbl, device


def evaluate_level(base_dir_path: Path, level: int, df: pd.DataFrame, batch_size: int = 32) -> float:
    print(f"\nEvaluate BiLSTM Level {level}...")
    data = _load_artifacts(base_dir_path, level)
    texts = df['text'].fillna('')
    y_true = df['type_level1'] if level == 1 else df['domain_level2']

    model, X_t, lbl, device = _rebuild_model_and_inputs(data, texts)
    all_pred_indices = []
    model.eval()
    with torch.inference_mode():
        num_samples = X_t.size(0)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_inputs = X_t[start:end]
            logits = model(batch_inputs)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred_indices.append(pred)

    pred_indices = np.concatenate(all_pred_indices, axis=0)
    y_pred = lbl.inverse_transform(pred_indices)
    acc = accuracy_score(y_true, y_pred)
    print(f"Level {level} Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Prediction distribution (optional)
    try:
        import collections
        cnt = collections.Counter(y_pred)
        print(f"Prediction distribution (top 10): {cnt.most_common(10)}")
    except Exception:
        pass
    return acc


def main(test_path: Optional[str] = None, base_dir: Optional[str] = None, batch_size: int = 32) -> None:
    base_dir_path = Path(base_dir).resolve() if base_dir else BASE_DIR
    if test_path is None:
        test_csv_path = base_dir_path / "data" / "processed" / "dataset_splits" / "test.csv"
    else:
        test_csv_path = Path(test_path).resolve()

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found at: {test_csv_path}")

    print("Evaluating BiLSTM models on test set...")
    df = pd.read_csv(test_csv_path, encoding='utf-8')
    acc1 = evaluate_level(base_dir_path, 1, df, batch_size=batch_size)
    acc2 = evaluate_level(base_dir_path, 2, df, batch_size=batch_size)
    print("\nEVALUATION DONE")
    print(f"Level 1 Acc: {acc1:.4f}")
    print(f"Level 2 Acc: {acc2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM models (Windows-friendly)")
    parser.add_argument("--test-path", type=str, default=None, help="Optional path to test.csv")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Optional repository base directory (defaults to this script's parent directory)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation (default: 32)")
    args = parser.parse_args()

    main(test_path=args.test_path, base_dir=args.base_dir, batch_size=args.batch_size) 