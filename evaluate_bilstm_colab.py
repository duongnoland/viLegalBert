#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = "/content/viLegalBert"

class BiLSTMClassifier(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
		super(BiLSTMClassifier, self).__init__()
		self.lstm = nn.LSTM(
			input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
			batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
		)
		self.attention = nn.Sequential(
			nn.Linear(hidden_size * 2, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, 1)
		)
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden_size * 2, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, num_classes)
		)

	def forward(self, x):
		out, _ = self.lstm(x)
		w = torch.softmax(self.attention(out), dim=1)
		attended = torch.sum(w * out, dim=1)
		return self.classifier(attended)

class BiLSTMTokenClassifier(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout=0.5):
		super(BiLSTMTokenClassifier, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
		self.lstm = nn.LSTM(
			input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
			batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
		)
		self.attention = nn.Sequential(
			nn.Linear(hidden_size * 2, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, 1)
		)
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden_size * 2, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, num_classes)
		)

	def forward(self, x):
		emb = self.embedding(x)
		out, _ = self.lstm(emb)
		w = torch.softmax(self.attention(out), dim=1)
		attended = torch.sum(w * out, dim=1)
		return self.classifier(attended)

def _load_artifacts(level: int):
	model_pkl = f"{BASE_DIR}/models/saved_models/level{level}_classifier/bilstm_level{level}/bilstm_level{level}_model.pkl"
	with open(model_pkl, "rb") as f:
		data = pickle.load(f)
	return data

def _rebuild_model_and_inputs(data, texts):
	cfg = data['config']
	lbl = data['label_encoder']
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Try tokenizer-based pipeline first
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
			dropout=cfg['dropout']
		).to(device)
		state = data['model_state_dict']
		model.load_state_dict(state, strict=True)
		if device.type == 'cpu' and next(model.parameters()).dtype == torch.float16:
			model = model.float()
		model.eval()
		print(f"🔧 Loaded Embedding BiLSTM | vocab_size={getattr(tokenizer, 'vocab_size', 'NA')} | max_len={cfg.get('max_length', 256)}")
		print(f"🔧 Label classes[{len(lbl.classes_)}]: {list(lbl.classes_)}")
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
		dropout=cfg['dropout']
	).to(device)
	state = data['model_state_dict']
	try:
		model_a.load_state_dict(state, strict=True)
		model = model_a; X_t = X_a.to(device); loaded_variant = 'A(seq_len=1,input_size=vocab)'
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
			dropout=cfg['dropout']
		).to(device)
		model_b.load_state_dict(state, strict=False)
		model = model_b; X_t = X_b.to(device); loaded_variant = 'B(seq_len=max_len,input_size=1)'

	model.eval()
	print(f"🔧 Loaded BiLSTM variant: {loaded_variant} | dtype: {next(model.parameters()).dtype}")
	print(f"🔧 Vectorizer features: {vec.vocabulary_ and len(vec.vocabulary_)}")
	print(f"🔧 Label classes[{len(lbl.classes_)}]: {list(lbl.classes_)}")
	return model, X_t, lbl, device

def evaluate_level(level: int, df):
	print(f"\n🏷️ Evaluate BiLSTM Level {level}...")
	data = _load_artifacts(level)
	texts = df['text'].fillna('')
	y_true = df['type_level1'] if level == 1 else df['domain_level2']

	model, X_t, lbl, device = _rebuild_model_and_inputs(data, texts)
	with torch.no_grad():
		logits = model(X_t)
		pred = torch.argmax(logits, dim=1).cpu().numpy()
	y_pred = lbl.inverse_transform(pred)
	acc = accuracy_score(y_true, y_pred)
	print(f"✅ Level {level} Test Accuracy: {acc:.4f}")
	print(classification_report(y_true, y_pred))
	# Prediction distribution
	try:
		import collections
		cnt = collections.Counter(y_pred)
		print(f"🔎 Prediction distribution (top 10): {cnt.most_common(10)}")
	except Exception:
		pass
	return acc

def main(test_path: str = None):
	if test_path is None:
		test_path = f"{BASE_DIR}/data/processed/dataset_splits/test.csv"
	df = pd.read_csv(test_path, encoding='utf-8')
	acc1 = evaluate_level(1, df)
	acc2 = evaluate_level(2, df)
	print("\n🎉 EVALUATION DONE")
	print(f"Level 1 Acc: {acc1:.4f}")
	print(f"Level 2 Acc: {acc2:.4f}")

if __name__ == "__main__":
	main()