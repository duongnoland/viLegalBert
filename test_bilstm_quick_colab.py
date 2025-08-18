#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn

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

def _load(level):
	p = f"{BASE_DIR}/models/saved_models/level{level}_classifier/bilstm_level{level}/bilstm_level{level}_model.pkl"
	with open(p, "rb") as f:
		return pickle.load(f)

def _build_inputs(data, texts):
	cfg = data['config']
	lbl = data['label_encoder']
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 1) Try tokenizer-based pipeline (embedding BiLSTM)
	tokenizer = data.get('tokenizer', None)
	tokenizer_state = data.get('tokenizer_state', None)
	if tokenizer_state is not None and tokenizer is None:
		class _Tok: pass
		tokenizer = _Tok()
		tokenizer.word_to_idx = tokenizer_state.get('word_to_idx', {})
		tokenizer.pad_token = tokenizer_state.get('pad_token', '<pad>')
		tokenizer.unk_token = tokenizer_state.get('unk_token', '<unk>')
		tokenizer.start_token = tokenizer_state.get('start_token', '<s>')
		tokenizer.end_token = tokenizer_state.get('end_token', '</s>')
		tokenizer.vocab_size = tokenizer_state.get('vocab_size', max(tokenizer.word_to_idx.values(), default=0) + 1)
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
			ml = cfg.get('max_length', 256)
			if len(ids) > ml:
				ids = ids[:ml]
			if len(ids) < ml:
				ids.extend([pad_id] * (ml - len(ids)))
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
		return model.eval(), X_t, lbl, device

	# 2) Fallback: TF-IDF based (legacy)
	vec = data['vectorizer']
	X = vec.transform(texts).toarray()
	vocab_size = X.shape[1]
	max_len = cfg.get('max_length', 1000)

	X_a = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1, vocab_size)
	model_a = BiLSTMClassifier(vocab_size, cfg['hidden_size'], cfg['num_layers'], len(lbl.classes_), cfg['dropout']).to(device)

	x1d = []
	for row in X:
		if len(row) > max_len:
			x1d.append(row[:max_len])
		else:
			pad = np.zeros(max_len, dtype=np.float32)
			pad[:len(row)] = row
			x1d.append(pad)
	X_b = torch.tensor(np.stack(x1d, axis=0), dtype=torch.float32).unsqueeze(-1).to(device)  # (N, max_len, 1)
	model_b = BiLSTMClassifier(1, cfg['hidden_size'], cfg['num_layers'], len(lbl.classes_), cfg['dropout']).to(device)

	state = data['model_state_dict']
	try:
		model_a.load_state_dict(state, strict=True)
		return model_a.eval(), X_a, lbl, device
	except Exception:
		model_b.load_state_dict(state, strict=True)
		return model_b.eval(), X_b, lbl, device

def predict_texts(texts):
	d1 = _load(1); m1, X1, l1, dev = _build_inputs(d1, texts)
	with torch.no_grad():
		y1 = torch.argmax(m1(X1), dim=1).cpu().numpy()
	lvl1 = l1.inverse_transform(y1)

	d2 = _load(2); m2, X2, l2, dev = _build_inputs(d2, texts)
	with torch.no_grad():
		y2 = torch.argmax(m2(X2), dim=1).cpu().numpy()
	lvl2 = l2.inverse_transform(y2)

	return list(zip(texts, lvl1, lvl2))

def main():
	parser = argparse.ArgumentParser(description="Quick test BiLSTM on Colab")
	parser.add_argument("--text", type=str)
	args, _ = parser.parse_known_args()  # ignore Jupyter -f

	if args.text:
		texts = [args.text]
	else:
		texts = [
			"Nghị định về quy định chi tiết thi hành một số điều của Luật Doanh nghiệp",
			"Thông tư hướng dẫn thực hiện các quy định về thuế thu nhập cá nhân",
			"Quyết định về việc phê duyệt quy hoạch giao thông vận tải",
		]

	results = predict_texts(texts)
	for i, (t, l1, l2) in enumerate(results, 1):
		snippet = t[:120] + ("..." if len(t) > 120 else "")
		print(f"\n📝 Sample {i}: {snippet}")
		print(f"🏷️ Level 1: {l1}")
		print(f"🏷️ Level 2: {l2}")

if __name__ == "__main__":
	main()