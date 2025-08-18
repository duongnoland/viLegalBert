#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = "/content/viLegalBert"


def _load_model_and_label_encoder(level: int):
	if level == 1:
		model_dir = f"{BASE_DIR}/models/saved_models/level1_classifier/phobert_level1/phobert_level1_model"
		label_path = f"{model_dir}/label_encoder.pkl"
	else:
		model_dir = f"{BASE_DIR}/models/saved_models/level2_classifier/phobert_level2/phobert_level2_model"
		label_path = f"{model_dir}/label_encoder.pkl"

	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = AutoModelForSequenceClassification.from_pretrained(model_dir)

	with open(label_path, 'rb') as f:
		label_encoder = pickle.load(f)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	return tokenizer, model, label_encoder, device


def _predict(tokenizer, model, device, texts, max_length: int = 512):
	enc = tokenizer(
		texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
	)
	enc = {k: v.to(device) for k, v in enc.items()}
	with torch.no_grad():
		outputs = model(**enc)
		logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
		pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
	return pred_ids


def evaluate_level(level: int, df: pd.DataFrame):
	print(f"\n🏷️ Evaluate PhoBERT Level {level}...")
	tokenizer, model, label_encoder, device = _load_model_and_label_encoder(level)
	texts = df['text'].fillna('').tolist()
	y_true = df['type_level1'] if level == 1 else df['domain_level2']

	pred_ids = _predict(tokenizer, model, device, texts)
	y_pred = label_encoder.inverse_transform(pred_ids)

	acc = accuracy_score(y_true, y_pred)
	print(f"✅ Level {level} Test Accuracy: {acc:.4f}")
	print(classification_report(y_true, y_pred))
	return acc


def main(test_path: str = None):
	if test_path is None:
		test_path = f"{BASE_DIR}/data/processed/dataset_splits/test.csv"
	if not os.path.exists(test_path):
		raise FileNotFoundError(f"Không tìm thấy test.csv tại: {test_path}")

	df = pd.read_csv(test_path, encoding='utf-8')
	acc1 = evaluate_level(1, df)
	acc2 = evaluate_level(2, df)
	print("\n🎉 EVALUATION DONE")
	print(f"Level 1 Acc: {acc1:.4f}")
	print(f"Level 2 Acc: {acc2:.4f}")


if __name__ == "__main__":
	main()


