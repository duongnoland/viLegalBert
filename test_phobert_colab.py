#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = "/content/viLegalBert"


def _load(level: int):
	if level == 1:
		model_dir = f"{BASE_DIR}/models/saved_models/level1_classifier/phobert_level1/phobert_level1_model"
	else:
		model_dir = f"{BASE_DIR}/models/saved_models/level2_classifier/phobert_level2/phobert_level2_model"

	label_path = f"{model_dir}/label_encoder.pkl"
	if not os.path.exists(model_dir):
		raise FileNotFoundError(f"Không tìm thấy thư mục model: {model_dir}")
	if not os.path.exists(label_path):
		raise FileNotFoundError(f"Không tìm thấy label_encoder.pkl tại: {label_path}")

	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	model = AutoModelForSequenceClassification.from_pretrained(model_dir)
	with open(label_path, 'rb') as f:
		label_encoder = pickle.load(f)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	return tokenizer, model, label_encoder, device


def _predict(tokenizer, model, device, texts, max_length: int = 512):
	enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
	enc = {k: v.to(device) for k, v in enc.items()}
	with torch.no_grad():
		outputs = model(**enc)
		logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
		pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
	return pred_ids


def predict_texts(texts):
	# Level 1
	tok1, mdl1, le1, dev = _load(1)
	ids1 = _predict(tok1, mdl1, dev, texts)
	lvl1 = le1.inverse_transform(ids1)

	# Level 2
	tok2, mdl2, le2, dev = _load(2)
	ids2 = _predict(tok2, mdl2, dev, texts)
	lvl2 = le2.inverse_transform(ids2)

	return list(zip(texts, lvl1, lvl2))


def main():
	parser = argparse.ArgumentParser(description="Quick test PhoBERT on Colab")
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


