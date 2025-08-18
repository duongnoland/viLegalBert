#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import argparse
import numpy as np

BASE_DIR = "/content/viLegalBert"

def _load_level(path):
	print(f"🔄 Loading model: {path}")
	with open(path, 'rb') as f:
		data = pickle.load(f)
	return data['model'], data['vectorizer'], data['feature_selector']

def _ensure_models():
	l1 = f"{BASE_DIR}/models/saved_models/level1_classifier/svm_level1/svm_level1_model.pkl"
	l2 = f"{BASE_DIR}/models/saved_models/level2_classifier/svm_level2/svm_level2_model.pkl"
	if not os.path.exists(l1) or not os.path.exists(l2):
		print("❌ Chưa có model. Hãy chạy: python main_colab.py")
		sys.exit(1)
	return l1, l2

def predict_texts(texts):
	l1_path, l2_path = _ensure_models()
	m1, v1, fs1 = _load_level(l1_path)
	m2, v2, fs2 = _load_level(l2_path)

	# Level 1
	X1 = fs1.transform(v1.transform(texts))
	y1 = m1.predict(X1)
	p1 = m1.predict_proba(X1)
	c1 = np.max(p1, axis=1)

	# Level 2
	X2 = fs2.transform(v2.transform(texts))
	y2 = m2.predict(X2)
	p2 = m2.predict_proba(X2)
	c2 = np.max(p2, axis=1)

	results = []
	for i, t in enumerate(texts):
		results.append({
			'text': t,
			'level1_label': y1[i],
			'level1_conf': float(c1[i]),
			'level2_label': y2[i],
			'level2_conf': float(c2[i]),
		})
	return results

def main():
	parser = argparse.ArgumentParser(description="Quick test SVM (Level1/Level2) trên Colab")
	parser.add_argument("--text", type=str, help="Văn bản cần phân loại")
	args = parser.parse_args()

	if args.text:
		texts = [args.text]
	else:
		texts = [
			"Nghị định về quy định chi tiết thi hành một số điều của Luật Doanh nghiệp",
			"Thông tư hướng dẫn thực hiện các quy định về thuế thu nhập cá nhân",
			"Quyết định về việc phê duyệt quy hoạch giao thông vận tải",
		]

	results = predict_texts(texts)
	for i, r in enumerate(results, 1):
		sample = r['text'][:120] + ("..." if len(r['text']) > 120 else "")
		print(f"\n📝 Sample {i}: {sample}")
		print(f"🏷️ Level 1: {r['level1_label']} (conf: {r['level1_conf']:.3f})")
		print(f"🏷️ Level 2: {r['level2_label']} (conf: {r['level2_conf']:.3f})")

if __name__ == "__main__":
	main()