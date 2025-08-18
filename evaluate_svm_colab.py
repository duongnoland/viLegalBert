#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = "/content/viLegalBert"

def evaluate_models(test_path: str = None):
    if test_path is None:
        test_path = f"{BASE_DIR}/data/processed/dataset_splits/test.csv"

    print("📊 Evaluating SVM models on test set...")
    df = pd.read_csv(test_path, encoding='utf-8')
    X_test = df['text'].fillna('')
    y_test_level1 = df['type_level1']
    y_test_level2 = df['domain_level2']

    # Level 1
    level1_path = f"{BASE_DIR}/models/saved_models/level1_classifier/svm_level1/svm_level1_model.pkl"
    with open(level1_path, 'rb') as f:
        level1_data = pickle.load(f)
    v1 = level1_data['vectorizer']; fs1 = level1_data['feature_selector']; m1 = level1_data['model']
    X1 = fs1.transform(v1.transform(X_test))
    y1_pred = m1.predict(X1)
    acc1 = accuracy_score(y_test_level1, y1_pred)
    print(f"🏷️ Level 1 Test Accuracy: {acc1:.4f}")
    print("📋 Level 1 report:\n", classification_report(y_test_level1, y1_pred))

    # Level 2
    level2_path = f"{BASE_DIR}/models/saved_models/level2_classifier/svm_level2/svm_level2_model.pkl"
    with open(level2_path, 'rb') as f:
        level2_data = pickle.load(f)
    v2 = level2_data['vectorizer']; fs2 = level2_data['feature_selector']; m2 = level2_data['model']
    X2 = fs2.transform(v2.transform(X_test))
    y2_pred = m2.predict(X2)
    acc2 = accuracy_score(y_test_level2, y2_pred)
    print(f"🏷️ Level 2 Test Accuracy: {acc2:.4f}")
    print("📋 Level 2 report:\n", classification_report(y_test_level2, y2_pred))

    # Save results
    results = {
        'level1': {'accuracy': acc1, 'gpu_optimized': level1_data.get('gpu_optimized', False)},
        'level2': {'accuracy': acc2, 'gpu_optimized': level2_data.get('gpu_optimized', False)},
    }
    out_dir = f"{BASE_DIR}/results/evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/svm_evaluation_results.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"💾 Saved results to: {out_path}")
    return results

if __name__ == "__main__":
    evaluate_models()