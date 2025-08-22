#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Resolve repository base directory based on this file location
BASE_DIR: Path = Path(__file__).resolve().parent

def evaluate_models(test_path: Optional[str] = None, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate SVM Level 1 and Level 2 models on the test set.

    Parameters
    ----------
    test_path: Optional[str]
        Optional path to the CSV test dataset. If not provided, defaults to
        <BASE_DIR>/data/processed/dataset_splits/test.csv
    base_dir: Optional[str]
        Optional repository base directory. If not provided, inferred from this file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing accuracy and metadata for each level.
    """
    base_dir_path = Path(base_dir).resolve() if base_dir else BASE_DIR

    if test_path is None:
        test_csv_path = base_dir_path / "data" / "processed" / "dataset_splits" / "test.csv"
    else:
        test_csv_path = Path(test_path).resolve()

    print("Evaluating SVM models on test set...")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found at: {test_csv_path}")

    df = pd.read_csv(test_csv_path, encoding="utf-8")
    X_test = df["text"].fillna("")
    y_test_level1 = df["type_level1"]
    y_test_level2 = df["domain_level2"]

    # Level 1
    level1_path = base_dir_path / "models" / "saved_models" / "level1_classifier" / "svm_level1" / "svm_level1_model.pkl"
    if not level1_path.exists():
        raise FileNotFoundError(f"Level 1 model pickle not found at: {level1_path}")

    with level1_path.open("rb") as f:
        level1_data = pickle.load(f)
    vectorizer_level1 = level1_data["vectorizer"]
    feature_selector_level1 = level1_data["feature_selector"]
    model_level1 = level1_data["model"]

    X_level1 = feature_selector_level1.transform(vectorizer_level1.transform(X_test))
    y_pred_level1 = model_level1.predict(X_level1)
    acc_level1 = accuracy_score(y_test_level1, y_pred_level1)
    print(f"Level 1 Test Accuracy: {acc_level1:.4f}")
    print("Level 1 classification report:\n", classification_report(y_test_level1, y_pred_level1, zero_division=0))

    # Level 2
    level2_path = base_dir_path / "models" / "saved_models" / "level2_classifier" / "svm_level2" / "svm_level2_model.pkl"
    if not level2_path.exists():
        raise FileNotFoundError(f"Level 2 model pickle not found at: {level2_path}")

    with level2_path.open("rb") as f:
        level2_data = pickle.load(f)
    vectorizer_level2 = level2_data["vectorizer"]
    feature_selector_level2 = level2_data["feature_selector"]
    model_level2 = level2_data["model"]

    X_level2 = feature_selector_level2.transform(vectorizer_level2.transform(X_test))
    y_pred_level2 = model_level2.predict(X_level2)
    acc_level2 = accuracy_score(y_test_level2, y_pred_level2)
    print(f"Level 2 Test Accuracy: {acc_level2:.4f}")
    print("Level 2 classification report:\n", classification_report(y_test_level2, y_pred_level2, zero_division=0))

    # Save results
    results = {
        "level1": {"accuracy": acc_level1, "gpu_optimized": level1_data.get("gpu_optimized", False)},
        "level2": {"accuracy": acc_level2, "gpu_optimized": level2_data.get("gpu_optimized", False)},
    }

    out_dir = base_dir_path / "results" / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "svm_evaluation_results.pkl"
    with out_path.open("wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to: {out_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SVM models (Windows-friendly)")
    parser.add_argument("--test-path", type=str, default=None, help="Optional path to test.csv")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Optional repository base directory (defaults to this script's parent directory)",
    )
    args = parser.parse_args()

    evaluate_models(test_path=args.test_path, base_dir=args.base_dir) 