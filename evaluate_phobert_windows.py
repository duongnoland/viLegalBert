#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path
from typing import Optional, List

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# Avoid tokenizer parallelism overhead/warnings on CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Base directory inferred from script location by default
BASE_DIR: Path = Path(__file__).resolve().parent


def _load_model_and_label_encoder(base_dir_path: Path, level: int):
    if level == 1:
        model_dir = base_dir_path / "models" / "saved_models" / "level1_classifier" / "phobert_level1" / "phobert_level1_model"
    else:
        model_dir = base_dir_path / "models" / "saved_models" / "level2_classifier" / "phobert_level2" / "phobert_level2_model"

    label_path = model_dir / "label_encoder.pkl"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {label_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # Force CPU device
    device = torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    with label_path.open('rb') as f:
        label_encoder = pickle.load(f)

    return tokenizer, model, label_encoder, device


def _predict_batches(tokenizer, model, device, texts: List[str], batch_size: int = 8, max_length: int = 256):
    pred_ids_all = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
            pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
            pred_ids_all.append(pred_ids)
    import numpy as np
    return list(np.concatenate(pred_ids_all, axis=0))


def evaluate_level(base_dir_path: Path, level: int, df: pd.DataFrame, batch_size: int = 8, max_length: int = 256) -> float:
    print(f"\nEvaluate PhoBERT Level {level}...")
    tokenizer, model, label_encoder, device = _load_model_and_label_encoder(base_dir_path, level)
    texts = df['text'].fillna('').tolist()
    y_true = df['type_level1'] if level == 1 else df['domain_level2']

    pred_ids = _predict_batches(tokenizer, model, device, texts, batch_size=batch_size, max_length=max_length)
    y_pred = label_encoder.inverse_transform(pred_ids)

    acc = accuracy_score(y_true, y_pred)
    print(f"Level {level} Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc


def main(test_path: Optional[str] = None, base_dir: Optional[str] = None, batch_size: int = 4, max_length: int = 192, num_threads: Optional[int] = None, interop_threads: Optional[int] = None, levels: Optional[List[int]] = None, limit: Optional[int] = None) -> None:
    if num_threads is not None and num_threads > 0:
        try:
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)
            torch.set_num_threads(num_threads)
        except Exception:
            pass
    if interop_threads is not None and interop_threads > 0:
        try:
            torch.set_num_interop_threads(interop_threads)
        except Exception:
            pass

    base_dir_path = Path(base_dir).resolve() if base_dir else BASE_DIR
    if test_path is None:
        test_csv_path = base_dir_path / "data" / "processed" / "dataset_splits" / "test.csv"
    else:
        test_csv_path = Path(test_path).resolve()

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found at: {test_csv_path}")

    print("Evaluating PhoBERT models on test set...")
    df = pd.read_csv(test_csv_path, encoding='utf-8')
    if limit is not None and limit > 0:
        df = df.head(limit)

    to_run = levels if levels else [1, 2]
    acc1 = acc2 = None
    if 1 in to_run:
        acc1 = evaluate_level(base_dir_path, 1, df, batch_size=batch_size, max_length=max_length)
    if 2 in to_run:
        acc2 = evaluate_level(base_dir_path, 2, df, batch_size=batch_size, max_length=max_length)
    print("\nEVALUATION DONE")
    if acc1 is not None:
        print(f"Level 1 Acc: {acc1:.4f}")
    if acc2 is not None:
        print(f"Level 2 Acc: {acc2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PhoBERT models (Windows-friendly, CPU-only)")
    parser.add_argument("--test-path", type=str, default=None, help="Optional path to test.csv")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Optional repository base directory (defaults to this script's parent directory)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation (default: 4)")
    parser.add_argument("--max-length", type=int, default=192, help="Max sequence length for tokenizer (default: 192)")
    parser.add_argument("--num-threads", type=int, default=None, help="torch.set_num_threads value for CPU")
    parser.add_argument("--interop-threads", type=int, default=None, help="torch.set_num_interop_threads value for CPU")
    parser.add_argument("--levels", type=str, default="1,2", help="Levels to run, comma-separated (e.g., '1', '2', or '1,2')")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples (for quick test)")
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(',') if x.strip() in {"1", "2"}]
    main(
        test_path=args.test_path,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_threads=args.num_threads,
        interop_threads=args.interop_threads,
        levels=levels,
        limit=args.limit,
    ) 