#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab comparison & report generator for Vietnamese legal text classification.

Outputs:
- Tables:
  - Data stats per class (Level 1 & 2)
  - Main results (Acc / Macro-F1 / Weighted-F1) for SVM, BiLSTM, PhoBERT
  - PhoBERT ablation (max_length, epochs, class_weights)
- Figures:
  - Confusion matrices (Level 1/2) for each model
  - Learning curves (PhoBERT) from trainer_state.json
  - Length distribution (tokens/chars)
  - Optional UMAP/TSNE embeddings (sampled)

Assumes repository mounted at /content/viLegalBert on Colab.
"""

import os
import json
import pickle
import warnings
import argparse
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.manifold import TSNE

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

BASE_DIR = "/content/viLegalBert"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "report")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Try to enable rich display if running in notebook/Colab
try:
    from IPython.display import display, HTML
    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False


# ==========================
# Data loading & statistics
# ==========================
def load_splits():
    ds_dir = os.path.join(BASE_DIR, "data", "processed", "dataset_splits")
    train = pd.read_csv(os.path.join(ds_dir, "train.csv"), encoding='utf-8')
    val = pd.read_csv(os.path.join(ds_dir, "validation.csv"), encoding='utf-8')
    test = pd.read_csv(os.path.join(ds_dir, "test.csv"), encoding='utf-8')
    return train, val, test


def data_stats_per_class(df: pd.DataFrame, label_col: str, title: str):
    cnt = df[label_col].value_counts().rename_axis('label').reset_index(name='count')
    cnt.to_csv(os.path.join(RESULTS_DIR, f"stats_{label_col}.csv"), index=False)
    print(f"\n[Stats] {title} ({label_col}) - top 10:")
    print(cnt.head(10))
    return cnt


def plot_length_distribution(df: pd.DataFrame, title: str, show_inline: bool = False):
    texts = df['text'].fillna('')
    char_len = texts.apply(len)
    # Rough token length by whitespace split
    token_len = texts.apply(lambda t: len(str(t).split()))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); sns.histplot(char_len, bins=50); plt.title(f"Char length - {title}")
    plt.subplot(1,2,2); sns.histplot(token_len, bins=50); plt.title(f"Token length - {title}")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"length_distribution_{title}.png")
    plt.savefig(out, dpi=150)
    if show_inline:
        plt.show()
    plt.close()
    print(f"Saved: {out}")


# ==========================
# SVM baseline (TF-IDF)
# ==========================
def train_eval_svm(train_df, test_df, label_col: str, model_tag: str, show_inline: bool = False):
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_df['text'].fillna(''))
    X_test = vectorizer.transform(test_df['text'].fillna(''))
    y_train = train_df[label_col]
    y_test = test_df[label_col]

    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Confusion matrix (limit to top-k labels by support for readability)
    labels_order = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    plt.figure(figsize=(10,8))
    annot_on = len(labels_order) <= 30
    sns.heatmap(
        cm,
        cmap='Blues', cbar=False,
        annot=annot_on, fmt='d',
        xticklabels=labels_order, yticklabels=labels_order
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(f"Confusion Matrix - SVM - {label_col}")
    plt.xlabel("Pred"); plt.ylabel("True")
    out = os.path.join(RESULTS_DIR, f"cm_svm_{model_tag}_{label_col}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    if show_inline:
        plt.show()
    plt.close(); print(f"Saved: {out}")

    return {
        'model': 'SVM',
        'label_col': label_col,
        'acc': acc,
        'macro_f1': f1_macro,
        'weighted_f1': f1_weighted,
    }


# ==========================
# Evaluate existing BiLSTM models
# ==========================
def eval_bilstm_level(level: int, test_df: pd.DataFrame, show_inline: bool = False):
    # Import evaluate_bilstm_colab by path to avoid PYTHONPATH issues on Colab
    import importlib.util
    mod_path = os.path.join(BASE_DIR, 'evaluate_bilstm_colab.py')
    spec = importlib.util.spec_from_file_location('evaluate_bilstm_colab', mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = mod._load_artifacts(level)
    texts = test_df['text'].fillna('')
    y_true = test_df['type_level1'] if level == 1 else test_df['domain_level2']
    model, X_t, lbl, device = mod._rebuild_model_and_inputs(data, texts)
    with torch.no_grad():
        logits = model(X_t)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_pred = lbl.inverse_transform(pred)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    labels_order = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    plt.figure(figsize=(10,8))
    annot_on = len(labels_order) <= 30
    sns.heatmap(
        cm,
        cmap='Greens', cbar=False,
        annot=annot_on, fmt='d',
        xticklabels=labels_order, yticklabels=labels_order
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(f"Confusion Matrix - BiLSTM - Level {level}")
    plt.xlabel("Pred"); plt.ylabel("True")
    out = os.path.join(RESULTS_DIR, f"cm_bilstm_level{level}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    if show_inline:
        plt.show()
    plt.close(); print(f"Saved: {out}")

    return acc, f1_macro, f1_weighted


# ==========================
# Evaluate existing PhoBERT models
# ==========================
def _load_phobert(level: int):
    if level == 1:
        model_dir = f"{BASE_DIR}/models/saved_models/level1_classifier/phobert_level1/phobert_level1_model"
    else:
        model_dir = f"{BASE_DIR}/models/saved_models/level2_classifier/phobert_level2/phobert_level2_model"
    label_path = os.path.join(model_dir, 'label_encoder.pkl')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    with open(label_path, 'rb') as f:
        label_encoder = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, label_encoder, device


def eval_phobert_level(level: int, test_df: pd.DataFrame, show_inline: bool = False):
    tok, mdl, le, dev = _load_phobert(level)
    texts = test_df['text'].fillna('').tolist()
    y_true = test_df['type_level1'] if level == 1 else test_df['domain_level2']
    enc = tok(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        logits = mdl(**enc).logits
        pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
    y_pred = le.inverse_transform(pred_ids)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    labels_order = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    plt.figure(figsize=(10,8))
    annot_on = len(labels_order) <= 30
    sns.heatmap(
        cm,
        cmap='Reds', cbar=False,
        annot=annot_on, fmt='d',
        xticklabels=labels_order, yticklabels=labels_order
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(f"Confusion Matrix - PhoBERT - Level {level}")
    plt.xlabel("Pred"); plt.ylabel("True")
    out = os.path.join(RESULTS_DIR, f"cm_phobert_level{level}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    if show_inline:
        plt.show()
    plt.close(); print(f"Saved: {out}")

    return acc, f1_macro, f1_weighted


# ==========================
# PhoBERT Ablation (small grid)
# ==========================
def phobert_ablation(train_df, val_df, grid=None, level: int = 1):
    """Run a small ablation for PhoBERT. Warning: training time-consuming on Colab.
    grid = list of dicts: {'max_length': int, 'num_epochs': int, 'use_class_weights': bool}
    """
    if grid is None:
        grid = [
            {'max_length': 256, 'num_epochs': 3, 'use_class_weights': False},
            {'max_length': 512, 'num_epochs': 20, 'use_class_weights': False},
        ]

    # Lazy import trainer to avoid heavy deps at module import
    from phobert_colab import PhoBERTTrainer

    results = []
    for cfg in grid:
        print(f"\n[ABLT] {cfg}")
        trainer = PhoBERTTrainer()
        # override config
        trainer.config['max_length'] = cfg['max_length']
        trainer.config['num_epochs'] = cfg['num_epochs']
        trainer.config['use_class_weights'] = cfg['use_class_weights']

        # persist temporary CSVs (trainer expects file paths)
        tmp_train = os.path.join(RESULTS_DIR, f"_ablt_train_level{level}.csv")
        tmp_val = os.path.join(RESULTS_DIR, f"_ablt_val_level{level}.csv")
        train_df[['text', 'type_level1', 'domain_level2']].to_csv(tmp_train, index=False)
        val_df[['text', 'type_level1', 'domain_level2']].to_csv(tmp_val, index=False)

        if level == 1:
            out = trainer.train_level1(tmp_train, tmp_val)
        else:
            out = trainer.train_level2(tmp_train, tmp_val)

        results.append({
            'level': level,
            'max_length': cfg['max_length'],
            'num_epochs': cfg['num_epochs'],
            'use_class_weights': cfg['use_class_weights'],
            # Note: add eval metrics if returned
            'eval': out.get('eval_results', {}),
            'model_path': out.get('model_path', ''),
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, f"ablation_phobert_level{level}.csv"), index=False)
    print(df)
    return df


# ==========================
# Learning curves (PhoBERT)
# ==========================
def plot_phobert_learning_curves(model_dir: str, tag: str, show_inline: bool = False):
    state_path = os.path.join(model_dir, 'trainer_state.json')
    if not os.path.exists(state_path):
        print(f"No trainer_state.json at {model_dir}")
        return
    with open(state_path, 'r') as f:
        state = json.load(f)
    logs = state.get('log_history', [])
    df = pd.DataFrame(logs)
    plt.figure(figsize=(8,5))
    if 'loss' in df:
        plt.plot(df.get('step', range(len(df))), df['loss'], label='train_loss', alpha=0.7)
    if 'eval_loss' in df:
        plt.plot(df.get('step', range(len(df))), df['eval_loss'], label='eval_loss', alpha=0.7)
    plt.xlabel('step'); plt.ylabel('loss'); plt.title(f'Learning Curves - {tag}')
    plt.legend(); plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"learning_curves_{tag}.png")
    plt.savefig(out, dpi=150)
    if show_inline:
        plt.show()
    plt.close(); print(f"Saved: {out}")


# ==========================
# Embedding visualization (UMAP/TSNE)
# ==========================
def plot_embeddings_umap_tsne(df: pd.DataFrame, sample_size: int = 1000, tag: str = 'phobert', show_inline: bool = False):
    try:
        import umap
        has_umap = True
    except Exception:
        has_umap = False

    texts = df['text'].fillna('')
    labels = df['type_level1']  # or choose domain_level2
    if len(texts) > sample_size:
        texts = texts.sample(sample_size, random_state=42)
        labels = labels.loc[texts.index]

    tok = AutoTokenizer.from_pretrained('vinai/phobert-base')
    mdl = AutoModel.from_pretrained('vinai/phobert-base').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    mdl.eval()

    device = next(mdl.parameters()).device
    enc = tok(texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc)
        # Use [CLS] embedding (first token)
        emb = out.last_hidden_state[:, 0, :].detach().cpu().numpy()

    if has_umap:
        reducer = umap.UMAP(n_components=2, random_state=42)
        Z = reducer.fit_transform(emb)
        method = 'UMAP'
    else:
        Z = TSNE(n_components=2, random_state=42, init='pca').fit_transform(emb)
        method = 't-SNE'

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=labels, s=12, palette='tab20', legend=False)
    plt.title(f"{method} embeddings - {tag}")
    out = os.path.join(RESULTS_DIR, f"embeddings_{tag}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    if show_inline:
        plt.show()
    plt.close(); print(f"Saved: {out}")


# ==========================
# Orchestrator
# ==========================
def main():
    parser = argparse.ArgumentParser(description="Colab compare & report")
    parser.add_argument("--show_ui", action="store_true", help="Hiển thị bảng/hình trực tiếp trong Colab")
    parser.add_argument("--run_ablation", action="store_true", help="Chạy ablation nhỏ cho PhoBERT")
    parser.add_argument("--embed_sample", type=int, default=1000, help="Số mẫu để vẽ UMAP/t-SNE")
    args, _ = parser.parse_known_args()

    show_inline = bool(args.show_ui and IN_NOTEBOOK)

    print("== Load dataset splits ==")
    train, val, test = load_splits()

    # Data stats
    stats_l1 = data_stats_per_class(train, 'type_level1', 'Train')
    stats_l2 = data_stats_per_class(train, 'domain_level2', 'Train')
    plot_length_distribution(train, 'Train', show_inline=show_inline)
    if show_inline:
        try:
            display(HTML("<h3>Thống kê lớp - Level 1</h3>")); display(stats_l1.head(20))
            display(HTML("<h3>Thống kê lớp - Level 2</h3>")); display(stats_l2.head(20))
        except Exception:
            pass

    # SVM baselines
    print("\n== Train & Evaluate SVM ==")
    res = []
    res.append(train_eval_svm(train, test, 'type_level1', 'level1', show_inline=show_inline))
    res.append(train_eval_svm(train, test, 'domain_level2', 'level2', show_inline=show_inline))

    # BiLSTM evaluation (requires saved models)
    print("\n== Evaluate BiLSTM (saved models) ==")
    try:
        acc1, f1m1, f1w1 = eval_bilstm_level(1, test, show_inline=show_inline)
        res.append({'model': 'BiLSTM', 'label_col': 'type_level1', 'acc': acc1, 'macro_f1': f1m1, 'weighted_f1': f1w1})
    except Exception as e:
        print(f"BiLSTM Level 1 eval failed: {e}")
    try:
        acc2, f1m2, f1w2 = eval_bilstm_level(2, test, show_inline=show_inline)
        res.append({'model': 'BiLSTM', 'label_col': 'domain_level2', 'acc': acc2, 'macro_f1': f1m2, 'weighted_f1': f1w2})
    except Exception as e:
        print(f"BiLSTM Level 2 eval failed: {e}")

    # PhoBERT evaluation (requires saved models)
    print("\n== Evaluate PhoBERT (saved models) ==")
    try:
        acc1, f1m1, f1w1 = eval_phobert_level(1, test, show_inline=show_inline)
        res.append({'model': 'PhoBERT', 'label_col': 'type_level1', 'acc': acc1, 'macro_f1': f1m1, 'weighted_f1': f1w1})
    except Exception as e:
        print(f"PhoBERT Level 1 eval failed: {e}")
    try:
        acc2, f1m2, f1w2 = eval_phobert_level(2, test, show_inline=show_inline)
        res.append({'model': 'PhoBERT', 'label_col': 'domain_level2', 'acc': acc2, 'macro_f1': f1m2, 'weighted_f1': f1w2})
    except Exception as e:
        print(f"PhoBERT Level 2 eval failed: {e}")

    # Results table
    res_df = pd.DataFrame(res)
    res_csv = os.path.join(RESULTS_DIR, "main_results.csv")
    res_df.to_csv(res_csv, index=False)
    print("\n== Main Results ==")
    print(res_df)
    print(f"Saved: {res_csv}")
    if show_inline:
        try:
            display(HTML("<h3>Kết quả chính</h3>")); display(res_df)
        except Exception:
            pass

    # Learning curves for PhoBERT (if available)
    plot_phobert_learning_curves(
        os.path.join(BASE_DIR, 'models', 'saved_models', 'level1_classifier', 'phobert_level1', 'phobert_level1_model'),
        tag='PhoBERT_Level1', show_inline=show_inline
    )
    plot_phobert_learning_curves(
        os.path.join(BASE_DIR, 'models', 'saved_models', 'level2_classifier', 'phobert_level2', 'phobert_level2_model'),
        tag='PhoBERT_Level2', show_inline=show_inline
    )

    # Optional: embeddings visualization (sampled)
    try:
        plot_embeddings_umap_tsne(test, sample_size=int(args.embed_sample), tag='PhoBERT_test_level1', show_inline=show_inline)
    except Exception as e:
        print(f"Embedding plot skipped: {e}")

    # PhoBERT Ablation (optional)
    if args.run_ablation:
        print("\n== PhoBERT Ablation (small grid) ==")
        ablt_df = phobert_ablation(train, val, level=1)
        if show_inline:
            try:
                display(HTML("<h3>Ablation PhoBERT (Level 1)</h3>")); display(ablt_df)
            except Exception:
                pass

    print("\nAll done. Outputs saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()


