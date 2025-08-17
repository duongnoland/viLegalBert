#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 viLegalBert - Main Pipeline cho Google Colab (Dataset Có Sẵn)
Phân loại văn bản pháp luật Việt Nam với kiến trúc phân cấp 2 tầng
"""

import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 📦 INSTALL & IMPORT DEPENDENCIES
# ============================================================================

def install_dependencies():
    """Cài đặt các thư viện cần thiết"""
    try:
        import sklearn
        print("✅ scikit-learn đã được cài đặt")
    except ImportError:
        print("📦 Đang cài đặt scikit-learn...")
        os.system("pip install scikit-learn")
    
    try:
        import transformers
        print("✅ transformers đã được cài đặt")
    except ImportError:
        print("📦 Đang cài đặt transformers...")
        os.system("pip install transformers")
    
    try:
        import torch
        print("✅ PyTorch đã được cài đặt")
    except ImportError:
        print("📦 Đang cài đặt PyTorch...")
        os.system("pip install torch")
    
    try:
        import yaml
        print("✅ PyYAML đã được cài đặt")
    except ImportError:
        print("📦 Đang cài đặt PyYAML...")
        os.system("pip install PyYAML")

# Cài đặt dependencies
install_dependencies()

# Import sau khi cài đặt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
import joblib

# ============================================================================
# 🏗️ CẤU TRÚC PROJECT
# ============================================================================

def create_project_structure():
    """Tạo cấu trúc thư mục cho project"""
    directories = [
        'models/saved_models/level1_classifier/svm_level1',
        'models/saved_models/level2_classifier/svm_level2',
        'models/saved_models/level1_classifier/phobert_level1',
        'models/saved_models/level2_classifier/phobert_level2',
        'models/saved_models/level1_classifier/bilstm_level1',
        'models/saved_models/level2_classifier/bilstm_level2',
        'models/saved_models/hierarchical_models',
        'results/training_results',
        'results/evaluation_results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Tạo thư mục: {directory}")

# ============================================================================
# 📊 DATASET LOADING (Có Sẵn)
# ============================================================================

def load_existing_dataset(dataset_path: str = "data/processed/hierarchical_legal_dataset.csv"):
    """Load dataset có sẵn"""
    print("📊 Loading dataset có sẵn...")
    
    try:
        # Kiểm tra file dataset
        if not Path(dataset_path).exists():
            print(f"❌ Không tìm thấy dataset: {dataset_path}")
            print("🔍 Tìm kiếm dataset trong các thư mục...")
            
            # Tìm kiếm dataset trong các thư mục khác
            possible_paths = [
                "hierarchical_legal_dataset.csv",
                "data/hierarchical_legal_dataset.csv",
                "dataset.csv",
                "legal_dataset.csv"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    dataset_path = path
                    print(f"✅ Tìm thấy dataset: {dataset_path}")
                    break
            else:
                print("❌ Không tìm thấy dataset nào. Vui lòng upload dataset vào Colab")
                return None
        
        # Load dataset
        df = pd.read_csv(dataset_path, encoding='utf-8')
        print(f"✅ Đã load dataset: {len(df)} samples")
        
        # Hiển thị thông tin dataset
        print(f"\n📈 THÔNG TIN DATASET:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Kiểm tra columns cần thiết
        required_columns = ['text', 'type_level1', 'domain_level2']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Thiếu columns: {missing_columns}")
            print(f"📋 Columns có sẵn: {list(df.columns)}")
            return None
        
        # Hiển thị thống kê
        print(f"\n🏷️ PHÂN LOẠI TẦNG 1 (Loại văn bản):")
        level1_counts = df['type_level1'].value_counts()
        for doc_type, count in level1_counts.items():
            print(f"  - {doc_type}: {count}")
        
        print(f"\n🏷️ PHÂN LOẠI TẦNG 2 (Domain pháp lý):")
        level2_counts = df['domain_level2'].value_counts()
        for domain, count in level2_counts.items():
            print(f"  - {domain}: {count}")
        
        return df
        
    except Exception as e:
        print(f"❌ Lỗi khi load dataset: {e}")
        return None

def check_dataset_splits():
    """Kiểm tra dataset splits có sẵn"""
    print("🔍 Kiểm tra dataset splits...")
    
    splits_dir = "data/processed/dataset_splits"
    train_path = Path(splits_dir) / "train.csv"
    val_path = Path(splits_dir) / "validation.csv"
    test_path = Path(splits_dir) / "test.csv"
    
    if train_path.exists() and val_path.exists() and test_path.exists():
        print("✅ Dataset splits đã có sẵn")
        
        # Load và hiển thị thông tin splits
        train_df = pd.read_csv(train_path, encoding='utf-8')
        val_df = pd.read_csv(val_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')
        
        print(f"📊 Train set: {len(train_df)} samples")
        print(f"📊 Validation set: {len(val_df)} samples")
        print(f"📊 Test set: {len(test_df)} samples")
        
        return True
    else:
        print("⚠️ Dataset splits chưa có, sẽ tạo mới...")
        return False

def create_training_splits_from_existing(dataset_path: str, splits_dir: str):
    """Tạo training splits từ dataset có sẵn"""
    print("🔄 Tạo training splits từ dataset có sẵn...")
    
    # Load dataset
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
    # Chia dữ liệu
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['type_level1'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['type_level1'])
    
    # Lưu các tập
    train_path = Path(splits_dir) / "train.csv"
    val_path = Path(splits_dir) / "validation.csv"
    test_path = Path(splits_dir) / "test.csv"
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    print(f"✅ Train set: {len(train_df)} samples -> {train_path}")
    print(f"✅ Validation set: {len(val_df)} samples -> {val_path}")
    print(f"✅ Test set: {len(test_df)} samples -> {test_path}")

# ============================================================================
# 🏋️ SVM TRAINER
# ============================================================================

class SVMTrainer:
    """Trainer cho mô hình SVM"""
    
    def __init__(self):
        """Khởi tạo trainer"""
        self.config = {
            'feature_extraction': {
                'tfidf': {
                    'max_features': 10000,
                    'min_df': 2,
                    'max_df': 0.95,
                    'ngram_range': [1, 2],
                    'stop_words': None
                }
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale'
            },
            'feature_selection': {
                'k_best': 5000
            }
        }
        self.models = {}
        self.vectorizers = {}
        self.feature_selectors = {}
    
    def train_level1(self, data_path: str):
        """Training cho Level 1 (Loại văn bản)"""
        print("🏷️ Training Level 1 (Loại văn bản)...")
        
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        X = df['text'].fillna('')
        y = df['type_level1']
        
        # Chia train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        print("📊 TF-IDF Vectorization...")
        vectorizer = TfidfVectorizer(
            max_features=self.config['feature_extraction']['tfidf']['max_features'],
            min_df=self.config['feature_extraction']['tfidf']['min_df'],
            max_df=self.config['feature_extraction']['tfidf']['max_df'],
            ngram_range=tuple(self.config['feature_extraction']['tfidf']['ngram_range'])
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        
        # Feature Selection
        print("🔍 Feature Selection...")
        feature_selector = SelectKBest(chi2, k=self.config['feature_selection']['k_best'])
        X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_train)
        X_val_selected = feature_selector.transform(X_val_tfidf)
        
        # SVM Training
        print("🏋️ Training SVM...")
        svm = SVC(
            kernel=self.config['svm']['kernel'],
            C=self.config['svm']['C'],
            gamma=self.config['svm']['gamma'],
            random_state=42,
            probability=True
        )
        
        svm.fit(X_train_selected, y_train)
        
        # Evaluation
        y_pred = svm.predict(X_val_selected)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"✅ Level 1 Training hoàn thành!")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"📊 Classification Report:")
        print(classification_report(y_val, y_pred))
        
        # Lưu model
        self.models['level1'] = svm
        self.vectorizers['level1'] = vectorizer
        self.feature_selectors['level1'] = feature_selector
        
        # Lưu model
        model_path = "models/saved_models/level1_classifier/svm_level1/svm_level1_model.pkl"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': svm,
            'vectorizer': vectorizer,
            'feature_selector': feature_selector
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model đã được lưu: {model_path}")
        
        return {
            'accuracy': accuracy,
            'model_path': model_path,
            'classification_report': classification_report(y_val, y_pred, output_dict=True)
        }
    
    def train_level2(self, data_path: str):
        """Training cho Level 2 (Domain pháp lý)"""
        print("🏷️ Training Level 2 (Domain pháp lý)...")
        
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        X = df['text'].fillna('')
        y = df['domain_level2']
        
        # Chia train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        print("📊 TF-IDF Vectorization...")
        vectorizer = TfidfVectorizer(
            max_features=self.config['feature_extraction']['tfidf']['max_features'],
            min_df=self.config['feature_extraction']['tfidf']['min_df'],
            max_df=self.config['feature_extraction']['tfidf']['max_df'],
            ngram_range=tuple(self.config['feature_extraction']['tfidf']['ngram_range'])
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        
        # Feature Selection
        print("🔍 Feature Selection...")
        feature_selector = SelectKBest(chi2, k=self.config['feature_selection']['k_best'])
        X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_train)
        X_val_selected = feature_selector.transform(X_val_tfidf)
        
        # SVM Training
        print("🏋️ Training SVM...")
        svm = SVC(
            kernel=self.config['svm']['kernel'],
            C=self.config['svm']['C'],
            gamma=self.config['svm']['gamma'],
            random_state=42,
            probability=True
        )
        
        svm.fit(X_train_selected, y_train)
        
        # Evaluation
        y_pred = svm.predict(X_val_selected)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"✅ Level 2 Training hoàn thành!")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"📊 Classification Report:")
        print(classification_report(y_val, y_pred))
        
        # Lưu model
        self.models['level2'] = svm
        self.vectorizers['level2'] = vectorizer
        self.feature_selectors['level2'] = feature_selector
        
        # Lưu model
        model_path = "models/saved_models/level2_classifier/svm_level2/svm_level2_model.pkl"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': svm,
            'vectorizer': vectorizer,
            'feature_selector': feature_selector
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model đã được lưu: {model_path}")
        
        return {
            'accuracy': accuracy,
            'model_path': model_path,
            'classification_report': classification_report(y_val, y_pred, output_dict=True)
        }

# ============================================================================
# 📊 EVALUATION
# ============================================================================

def evaluate_svm_models(test_data_path: str):
    """Đánh giá mô hình SVM trên test set"""
    print("📊 Đánh giá mô hình SVM...")
    
    # Load test data
    test_df = pd.read_csv(test_data_path, encoding='utf-8')
    X_test = test_df['text'].fillna('')
    y_test_level1 = test_df['type_level1']
    y_test_level2 = test_df['domain_level2']
    
    # Load models
    try:
        # Level 1
        with open("models/saved_models/level1_classifier/svm_level1/svm_level1_model.pkl", 'rb') as f:
            level1_data = pickle.load(f)
        
        level1_model = level1_data['model']
        level1_vectorizer = level1_data['vectorizer']
        level1_feature_selector = level1_data['feature_selector']
        
        # Level 2
        with open("models/saved_models/level2_classifier/svm_level2/svm_level2_model.pkl", 'rb') as f:
            level2_data = pickle.load(f)
        
        level2_model = level2_data['model']
        level2_vectorizer = level2_data['vectorizer']
        level2_feature_selector = level2_data['feature_selector']
        
        # Evaluation Level 1
        print("\n🏷️ EVALUATION LEVEL 1 (Loại văn bản):")
        X_test_level1 = level1_vectorizer.transform(X_test)
        X_test_level1_selected = level1_feature_selector.transform(X_test_level1)
        y_pred_level1 = level1_model.predict(X_test_level1_selected)
        
        accuracy_level1 = accuracy_score(y_test_level1, y_pred_level1)
        print(f"📊 Accuracy: {accuracy_level1:.4f}")
        print(f"📊 Classification Report:")
        print(classification_report(y_test_level1, y_pred_level1))
        
        # Evaluation Level 2
        print("\n🏷️ EVALUATION LEVEL 2 (Domain pháp lý):")
        X_test_level2 = level2_vectorizer.transform(X_test)
        X_test_level2_selected = level2_feature_selector.transform(X_test_level2)
        y_pred_level2 = level2_model.predict(X_test_level2_selected)
        
        accuracy_level2 = accuracy_score(y_test_level2, y_pred_level2)
        print(f"📊 Accuracy: {accuracy_level2:.4f}")
        print(f"📊 Classification Report:")
        print(classification_report(y_test_level2, y_pred_level2))
        
        # Lưu kết quả
        results = {
            'level1': {
                'accuracy': accuracy_level1,
                'classification_report': classification_report(y_test_level1, y_pred_level1, output_dict=True)
            },
            'level2': {
                'accuracy': accuracy_level2,
                'classification_report': classification_report(y_test_level2, y_pred_level2, output_dict=True)
            }
        }
        
        results_path = "results/evaluation_results/svm_evaluation_results.pkl"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n💾 Kết quả evaluation đã được lưu: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá: {e}")
        return None

# ============================================================================
# 🚀 MAIN PIPELINE
# ============================================================================

def main():
    """Hàm chính chạy pipeline"""
    print("🚀 KHỞI ĐỘNG VILEGALBERT PIPELINE CHO GOOGLE COLAB!")
    print("📊 SỬ DỤNG DATASET CÓ SẴN")
    print("=" * 80)
    
    # Bước 1: Tạo cấu trúc project
    create_project_structure()
    
    # Bước 2: Load dataset có sẵn
    print("\n📊 BƯỚC 1: LOAD DATASET CÓ SẴN")
    print("-" * 50)
    
    df = load_existing_dataset()
    if df is None:
        print("❌ Không thể load dataset")
        return
    
    # Bước 3: Kiểm tra và tạo dataset splits
    print("\n🔄 BƯỚC 2: KIỂM TRA DATASET SPLITS")
    print("-" * 50)
    
    if not check_dataset_splits():
        # Tạo splits mới từ dataset có sẵn
        dataset_path = "data/processed/hierarchical_legal_dataset.csv"
        splits_dir = "data/processed/dataset_splits"
        create_training_splits_from_existing(dataset_path, splits_dir)
    
    # Bước 4: Training SVM
    print("\n🏋️ BƯỚC 3: TRAINING SVM")
    print("-" * 50)
    
    trainer = SVMTrainer()
    
    # Training Level 1
    results_level1 = trainer.train_level1("data/processed/hierarchical_legal_dataset.csv")
    
    # Training Level 2
    results_level2 = trainer.train_level2("data/processed/hierarchical_legal_dataset.csv")
    
    # Bước 5: Evaluation
    print("\n📊 BƯỚC 4: EVALUATION")
    print("-" * 50)
    
    test_data_path = "data/processed/dataset_splits/test.csv"
    evaluation_results = evaluate_svm_models(test_data_path)
    
    # Tóm tắt kết quả
    print("\n🎉 TÓM TẮT KẾT QUẢ")
    print("=" * 80)
    print(f"📊 Dataset: {len(df)} samples")
    print(f"🏷️ Level 1 Accuracy: {results_level1['accuracy']:.4f}")
    print(f"🏷️ Level 2 Accuracy: {results_level2['accuracy']:.4f}")
    print(f"💾 Models đã được lưu trong thư mục models/")
    print(f"📊 Kết quả evaluation đã được lưu trong thư mục results/")
    
    print("\n✅ PIPELINE HOÀN THÀNH THÀNH CÔNG!")
    print("🚀 Bạn có thể tiếp tục với training PhoBERT, BiLSTM hoặc Ensemble!")

if __name__ == "__main__":
    main() 