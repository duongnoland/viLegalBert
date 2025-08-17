#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 viLegalBert - Main Pipeline cho Google Colab (GPU Optimized)
Phân loại văn bản pháp luật Việt Nam với kiến trúc phân cấp 2 tầng
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 🚀 GPU SETUP & DEPENDENCIES
# ============================================================================

def setup_gpu():
    """Setup GPU environment cho Linux"""
    import torch
    
    if torch.cuda.is_available():
        print("🚀 GPU CUDA available!")
        print(f"📊 GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Set default device
        torch.cuda.set_device(0)
        return True
    else:
        print("⚠️ GPU CUDA không available, sử dụng CPU")
        return False

def install_deps():
    """Cài đặt dependencies cho Linux"""
    import subprocess
    import sys
    
    packages = [
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} đã có sẵn")
        except ImportError:
            print(f"📦 Cài đặt {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} đã cài đặt xong")

# Import sau khi cài đặt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ============================================================================
# 🏗️ PROJECT STRUCTURE
# ============================================================================

def create_dirs():
    """Tạo thư mục cho Linux từ /content/viLegalBert"""
    import os
    
    # Base directory cho Google Colab
    base_dir = "/content/viLegalBert"
    
    dirs = [
        f"{base_dir}/models/saved_models/level1_classifier/svm_level1",
        f"{base_dir}/models/saved_models/level2_classifier/svm_level2",
        f"{base_dir}/data/processed/dataset_splits"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Đã tạo thư mục: {dir_path}")

# ============================================================================
# 📊 DATASET LOADING
# ============================================================================

def check_splits():
    """Kiểm tra dataset splits có sẵn cho Linux từ /content/viLegalBert"""
    import os
    
    # Base directory cho Google Colab
    base_dir = "/content/viLegalBert"
    
    splits_dir = f"{base_dir}/data/processed/dataset_splits"
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        # Load và hiển thị thông tin splits
        import pandas as pd
        train_df = pd.read_csv(train_path, encoding='utf-8')
        val_df = pd.read_csv(val_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')
        
        print(f"✅ Dataset splits đã có sẵn:")
        print(f"📊 Train set: {len(train_df)} samples")
        print(f"📊 Validation set: {len(val_df)} samples")
        print(f"📊 Test set: {len(test_df)} samples")
        return True
    else:
        print("⚠️ Dataset splits chưa có, sẽ tạo mới...")
        return False

# ============================================================================
# 🏋️ SVM TRAINER
# ============================================================================

class SVMTrainer:
    """Trainer cho mô hình SVM với GPU optimization"""
    
    def __init__(self):
        self.use_gpu = setup_gpu()
        
        # Cấu hình tối ưu cho GPU/CPU
        if self.use_gpu:
            self.config = {
                'max_features': 15000,
                'k_best': 8000,
                'cv': 5,
                'verbose': 2
            }
        else:
            self.config = {
                'max_features': 10000,
                'k_best': 5000,
                'cv': 3,
                'verbose': 1
            }
        
        self.models = {}
        self.vectorizers = {}
        self.feature_selectors = {}
        
        print(f"🚀 SVMTrainer - GPU: {'✅' if self.use_gpu else '❌'}")
    
    def train_level1(self, data_path):
        """Training cho Level 1"""
        print("🏷️ Training Level 1...")
        
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        X = df['text'].fillna('')
        y = df['type_level1']
        
        # Chia data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # TF-IDF + Feature Selection
        vectorizer = TfidfVectorizer(max_features=self.config['max_features'], ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        
        feature_selector = SelectKBest(chi2, k=self.config['k_best'])
        X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_train)
        X_val_selected = feature_selector.transform(X_val_tfidf)
        
        # Training với hyperparameter tuning nếu có GPU
        if self.use_gpu:
            param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
            grid_search = GridSearchCV(SVC(random_state=42, probability=True), param_grid, 
                                     cv=self.config['cv'], n_jobs=-1, verbose=self.config['verbose'])
            grid_search.fit(X_train_selected, y_train)
            svm = grid_search.best_estimator_
            print(f"✅ Best params: {grid_search.best_params_}")
        else:
            svm = SVC(kernel='rbf', random_state=42, probability=True)
            svm.fit(X_train_selected, y_train)
        
        # Evaluation
        y_pred = svm.predict(X_val_selected)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"✅ Level 1 Accuracy: {accuracy:.4f}")
        print(classification_report(y_val, y_pred))
        
        # Lưu model
        self.models['level1'] = svm
        self.vectorizers['level1'] = vectorizer
        self.feature_selectors['level1'] = feature_selector
        
        model_path = "models/saved_models/level1_classifier/svm_level1/svm_level1_model.pkl"
        model_data = {
            'model': svm, 'vectorizer': vectorizer, 'feature_selector': feature_selector,
            'config': self.config, 'gpu_optimized': self.use_gpu
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model đã lưu: {model_path}")
        return {'accuracy': accuracy, 'model_path': model_path, 'gpu_optimized': self.use_gpu}
    
    def train_level2(self, data_path):
        """Training cho Level 2"""
        print("🏷️ Training Level 2...")
        
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        X = df['text'].fillna('')
        y = df['domain_level2']
        
        # Chia data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # TF-IDF + Feature Selection
        vectorizer = TfidfVectorizer(max_features=self.config['max_features'], ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        
        feature_selector = SelectKBest(chi2, k=self.config['k_best'])
        X_train_selected = feature_selector.fit_transform(X_train_tfidf, y_train)
        X_val_selected = feature_selector.transform(X_val_tfidf)
        
        # Training
        if self.use_gpu:
            param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
            grid_search = GridSearchCV(SVC(random_state=42, probability=True), param_grid, 
                                     cv=self.config['cv'], n_jobs=-1, verbose=self.config['verbose'])
            grid_search.fit(X_train_selected, y_train)
            svm = grid_search.best_estimator_
        else:
            svm = SVC(kernel='rbf', random_state=42, probability=True)
            svm.fit(X_train_selected, y_train)
        
        # Evaluation
        y_pred = svm.predict(X_val_selected)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"✅ Level 2 Accuracy: {accuracy:.4f}")
        print(classification_report(y_val, y_pred))
        
        # Lưu model
        self.models['level2'] = svm
        self.vectorizers['level2'] = vectorizer
        self.feature_selectors['level2'] = feature_selector
        
        model_path = "models/saved_models/level2_classifier/svm_level2/svm_level2_model.pkl"
        model_data = {
            'model': svm, 'vectorizer': vectorizer, 'feature_selector': feature_selector,
            'config': self.config, 'gpu_optimized': self.use_gpu
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model đã lưu: {model_path}")
        return {'accuracy': accuracy, 'model_path': model_path, 'gpu_optimized': self.use_gpu}

# ============================================================================
# 📊 EVALUATION
# ============================================================================

def evaluate_models(test_path):
    """Đánh giá models trên test set"""
    print("📊 Đánh giá models...")
    
    # Load test data
    test_df = pd.read_csv(test_path, encoding='utf-8')
    X_test = test_df['text'].fillna('')
    y_test_level1 = test_df['type_level1']
    y_test_level2 = test_df['domain_level2']
    
    try:
        # Load và evaluate Level 1
        with open("models/saved_models/level1_classifier/svm_level1/svm_level1_model.pkl", 'rb') as f:
            level1_data = pickle.load(f)
        
        level1_model = level1_data['model']
        level1_vectorizer = level1_data['vectorizer']
        level1_feature_selector = level1_data['feature_selector']
        
        X_test_level1 = level1_vectorizer.transform(X_test)
        X_test_level1_selected = level1_feature_selector.transform(X_test_level1)
        y_pred_level1 = level1_model.predict(X_test_level1_selected)
        
        accuracy_level1 = accuracy_score(y_test_level1, y_pred_level1)
        print(f"🏷️ Level 1 Test Accuracy: {accuracy_level1:.4f}")
        
        # Load và evaluate Level 2
        with open("models/saved_models/level2_classifier/svm_level2/svm_level2_model.pkl", 'rb') as f:
            level2_data = pickle.load(f)
        
        level2_model = level2_data['model']
        level2_vectorizer = level2_data['vectorizer']
        level2_feature_selector = level2_data['feature_selector']
        
        X_test_level2 = level2_vectorizer.transform(X_test)
        X_test_level2_selected = level2_feature_selector.transform(X_test_level2)
        y_pred_level2 = level2_model.predict(X_test_level2_selected)
        
        accuracy_level2 = accuracy_score(y_test_level2, y_pred_level2)
        print(f"🏷️ Level 2 Test Accuracy: {accuracy_level2:.4f}")
        
        # Lưu kết quả
        results = {
            'level1': {'accuracy': accuracy_level1, 'gpu_optimized': level1_data.get('gpu_optimized', False)},
            'level2': {'accuracy': accuracy_level2, 'gpu_optimized': level2_data.get('gpu_optimized', False)}
        }
        
        results_path = "results/evaluation_results/svm_evaluation_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Kết quả đã lưu: {results_path}")
        return results
        
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá: {e}")
        return None

# ============================================================================
# 🚀 MAIN PIPELINE
# ============================================================================

def main():
    """Hàm chính chạy pipeline"""
    print("🚀 VILEGALBERT PIPELINE - GPU OPTIMIZED")
    print("=" * 60)
    
    # Base directory cho Google Colab
    base_dir = "/content/viLegalBert"
    
    # Bước 1: GPU setup
    print("\n🚀 BƯỚC 1: GPU SETUP")
    gpu_available = setup_gpu()
    
    # Bước 2: Cài đặt dependencies
    print("\n📦 BƯỚC 2: CÀI ĐẶT DEPENDENCIES")
    install_deps()
    
    # Bước 3: Tạo thư mục
    print("\n🏗️ BƯỚC 3: TẠO THƯ MỤC")
    create_dirs()
    
    # Bước 4: Kiểm tra splits
    print("\n🔄 BƯỚC 4: KIỂM TRA SPLITS")
    if not check_splits():
        print("❌ Pipeline dừng do không có dataset splits")
        return
    
    # Bước 5: Training SVM
    print("\n🏋️ BƯỚC 5: TRAINING SVM")
    trainer = SVMTrainer()
    
    dataset_path = f"{base_dir}/data/processed/hierarchical_legal_dataset.csv"
    results_level1 = trainer.train_level1(dataset_path)
    results_level2 = trainer.train_level2(dataset_path)
    
    # Bước 6: Evaluation
    print("\n📊 BƯỚC 6: EVALUATION")
    evaluate_models(f"{base_dir}/data/processed/dataset_splits/test.csv")
    
    # Tóm tắt
    print("\n🎉 PIPELINE HOÀN THÀNH!")
    print(f"📊 Dataset: {dataset_path}")
    print(f"🏷️ Level 1 Accuracy: {results_level1['accuracy']:.4f}")
    print(f"🏷️ Level 2 Accuracy: {results_level2['accuracy']:.4f}")
    print(f"🚀 GPU Status: {'✅ Available' if gpu_available else '❌ Not Available'}")

if __name__ == "__main__":
    main() 