#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 viLegalBert - Main Pipeline cho Google Colab
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
        'data/raw',
        'data/processed',
        'data/processed/dataset_splits',
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
# 📊 DATASET CREATION
# ============================================================================

def create_hierarchical_dataset(json_file: str, output_csv: str, target_size: int = 10000) -> pd.DataFrame:
    """Tạo dataset phân cấp 2 tầng từ JSON gốc"""
    print("🔍 Đang load file JSON...")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Load thành công {len(data)} items từ {json_file}")
    except Exception as e:
        print(f"❌ Lỗi khi load JSON: {e}")
        return None
    
    # Lấy mẫu ngẫu nhiên
    if len(data) > target_size:
        data = np.random.choice(data, target_size, replace=False).tolist()
        print(f"📊 Đã lấy mẫu {target_size} items")
    
    # Xử lý từng item
    processed_data = []
    
    for item in data:
        try:
            # Trích xuất thông tin cơ bản
            doc_id = item.get('id', 'unknown')
            ministry = item.get('ministry', 'unknown')
            doc_type = item.get('type', 'unknown')
            name = item.get('name', '')
            chapter_name = item.get('chapter_name', '')
            article = item.get('article', '')
            content = item.get('content', '')
            
            # Làm sạch text
            text = f"{name} {chapter_name} {article} {content}".strip()
            text = ' '.join(text.split())  # Loại bỏ khoảng trắng thừa
            
            # Phân loại Level 1 (Loại văn bản)
            type_level1 = extract_document_type(doc_type, name)
            
            # Phân loại Level 2 (Domain pháp lý)
            domain_level2 = extract_legal_domain(content, name, chapter_name)
            
            # Tính độ dài nội dung
            content_length = len(content) if content else 0
            
            processed_data.append({
                'id': doc_id,
                'text': text,
                'type_level1': type_level1,
                'domain_level2': domain_level2,
                'ministry': ministry,
                'name': name,
                'chapter': chapter_name,
                'article': article,
                'content_length': content_length
            })
            
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý item {item.get('id', 'unknown')}: {e}")
            continue
    
    # Tạo DataFrame
    df = pd.DataFrame(processed_data)
    
    # Lưu dataset
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ Đã lưu dataset vào: {output_csv}")
    
    # Hiển thị thống kê
    print(f"\n📈 THỐNG KÊ DATASET:")
    print(f"Tổng số samples: {len(df)}")
    
    print(f"\n🏷️ PHÂN LOẠI TẦNG 1 (Loại văn bản):")
    level1_counts = df['type_level1'].value_counts()
    for doc_type, count in level1_counts.items():
        print(f"  - {doc_type}: {count}")
    
    print(f"\n🏷️ PHÂN LOẠI TẦNG 2 (Domain pháp lý):")
    level2_counts = df['domain_level2'].value_counts()
    for domain, count in level2_counts.items():
        print(f"  - {domain}: {count}")
    
    return df

def extract_document_type(doc_type: str, name: str) -> str:
    """Trích xuất loại văn bản từ type và name"""
    if not doc_type and not name:
        return "KHÁC"
    
    text = f"{doc_type} {name}".upper()
    
    if any(keyword in text for keyword in ["LUẬT", "LAW"]):
        return "LUẬT"
    elif any(keyword in text for keyword in ["NGHỊ ĐỊNH", "DECREE"]):
        return "NGHỊ ĐỊNH"
    elif any(keyword in text for keyword in ["THÔNG TƯ", "CIRCULAR"]):
        return "THÔNG TƯ"
    elif any(keyword in text for keyword in ["QUYẾT ĐỊNH", "DECISION"]):
        return "QUYẾT ĐỊNH"
    elif any(keyword in text for keyword in ["NGHỊ QUYẾT", "RESOLUTION"]):
        return "NGHỊ QUYẾT"
    elif any(keyword in text for keyword in ["PHÁP LỆNH", "ORDINANCE"]):
        return "PHÁP LỆNH"
    else:
        return "KHÁC"

def extract_legal_domain(content: str, name: str, chapter_name: str) -> str:
    """Trích xuất domain pháp lý từ nội dung"""
    if not content:
        return "KHÁC"
    
    # Kết hợp nội dung để phân tích
    full_text = f"{name} {chapter_name} {content}".upper()
    
    # Mapping các domain pháp lý với từ khóa tiếng Việt
    domain_keywords = {
        "HÌNH SỰ": ["hình sự", "tội phạm", "xử lý vi phạm", "phạt tù", "cải tạo"],
        "DÂN SỰ": ["dân sự", "hợp đồng", "quyền sở hữu", "thừa kế", "hôn nhân gia đình"],
        "HÀNH CHÍNH": ["hành chính", "xử phạt vi phạm", "thủ tục hành chính", "quyết định hành chính"],
        "LAO ĐỘNG": ["lao động", "hợp đồng lao động", "tiền lương", "bảo hiểm xã hội"],
        "THUẾ": ["thuế", "thuế thu nhập", "thuế giá trị gia tăng", "khai thuế", "nộp thuế"],
        "DOANH NGHIỆP": ["doanh nghiệp", "công ty", "thành lập doanh nghiệp", "quản lý doanh nghiệp"],
        "ĐẤT ĐAI": ["đất đai", "quyền sử dụng đất", "thủ tục đất đai", "bồi thường đất đai"],
        "XÂY DỰNG": ["xây dựng", "giấy phép xây dựng", "quy hoạch", "kiến trúc", "thiết kế"],
        "GIAO THÔNG": ["giao thông", "luật giao thông", "vi phạm giao thông", "phương tiện giao thông"],
        "Y TẾ": ["y tế", "khám chữa bệnh", "dược phẩm", "vệ sinh an toàn thực phẩm"],
        "GIÁO DỤC": ["giáo dục", "đào tạo", "chương trình giáo dục", "bằng cấp", "chứng chỉ"],
        "TÀI CHÍNH": ["tài chính", "ngân hàng", "tín dụng", "tiền tệ", "đầu tư"],
        "MÔI TRƯỜNG": ["môi trường", "bảo vệ môi trường", "ô nhiễm", "xử lý chất thải"],
        "AN NINH": ["an ninh", "quốc phòng", "bảo vệ an ninh", "trật tự an toàn xã hội"]
    }
    
    # Đếm số từ khóa xuất hiện cho mỗi domain
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword.upper() in full_text:
                score += 1
        
        if score > 0:
            domain_scores[domain] = score
    
    # Trả về domain có điểm cao nhất
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain
    
    return "KHÁC"

def create_training_splits(csv_path: str, splits_dir: str):
    """Tạo các tập train/validation/test"""
    print("🔄 Tạo các tập train/validation/test...")
    
    # Load dataset
    df = pd.read_csv(csv_path, encoding='utf-8')
    
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
    
    def train_level1(self, data_path: str) -> Dict[str, Any]:
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
    
    def train_level2(self, data_path: str) -> Dict[str, Any]:
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
    print("=" * 80)
    
    # Tạo cấu trúc project
    create_project_structure()
    
    # Bước 1: Tạo dataset
    print("\n📊 BƯỚC 1: TẠO DATASET")
    print("-" * 50)
    
    # Kiểm tra xem có file JSON không
    json_files = list(Path('.').glob('*.json'))
    if json_files:
        json_file = str(json_files[0])
        print(f"🔍 Tìm thấy file JSON: {json_file}")
    else:
        print("⚠️ Không tìm thấy file JSON. Vui lòng upload file vbpl_crawl.json vào Colab")
        return
    
    # Tạo dataset
    output_csv = "data/processed/hierarchical_legal_dataset.csv"
    df = create_hierarchical_dataset(json_file, output_csv, target_size=10000)
    
    if df is None:
        print("❌ Không thể tạo dataset")
        return
    
    # Tạo splits
    splits_dir = "data/processed/dataset_splits"
    create_training_splits(output_csv, splits_dir)
    
    # Bước 2: Training SVM
    print("\n🏋️ BƯỚC 2: TRAINING SVM")
    print("-" * 50)
    
    trainer = SVMTrainer()
    
    # Training Level 1
    results_level1 = trainer.train_level1(output_csv)
    
    # Training Level 2
    results_level2 = trainer.train_level2(output_csv)
    
    # Bước 3: Evaluation
    print("\n📊 BƯỚC 3: EVALUATION")
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