#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Demo viLegalBert cho Google Colab
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cài đặt dependencies
def install_deps():
    try:
        import sklearn
        print("✅ scikit-learn đã sẵn sàng")
    except:
        os.system("pip install scikit-learn")
        print("📦 Đã cài đặt scikit-learn")
    
    try:
        import torch
        print("✅ PyTorch đã sẵn sàng")
    except:
        os.system("pip install torch")
        print("📦 Đã cài đặt PyTorch")

# Tạo cấu trúc thư mục
def create_dirs():
    dirs = ['data/processed', 'models', 'results', 'logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"✅ Tạo thư mục: {d}")

# Tạo dataset mẫu
def create_sample_dataset():
    print("📊 Tạo dataset mẫu...")
    
    # Dữ liệu mẫu
    sample_data = [
        {
            'id': '001',
            'text': 'Luật về quyền sở hữu trí tuệ và bảo vệ quyền lợi người tiêu dùng',
            'type_level1': 'LUẬT',
            'domain_level2': 'DÂN SỰ',
            'ministry': 'QUỐC HỘI'
        },
        {
            'id': '002', 
            'text': 'Nghị định về xử phạt vi phạm hành chính trong lĩnh vực giao thông',
            'type_level1': 'NGHỊ ĐỊNH',
            'domain_level2': 'GIAO THÔNG',
            'ministry': 'CHÍNH PHỦ'
        },
        {
            'id': '003',
            'text': 'Thông tư hướng dẫn về thuế thu nhập cá nhân và thuế giá trị gia tăng',
            'type_level1': 'THÔNG TƯ',
            'domain_level2': 'THUẾ',
            'ministry': 'BỘ TÀI CHÍNH'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/processed/sample_dataset.csv', index=False, encoding='utf-8')
    print(f"✅ Đã tạo dataset mẫu với {len(df)} samples")
    return df

# Training SVM đơn giản
def train_simple_svm():
    print("🏋️ Training SVM đơn giản...")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load data
    df = pd.read_csv('data/processed/sample_dataset.csv', encoding='utf-8')
    X = df['text']
    y = df['type_level1']
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Chia data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
    
    # Training
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    
    # Evaluation
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Lưu model
    import pickle
    model_data = {'model': svm, 'vectorizer': vectorizer}
    
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("💾 Model đã được lưu: models/svm_model.pkl")
    return svm, vectorizer

# Main function
def main():
    print("🚀 DEMO VILEGALBERT CHO GOOGLE COLAB!")
    print("=" * 50)
    
    # Cài đặt dependencies
    install_deps()
    
    # Tạo cấu trúc
    create_dirs()
    
    # Tạo dataset mẫu
    df = create_sample_dataset()
    
    # Training SVM
    model, vectorizer = train_simple_svm()
    
    print("\n🎉 DEMO HOÀN THÀNH!")
    print("📊 Dataset:", len(df), "samples")
    print("🏋️ Model SVM đã được train")
    print("💾 Có thể sử dụng cho dự đoán")

if __name__ == "__main__":
    main() 