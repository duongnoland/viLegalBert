#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script hiển thị thông tin về các tập train/validation/test
"""

import pandas as pd

def show_splits_info():
    """Hiển thị thông tin về các tập dữ liệu"""
    
    print("🔍 Đang load các tập dữ liệu...")
    
    # Load các tập dữ liệu
    train_df = pd.read_csv('data/processed/dataset_splits/train.csv', encoding='utf-8')
    val_df = pd.read_csv('data/processed/dataset_splits/validation.csv', encoding='utf-8')
    test_df = pd.read_csv('data/processed/dataset_splits/test.csv', encoding='utf-8')
    
    print("=" * 80)
    print("📊 THÔNG TIN CÁC TẬP DỮ LIỆU")
    print("=" * 80)
    
    # Thống kê cơ bản
    print(f"📈 THỐNG KÊ CƠ BẢN:")
    print(f"  - Train set: {len(train_df):,} samples (70%)")
    print(f"  - Validation set: {len(val_df):,} samples (15%)")
    print(f"  - Test set: {len(test_df):,} samples (15%)")
    print(f"  - Tổng cộng: {len(train_df) + len(val_df) + len(test_df):,} samples")
    
    # Phân loại Level 1 trong từng tập
    print(f"\n🏷️ PHÂN LOẠI TẦNG 1 (Loại văn bản):")
    
    print(f"\n  📚 TRAIN SET:")
    train_level1 = train_df['type_level1'].value_counts()
    for doc_type, count in train_level1.items():
        percentage = (count / len(train_df)) * 100
        print(f"    - {doc_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n  📚 VALIDATION SET:")
    val_level1 = val_df['type_level1'].value_counts()
    for doc_type, count in val_level1.items():
        percentage = (count / len(val_df)) * 100
        print(f"    - {doc_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n  📚 TEST SET:")
    test_level1 = test_df['type_level1'].value_counts()
    for doc_type, count in test_level1.items():
        percentage = (count / len(test_df)) * 100
        print(f"    - {doc_type}: {count:,} ({percentage:.1f}%)")
    
    # Phân loại Level 2 trong từng tập
    print(f"\n🏷️ PHÂN LOẠI TẦNG 2 (Domain pháp lý):")
    
    print(f"\n  📚 TRAIN SET:")
    train_level2 = train_df['domain_level2'].value_counts().head(5)
    for domain, count in train_level2.items():
        percentage = (count / len(train_df)) * 100
        print(f"    - {domain}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n  📚 VALIDATION SET:")
    val_level2 = val_df['domain_level2'].value_counts().head(5)
    for domain, count in val_level2.items():
        percentage = (count / len(val_df)) * 100
        print(f"    - {domain}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n  📚 TEST SET:")
    test_level2 = test_df['domain_level2'].value_counts().head(5)
    for domain, count in test_level2.items():
        percentage = (count / len(test_df)) * 100
        print(f"    - {domain}: {count:,} ({percentage:.1f}%)")
    
    # Kiểm tra tính cân bằng
    print(f"\n⚖️ KIỂM TRA TÍNH CÂN BẰNG:")
    
    # So sánh phân bố giữa các tập
    print(f"\n  📊 So sánh phân bố Level 1 giữa Train và Test:")
    for doc_type in train_level1.index:
        train_count = train_level1.get(doc_type, 0)
        test_count = test_level1.get(doc_type, 0)
        train_pct = (train_count / len(train_df)) * 100
        test_pct = (test_count / len(test_df)) * 100
        diff = abs(train_pct - test_pct)
        print(f"    - {doc_type}: Train {train_pct:.1f}% vs Test {test_pct:.1f}% (chênh lệch: {diff:.1f}%)")
    
    print(f"\n" + "=" * 80)
    print("✅ HOÀN THÀNH HIỂN THỊ THÔNG TIN CÁC TẬP DỮ LIỆU!")
    print("=" * 80)

if __name__ == "__main__":
    show_splits_info() 