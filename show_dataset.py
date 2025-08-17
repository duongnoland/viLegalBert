#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script hiển thị thông tin dataset viLegalBert
"""

import pandas as pd
import numpy as np

def show_dataset_info():
    """Hiển thị thông tin tổng quan về dataset"""
    
    # Load dataset
    print("🔍 Đang load dataset...")
    df = pd.read_csv('data/processed/hierarchical_legal_dataset.csv', encoding='utf-8')
    
    print("=" * 80)
    print("📊 THÔNG TIN TỔNG QUAN DATASET VILEGALBERT")
    print("=" * 80)
    
    # Thông tin cơ bản
    print(f"📈 Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"💾 Kích thước file: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Thống kê cơ bản
    print(f"\n📊 THỐNG KÊ CƠ BẢN:")
    print(f"  - Tổng số samples: {len(df):,}")
    print(f"  - Số cột: {len(df.columns)}")
    print(f"  - Dữ liệu bị thiếu: {df.isnull().sum().sum()}")
    
    # Phân loại Level 1
    print(f"\n🏷️ PHÂN LOẠI TẦNG 1 (Loại văn bản):")
    level1_counts = df['type_level1'].value_counts()
    for doc_type, count in level1_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  - {doc_type}: {count:,} samples ({percentage:.1f}%)")
    
    # Phân loại Level 2
    print(f"\n🏷️ PHÂN LOẠI TẦNG 2 (Domain pháp lý):")
    level2_counts = df['domain_level2'].value_counts()
    for domain, count in level2_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  - {domain}: {count:,} samples ({percentage:.1f}%)")
    
    # Thống kê về độ dài văn bản
    print(f"\n📏 THỐNG KÊ ĐỘ DÀI VĂN BẢN:")
    text_lengths = df['text'].str.len()
    print(f"  - Độ dài trung bình: {text_lengths.mean():.0f} ký tự")
    print(f"  - Độ dài ngắn nhất: {text_lengths.min():,} ký tự")
    print(f"  - Độ dài dài nhất: {text_lengths.max():,} ký tự")
    print(f"  - Độ lệch chuẩn: {text_lengths.std():.0f} ký tự")
    
    # Hiển thị một số mẫu
    print(f"\n📝 MẪU DỮ LIỆU (5 samples đầu tiên):")
    print("-" * 80)
    
    for i in range(min(5, len(df))):
        sample = df.iloc[i]
        print(f"\n🔍 SAMPLE {i+1}:")
        print(f"  ID: {sample['id']}")
        print(f"  Loại văn bản (Level 1): {sample['type_level1']}")
        print(f"  Domain pháp lý (Level 2): {sample['domain_level2']}")
        print(f"  Bộ/ngành: {sample['ministry']}")
        print(f"  Tên văn bản: {sample['name'][:100]}...")
        print(f"  Chương: {sample['chapter'][:50]}...")
        print(f"  Điều: {sample['article'][:50]}...")
        print(f"  Độ dài nội dung: {sample['content_length']:,} ký tự")
        print(f"  Text preview: {sample['text'][:200]}...")
    
    # Thống kê về các bộ/ngành
    print(f"\n🏛️ THỐNG KÊ THEO BỘ/NGÀNH:")
    ministry_counts = df['ministry'].value_counts().head(10)
    for ministry, count in ministry_counts.items():
        if pd.notna(ministry) and ministry.strip():
            percentage = (count / len(df)) * 100
            print(f"  - {ministry}: {count:,} samples ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH HIỂN THỊ THÔNG TIN DATASET!")
    print("=" * 80)

if __name__ == "__main__":
    show_dataset_info() 