#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script hiển thị một số mẫu cụ thể từ dataset
"""

import pandas as pd

def show_detailed_samples():
    """Hiển thị một số mẫu chi tiết từ dataset"""
    
    print("🔍 Đang load dataset...")
    df = pd.read_csv('data/processed/hierarchical_legal_dataset.csv', encoding='utf-8')
    
    print("=" * 80)
    print("📝 MẪU CHI TIẾT TỪ DATASET VILEGALBERT")
    print("=" * 80)
    
    # Hiển thị 3 mẫu từ mỗi loại văn bản chính
    main_types = ['LUẬT', 'NGHỊ ĐỊNH', 'THÔNG TƯ', 'QUYẾT ĐỊNH']
    
    for doc_type in main_types:
        print(f"\n📚 LOẠI VĂN BẢN: {doc_type}")
        print("-" * 60)
        
        # Lấy 3 mẫu của loại văn bản này
        samples = df[df['type_level1'] == doc_type].head(3)
        
        for idx, sample in samples.iterrows():
            print(f"\n🔍 MẪU {idx + 1}:")
            print(f"  📋 ID: {sample['id']}")
            print(f"  🏷️ Domain: {sample['domain_level2']}")
            print(f"  🏛️ Bộ/ngành: {sample['ministry']}")
            print(f"  📖 Tên: {sample['name'][:150]}...")
            print(f"  📑 Chương: {sample['chapter'][:100]}...")
            print(f"  📝 Điều: {sample['article'][:100]}...")
            print(f"  📏 Độ dài: {sample['content_length']:,} ký tự")
            print(f"  📄 Text preview: {sample['text'][:300]}...")
    
    # Hiển thị một số mẫu từ các domain pháp lý khác nhau
    print(f"\n\n🏷️ MẪU TỪ CÁC DOMAIN PHÁP LÝ KHÁC NHAU")
    print("=" * 80)
    
    domains = ['HÌNH SỰ', 'DÂN SỰ', 'TÀI CHÍNH', 'XÂY DỰNG', 'GIÁO DỤC']
    
    for domain in domains:
        print(f"\n🔍 DOMAIN: {domain}")
        print("-" * 40)
        
        # Lấy 1 mẫu của domain này
        sample = df[df['domain_level2'] == domain].iloc[0] if len(df[df['domain_level2'] == domain]) > 0 else None
        
        if sample is not None:
            print(f"  📋 ID: {sample['id']}")
            print(f"  🏷️ Loại văn bản: {sample['type_level1']}")
            print(f"  🏛️ Bộ/ngành: {sample['ministry']}")
            print(f"  📖 Tên: {sample['name'][:150]}...")
            print(f"  📏 Độ dài: {sample['content_length']:,} ký tự")
            print(f"  📄 Text preview: {sample['text'][:250]}...")
        else:
            print(f"  ❌ Không có mẫu nào cho domain {domain}")
    
    # Hiển thị thống kê về độ dài văn bản
    print(f"\n\n📏 THỐNG KÊ CHI TIẾT VỀ ĐỘ DÀI VĂN BẢN")
    print("=" * 80)
    
    text_lengths = df['text'].str.len()
    
    print(f"📊 Thống kê tổng quát:")
    print(f"  - Trung bình: {text_lengths.mean():.0f} ký tự")
    print(f"  - Trung vị: {text_lengths.median():.0f} ký tự")
    print(f"  - Min: {text_lengths.min():,} ký tự")
    print(f"  - Max: {text_lengths.max():,} ký tự")
    print(f"  - Độ lệch chuẩn: {text_lengths.std():.0f} ký tự")
    
    # Phân bố theo độ dài
    print(f"\n📊 Phân bố theo độ dài:")
    print(f"  - Ngắn (< 500 ký tự): {len(text_lengths[text_lengths < 500]):,} ({len(text_lengths[text_lengths < 500])/len(df)*100:.1f}%)")
    print(f"  - Trung bình (500-2000 ký tự): {len(text_lengths[(text_lengths >= 500) & (text_lengths < 2000)]):,} ({len(text_lengths[(text_lengths >= 500) & (text_lengths < 2000)])/len(df)*100:.1f}%)")
    print(f"  - Dài (2000-5000 ký tự): {len(text_lengths[(text_lengths >= 2000) & (text_lengths < 5000)]):,} ({len(text_lengths[(text_lengths >= 2000) & (text_lengths < 5000)])/len(df)*100:.1f}%)")
    print(f"  - Rất dài (> 5000 ký tự): {len(text_lengths[text_lengths >= 5000]):,} ({len(text_lengths[text_lengths >= 5000])/len(df)*100:.1f}%)")
    
    print(f"\n" + "=" * 80)
    print("✅ HOÀN THÀNH HIỂN THỊ MẪU CHI TIẾT!")
    print("=" * 80)

if __name__ == "__main__":
    show_detailed_samples() 