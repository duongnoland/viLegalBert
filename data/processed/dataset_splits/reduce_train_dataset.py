import pandas as pd
import numpy as np

def reduce_train_dataset():
    # Đọc file train.csv
    print("Đang đọc file train.csv...")
    df = pd.read_csv('train.csv')
    
    print(f"Kích thước ban đầu: {len(df)} samples")
    print(f"Các columns hiện tại: {list(df.columns)}")
    
    # Loại bỏ columns content_length và chapter
    columns_to_drop = ['content_length', 'chapter']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"Đã loại bỏ columns: {columns_to_drop}")
    print(f"Các columns còn lại: {list(df.columns)}")
    
    # Giới hạn text và article xuống còn tối đa 500 ký tự
    if 'text' in df.columns:
        df['text'] = df['text'].astype(str).str[:500]
        print("Đã giới hạn text xuống còn tối đa 500 ký tự")
    
    if 'article' in df.columns:
        df['article'] = df['article'].astype(str).str[:500]
        print("Đã giới hạn article xuống còn tối đa 500 ký tự")
    
    # Rút gọn xuống còn khoảng 3k samples
    target_size = 3000
    if len(df) > target_size:
        # Lấy ngẫu nhiên 3k samples
        df = df.sample(n=target_size, random_state=42)
        print(f"Đã rút gọn xuống còn {len(df)} samples")
    
    # Lưu file mới
    output_file = 'train_reduced.csv'
    df.to_csv(output_file, index=False)
    print(f"Đã lưu file mới: {output_file}")
    
    # Hiển thị thông tin về dataset mới
    print(f"\nThông tin dataset mới:")
    print(f"- Số samples: {len(df)}")
    print(f"- Số columns: {len(df.columns)}")
    print(f"- Columns: {list(df.columns)}")
    
    # Hiển thị một vài mẫu
    print(f"\nMẫu dữ liệu:")
    print(df.head())
    
    # Thống kê về độ dài text và article
    if 'text' in df.columns:
        print(f"\nThống kê độ dài text:")
        print(f"- Trung bình: {df['text'].str.len().mean():.1f}")
        print(f"- Tối đa: {df['text'].str.len().max()}")
        print(f"- Tối thiểu: {df['text'].str.len().min()}")
    
    if 'article' in df.columns:
        print(f"\nThống kê độ dài article:")
        print(f"- Trung bình: {df['article'].str.len().mean():.1f}")
        print(f"- Tối đa: {df['article'].str.len().max()}")
        print(f"- Tối thiểu: {df['article'].str.len().min()}")

if __name__ == "__main__":
    reduce_train_dataset()
reduce_test_validation