import pandas as pd
import os

def reduce_dataset(input_file, output_file):
    """
    Rút gọn dataset theo yêu cầu:
    - Loại bỏ columns content_length và chapter
    - Giới hạn text và article xuống còn tối đa 500 ký tự
    - Giữ nguyên số samples
    """
    print(f"\n{'='*50}")
    print(f"Xử lý file: {input_file}")
    print(f"{'='*50}")
    
    # Đọc file CSV
    print("Đang đọc file...")
    df = pd.read_csv(input_file)
    
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
    
    # Lưu file mới
    df.to_csv(output_file, index=False)
    print(f"Đã lưu file mới: {output_file}")
    
    # Hiển thị thông tin về dataset mới
    print(f"\nThông tin dataset mới:")
    print(f"- Số samples: {len(df)}")
    print(f"- Số columns: {len(df.columns)}")
    print(f"- Columns: {list(df.columns)}")
    
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
    
    return df

def main():
    """
    Xử lý test.csv và validation.csv:
    - test.csv -> test_reduced.csv
    - validation.csv -> validation_reduced.csv
    """
    print("�� Bắt đầu xử lý test.csv và validation.csv...")
    
    # Định nghĩa các files cần xử lý
    files_to_process = [
        {
            'input': 'test.csv',
            'output': 'test_reduced.csv'
        },
        {
            'input': 'validation.csv',
            'output': 'validation_reduced.csv'
        }
    ]
    
    # Xử lý từng file
    for file_info in files_to_process:
        input_file = file_info['input']
        output_file = file_info['output']
        
        if os.path.exists(input_file):
            try:
                reduce_dataset(input_file, output_file)
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {input_file}: {str(e)}")
        else:
            print(f"⚠️  File {input_file} không tồn tại, bỏ qua...")
    
    print(f"\n{'='*50}")
    print("✅ Hoàn thành xử lý test.csv và validation.csv!")
    print(f"{'='*50}")
    
    # Hiển thị tổng kết
    print("\n📊 Tổng kết:")
    for file_info in files_to_process:
        output_file = file_info['output']
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"- {output_file}: {file_size:.1f} MB")
        else:
            print(f"- {output_file}: Không tạo được")

if __name__ == "__main__":
    main()
