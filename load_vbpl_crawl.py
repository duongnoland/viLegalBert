import json
import os
from pathlib import Path

def load_vbpl_crawl():
    """Load file vbpl_crawl.json và hiển thị thông tin cơ bản"""
    
    # Đường dẫn đến file
    file_path = Path("data/vbpl_crawl.json")
    
    if not file_path.exists():
        print(f"Không tìm thấy file: {file_path}")
        return None
    
    print(f"Đang load file: {file_path}")
    print(f"Kích thước file: {file_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # Load file JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n✅ Load thành công file JSON!")
        print(f"Kiểu dữ liệu: {type(data)}")
        
        if isinstance(data, list):
            print(f"Số lượng items: {len(data)}")
            if len(data) > 0:
                print(f"\nVí dụ item đầu tiên:")
                print(json.dumps(data[0], indent=2, ensure_ascii=False)[:500] + "...")
                
                # Hiển thị cấu trúc của item đầu tiên
                if isinstance(data[0], dict):
                    print(f"\nCác key trong item đầu tiên:")
                    for key, value in data[0].items():
                        value_type = type(value).__name__
                        if isinstance(value, (list, dict)):
                            value_info = f"{value_type} (len: {len(value)})"
                        else:
                            value_info = f"{value_type}: {str(value)[:100]}"
                        print(f"  - {key}: {value_info}")
        
        elif isinstance(data, dict):
            print(f"Số lượng key: {len(data)}")
            print(f"Các key chính:")
            for key, value in data.items():
                value_type = type(value).__name__
                if isinstance(value, (list, dict)):
                    value_info = f"{value_type} (len: {len(value)})"
                else:
                    value_info = f"{value_type}: {str(value)[:100]}"
                print(f"  - {key}: {value_info}")
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi khi parse JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        return None

def explore_data_structure(data, max_items=5):
    """Khám phá cấu trúc dữ liệu chi tiết hơn"""
    if not data:
        return
    
    print(f"\n{'='*50}")
    print("KHÁM PHÁ CẤU TRÚC DỮ LIỆU CHI TIẾT")
    print(f"{'='*50}")
    
    if isinstance(data, list):
        print(f"Tổng số items: {len(data)}")
        
        # Hiển thị một số items đầu tiên
        for i in range(min(max_items, len(data))):
            print(f"\n--- Item {i+1} ---")
            item = data[i]
            if isinstance(item, dict):
                for key, value in item.items():
                    print(f"  {key}: {type(value).__name__} = {str(value)[:200]}")
            else:
                print(f"  {type(item).__name__}: {str(item)[:200]}")
    
    elif isinstance(data, dict):
        print(f"Tổng số key: {len(data)}")
        
        # Hiển thị một số key đầu tiên
        for i, (key, value) in enumerate(data.items()):
            if i >= max_items:
                print(f"  ... và {len(data) - max_items} key khác")
                break
            print(f"\n--- Key {i+1}: {key} ---")
            print(f"  Type: {type(value).__name__}")
            if isinstance(value, (list, dict)):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  Sample: {str(value)[:200]}...")
            else:
                print(f"  Value: {str(value)[:200]}")

if __name__ == "__main__":
    print("🚀 Bắt đầu load file vbpl_crawl.json...")
    
    # Load dữ liệu
    data = load_vbpl_crawl()
    
    if data:
        # Khám phá cấu trúc chi tiết
        explore_data_structure(data)
        
        print(f"\n✅ Hoàn thành! File đã được load thành công.")
    else:
        print("❌ Không thể load file.") 