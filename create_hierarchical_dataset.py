import json
import pandas as pd
import random
from pathlib import Path
from collections import defaultdict, Counter
import re

def clean_text(text):
    """Làm sạch văn bản"""
    if not text:
        return ""
    
    # Loại bỏ ký tự đặc biệt và chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}]', '', text)
    return text

def extract_document_type(type_text):
    """Trích xuất loại văn bản cơ bản từ trường type"""
    if not type_text:
        return "KHÁC"
    
    type_text = type_text.upper().strip()
    
    # Mapping các loại văn bản cơ bản
    type_mapping = {
        "LUẬT": "LUẬT",
        "NGHỊ ĐỊNH": "NGHỊ ĐỊNH", 
        "THÔNG TƯ": "THÔNG TƯ",
        "NGHỊ QUYẾT": "NGHỊ QUYẾT",
        "QUYẾT ĐỊNH": "QUYẾT ĐỊNH",
        "CHỈ THỊ": "CHỈ THỊ",
        "PHÁP LỆNH": "PHÁP LỆNH",
        "NGHỊ QUYẾT LIÊN TỊCH": "NGHỊ QUYẾT LIÊN TỊCH",
        "THÔNG TƯ LIÊN TỊCH": "THÔNG TƯ LIÊN TỊCH",
        "NGHỊ ĐỊNH LIÊN TỊCH": "NGHỊ ĐỊNH LIÊN TỊCH"
    }
    
    for key, value in type_mapping.items():
        if key in type_text:
            return value
    
    return "KHÁC"

def extract_legal_domain(content, name, chapter_name):
    """Trích xuất domain pháp lý từ nội dung"""
    if not content:
        return "KHÁC"
    
    # Kết hợp nội dung để phân tích
    full_text = f"{name} {chapter_name} {content}".upper()
    
    # Mapping các domain pháp lý với từ khóa tiếng Việt
    domain_keywords = {
        "HÌNH SỰ": [
            "hình sự", "tội phạm", "xử lý vi phạm", "phạt tù", "cải tạo", 
            "truy cứu trách nhiệm", "hình phạt", "tội danh", "vụ án", "bị can",
            "bị cáo", "thẩm phán", "kiểm sát viên", "luật sư", "tòa án"
        ],
        "DÂN SỰ": [
            "dân sự", "hợp đồng", "quyền sở hữu", "thừa kế", "hôn nhân gia đình", 
            "bồi thường", "tranh chấp", "quyền lợi", "nghĩa vụ", "tài sản",
            "quyền tài sản", "quyền nhân thân", "bảo vệ quyền lợi"
        ],
        "HÀNH CHÍNH": [
            "hành chính", "xử phạt vi phạm", "thủ tục hành chính", "quyết định hành chính",
            "khiếu nại", "tố cáo", "cơ quan hành chính", "chính quyền", "ủy ban",
            "sở", "phòng", "ban", "cơ quan nhà nước"
        ],
        "LAO ĐỘNG": [
            "lao động", "hợp đồng lao động", "tiền lương", "bảo hiểm xã hội", 
            "an toàn lao động", "thời gian làm việc", "nghỉ phép", "đình công",
            "người lao động", "người sử dụng lao động", "quan hệ lao động"
        ],
        "THUẾ": [
            "thuế", "thuế thu nhập", "thuế giá trị gia tăng", "thuế xuất nhập khẩu", 
            "khai thuế", "nộp thuế", "hoàn thuế", "miễn thuế", "giảm thuế",
            "cơ quan thuế", "tổng cục thuế", "chi cục thuế"
        ],
        "DOANH NGHIỆP": [
            "doanh nghiệp", "công ty", "thành lập doanh nghiệp", "quản lý doanh nghiệp",
            "đăng ký kinh doanh", "giấy phép kinh doanh", "vốn điều lệ", "cổ đông",
            "hội đồng quản trị", "giám đốc", "phó giám đốc"
        ],
        "ĐẤT ĐAI": [
            "đất đai", "quyền sử dụng đất", "thủ tục đất đai", "bồi thường đất đai",
            "giấy chứng nhận quyền sử dụng đất", "quy hoạch đất đai", "thu hồi đất",
            "giao đất", "cho thuê đất", "chuyển đổi mục đích sử dụng đất"
        ],
        "XÂY DỰNG": [
            "xây dựng", "giấy phép xây dựng", "quy hoạch", "kiến trúc", "thiết kế",
            "thi công", "giám sát", "nghiệm thu", "bảo hành", "bảo trì",
            "công trình xây dựng", "dự án xây dựng"
        ],
        "GIAO THÔNG": [
            "giao thông", "luật giao thông", "vi phạm giao thông", "phương tiện giao thông",
            "đường bộ", "đường sắt", "đường thủy", "đường hàng không", "biển báo",
            "đèn tín hiệu", "vạch kẻ đường", "cầu đường"
        ],
        "Y TẾ": [
            "y tế", "khám chữa bệnh", "dược phẩm", "vệ sinh an toàn thực phẩm",
            "bệnh viện", "phòng khám", "bác sĩ", "y tá", "dược sĩ", "thuốc",
            "thiết bị y tế", "dịch vụ y tế", "bảo hiểm y tế"
        ],
        "GIÁO DỤC": [
            "giáo dục", "đào tạo", "chương trình giáo dục", "bằng cấp", "chứng chỉ",
            "trường học", "giáo viên", "học sinh", "sinh viên", "giảng viên",
            "chương trình đào tạo", "cơ sở giáo dục"
        ],
        "TÀI CHÍNH": [
            "tài chính", "ngân hàng", "tín dụng", "tiền tệ", "đầu tư", "cho vay",
            "tiết kiệm", "bảo hiểm", "chứng khoán", "quỹ đầu tư", "công ty tài chính",
            "ngân hàng nhà nước", "ngân hàng thương mại"
        ],
        "MÔI TRƯỜNG": [
            "môi trường", "bảo vệ môi trường", "ô nhiễm", "xử lý chất thải",
            "khí thải", "nước thải", "rác thải", "tiếng ồn", "bụi", "hóa chất",
            "đánh giá tác động môi trường", "giấy phép môi trường"
        ],
        "AN NINH": [
            "an ninh", "quốc phòng", "bảo vệ an ninh", "trật tự an toàn xã hội",
            "công an", "bộ đội", "quân đội", "cảnh sát", "an ninh quốc gia",
            "an ninh trật tự", "phòng chống tội phạm"
        ]
    }
    
    # Đếm số từ khóa xuất hiện cho mỗi domain
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = 0
        for keyword in keywords:
            # Tìm kiếm từ khóa trong văn bản (không phân biệt dấu)
            if keyword.upper() in full_text:
                score += 1
            # Tìm kiếm với các biến thể dấu
            elif keyword.replace(' ', '').upper() in full_text.replace(' ', ''):
                score += 1
        
        if score > 0:
            domain_scores[domain] = score
    
    # Trả về domain có điểm cao nhất
    if domain_scores:
        best_domain = max(domain_scores, key=domain_scores.get)
        print(f"🔍 Domain được chọn: {best_domain} (điểm: {domain_scores[best_domain]})")
        return best_domain
    
    return "KHÁC"

def create_hierarchical_dataset(json_file_path, output_csv_path, target_size=10000):
    """Tạo dataset phân cấp 2 tầng từ file JSON"""
    
    print(f"🚀 Bắt đầu tạo dataset từ file: {json_file_path}")
    
    # Kiểm tra file tồn tại
    if not Path(json_file_path).exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file: {json_file_path}")
    
    # Load dữ liệu JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Load thành công {len(data)} items từ JSON")
    
    # Chuẩn bị dữ liệu cho dataset
    dataset_items = []
    
    # Lấy mẫu ngẫu nhiên để đảm bảo đa dạng
    if len(data) > target_size:
        sampled_data = random.sample(data, target_size)
    else:
        sampled_data = data
    
    print(f"📊 Xử lý {len(sampled_data)} items...")
    
    for item in sampled_data:
        try:
            # Trích xuất thông tin cơ bản
            doc_id = item.get('id', '')
            doc_type = extract_document_type(item.get('type', ''))
            doc_name = clean_text(item.get('name', ''))
            ministry = clean_text(item.get('ministry', ''))
            chapter_name = clean_text(item.get('chapter_name', ''))
            article = clean_text(item.get('article', ''))
            content = clean_text(item.get('content', ''))
            
            # Tạo văn bản đầy đủ để phân loại
            full_text = f"{doc_name} {chapter_name} {article} {content}"
            
            # Trích xuất domain pháp lý
            legal_domain = extract_legal_domain(content, doc_name, chapter_name)
            
            # Tạo item cho dataset
            dataset_item = {
                'id': doc_id,
                'text': full_text,
                'type_level1': doc_type,  # Tầng 1: Loại văn bản cơ bản
                'domain_level2': legal_domain,  # Tầng 2: Domain pháp lý
                'ministry': ministry,
                'name': doc_name,
                'chapter': chapter_name,
                'article': article,
                'content_length': len(content)
            }
            
            dataset_items.append(dataset_item)
            
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý item: {e}")
            continue
    
    # Tạo DataFrame
    df = pd.DataFrame(dataset_items)
    
    # Thống kê dataset
    print(f"\n📈 THỐNG KÊ DATASET:")
    print(f"Tổng số samples: {len(df)}")
    
    print(f"\n🏷️ PHÂN LOẠI TẦNG 1 (Loại văn bản):")
    type_counts = df['type_level1'].value_counts()
    for doc_type, count in type_counts.items():
        print(f"  - {doc_type}: {count}")
    
    print(f"\n🏷️ PHÂN LOẠI TẦNG 2 (Domain pháp lý):")
    domain_counts = df['domain_level2'].value_counts()
    for domain, count in domain_counts.items():
        print(f"  - {domain}: {count}")
    
    # Tạo thư mục output nếu chưa có
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu dataset
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n✅ Đã lưu dataset vào: {output_csv_path}")
    
    return df

def create_training_splits(dataset_path, output_dir):
    """Tạo các tập train/validation/test từ dataset"""
    
    print(f"\n🔄 Tạo các tập train/validation/test...")
    
    # Load dataset
    df = pd.read_csv(dataset_path, encoding='utf-8')
    
    # Tạo thư mục output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Chia dữ liệu theo tỷ lệ 70/15/15
    total_size = len(df)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    # Shuffle dữ liệu
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Chia dữ liệu
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:train_size + val_size]
    test_df = df_shuffled[train_size + val_size:]
    
    # Lưu các tập
    train_path = output_path / "train.csv"
    val_path = output_path / "validation.csv"
    test_path = output_path / "test.csv"
    
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    print(f"✅ Train set: {len(train_df)} samples -> {train_path}")
    print(f"✅ Validation set: {len(val_df)} samples -> {val_path}")
    print(f"✅ Test set: {len(test_df)} samples -> {test_path}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Đường dẫn file - ĐÃ SỬA
    json_file = "data/raw/vbpl_crawl.json"  # ✅ Đường dẫn đúng
    output_csv = "data/processed/hierarchical_legal_dataset.csv"  # ✅ Lưu vào processed
    splits_dir = "data/processed/dataset_splits"  # ✅ Lưu vào processed
    
    print("🔍 Kiểm tra cấu trúc thư mục...")
    
    # Kiểm tra file JSON
    if not Path(json_file).exists():
        print(f"❌ Không tìm thấy file: {json_file}")
        print("💡 Hãy đảm bảo file vbpl_crawl.json đã được di chuyển vào data/raw/")
        exit(1)
    
    print(f"✅ Tìm thấy file JSON: {json_file}")
    
    try:
        # Tạo dataset chính
        df = create_hierarchical_dataset(json_file, output_csv, target_size=10000)
        
        # Tạo các tập train/validation/test
        create_training_splits(output_csv, splits_dir)
        
        print(f"\n🎉 HOÀN THÀNH! Dataset đã được tạo thành công:")
        print(f"  - Dataset chính: {output_csv}")
        print(f"  - Các tập chia: {splits_dir}/")
        print(f"  - Tổng số samples: {len(df)}")
        
        # Thông tin về cấu trúc thư mục
        print(f"\n📁 Cấu trúc thư mục đã tạo:")
        print(f"  - data/processed/hierarchical_legal_dataset.csv")
        print(f"  - data/processed/dataset_splits/train.csv")
        print(f"  - data/processed/dataset_splits/validation.csv")
        print(f"  - data/processed/dataset_splits/test.csv")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình tạo dataset: {e}")
        exit(1) 