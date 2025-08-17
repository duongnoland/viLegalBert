# 📊 Dataset Analysis Scripts - viLegalBert

Thư mục này chứa các script Python để phân tích và hiển thị thông tin chi tiết về dataset viLegalBert.

## 🚀 Các Script Có Sẵn

### 1. `show_dataset.py` - Thông Tin Tổng Quan
**Mục đích**: Hiển thị thông tin tổng quan về dataset chính
**Chức năng**:
- Thống kê cơ bản (shape, columns, kích thước)
- Phân loại Level 1 (loại văn bản)
- Phân loại Level 2 (domain pháp lý)
- Thống kê độ dài văn bản
- Mẫu dữ liệu (5 samples đầu tiên)
- Thống kê theo bộ/ngành

**Cách chạy**:
```bash
python doc/dataset_analysis/show_dataset.py
```

### 2. `show_splits.py` - Thông Tin Các Tập Dữ Liệu
**Mục đích**: Hiển thị thông tin chi tiết về các tập train/validation/test
**Chức năng**:
- Thống kê số lượng samples trong từng tập
- Phân loại Level 1 trong từng tập
- Phân loại Level 2 trong từng tập
- Kiểm tra tính cân bằng giữa các tập

**Cách chạy**:
```bash
python doc/dataset_analysis/show_splits.py
```

### 3. `show_samples.py` - Mẫu Chi Tiết
**Mục đích**: Hiển thị các mẫu cụ thể từ dataset
**Chức năng**:
- Mẫu từ các loại văn bản chính (LUẬT, NGHỊ ĐỊNH, THÔNG TƯ, QUYẾT ĐỊNH)
- Mẫu từ các domain pháp lý khác nhau
- Thống kê chi tiết về độ dài văn bản
- Phân bố theo độ dài

**Cách chạy**:
```bash
python doc/dataset_analysis/show_samples.py
```

## 📋 Yêu Cầu Hệ Thống

- Python 3.7+
- pandas
- numpy

## 🔧 Cài Đặt Dependencies

```bash
pip install pandas numpy
```

## 📁 Cấu Trúc Thư Mục

```
doc/dataset_analysis/
├── README.md              # File hướng dẫn này
├── show_dataset.py        # Script hiển thị thông tin tổng quan
├── show_splits.py         # Script hiển thị thông tin các tập dữ liệu
└── show_samples.py        # Script hiển thị mẫu chi tiết
```

## 🎯 Kết Quả Mong Đợi

### Từ `show_dataset.py`:
- Thông tin tổng quan về 10,000 samples
- Phân loại 7 loại văn bản (Level 1)
- Phân loại 15 domain pháp lý (Level 2)
- Thống kê độ dài văn bản từ 11 đến 85,849 ký tự

### Từ `show_splits.py`:
- Train set: 7,000 samples (70%)
- Validation set: 1,500 samples (15%)
- Test set: 1,500 samples (15%)
- Phân bố cân bằng giữa các tập

### Từ `show_samples.py`:
- Mẫu cụ thể từ mỗi loại văn bản
- Mẫu từ các domain pháp lý khác nhau
- Thống kê chi tiết về độ dài văn bản

## 💡 Lưu Ý Sử Dụng

1. **Đảm bảo dataset đã được tạo**: Các script này yêu cầu dataset đã được tạo và lưu trong `data/processed/`
2. **Encoding**: Dataset sử dụng UTF-8 encoding
3. **Memory**: Dataset có kích thước ~38MB, đảm bảo đủ RAM
4. **Performance**: Các script được tối ưu để chạy nhanh trên dataset 10K samples

## 🔍 Troubleshooting

### Lỗi "File not found":
- Kiểm tra đường dẫn `data/processed/hierarchical_legal_dataset.csv`
- Đảm bảo đã chạy `create_hierarchical_dataset.py` trước

### Lỗi "Memory error":
- Dataset khá lớn, đảm bảo có đủ RAM
- Có thể giảm số lượng samples trong dataset

### Lỗi "Encoding error":
- Đảm bảo terminal hỗ trợ UTF-8
- Kiểm tra Python version (3.7+)

## 📞 Hỗ Trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Dataset đã được tạo thành công chưa
2. Dependencies đã được cài đặt chưa
3. Python version có tương thích không
4. Đường dẫn file có chính xác không

---

**Tác giả**: viLegalBert Team  
**Ngày tạo**: 2025  
**Phiên bản**: 1.0 