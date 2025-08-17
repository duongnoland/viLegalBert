# 📚 Index - Dataset Analysis Documentation

## 🚀 Quick Start

### 📊 Xem Thông Tin Nhanh
- **[Dataset Summary](dataset_summary.md)** - Tóm tắt thống kê dataset
- **[README](README.md)** - Hướng dẫn sử dụng các script

### 🔧 Chạy Script Phân Tích
```bash
# Thông tin tổng quan
python doc/dataset_analysis/show_dataset.py

# Thông tin các tập dữ liệu
python doc/dataset_analysis/show_splits.py

# Mẫu chi tiết
python doc/dataset_analysis/show_samples.py
```

## 📁 Cấu Trúc Thư Mục

```
doc/dataset_analysis/
├── index.md                 # File này - Index chính
├── README.md                # Hướng dẫn sử dụng
├── dataset_summary.md       # Tóm tắt thống kê
├── show_dataset.py          # Script hiển thị thông tin tổng quan
├── show_splits.py           # Script hiển thị thông tin các tập dữ liệu
└── show_samples.py          # Script hiển thị mẫu chi tiết
```

## 🎯 Nội Dung Chính

### 1. **Thông Tin Tổng Quan** 📈
- **Dataset**: 10,000 samples, 9 cột, 38.21 MB
- **Phân loại Level 1**: 7 loại văn bản (THÔNG TƯ, QUYẾT ĐỊNH, NGHỊ QUYẾT, NGHỊ ĐỊNH, LUẬT, KHÁC, PHÁP LỆNH)
- **Phân loại Level 2**: 15 domain pháp lý (HÀNH CHÍNH, DÂN SỰ, XÂY DỰNG, TÀI CHÍNH, GIÁO DỤC, HÌNH SỰ, DOANH NGHIỆP, LAO ĐỘNG, THUẾ, MÔI TRƯỜNG, AN NINH, Y TẾ, ĐẤT ĐAI, GIAO THÔNG, KHÁC)

### 2. **Các Tập Dữ Liệu** 📚
- **Train**: 7,000 samples (70%)
- **Validation**: 1,500 samples (15%)
- **Test**: 1,500 samples (15%)

### 3. **Thống Kê Độ Dài** 📏
- **Trung bình**: 1,487 ký tự
- **Phân bố**: 32.4% ngắn, 50.0% trung bình, 12.9% dài, 4.7% rất dài

## 🔍 Cách Sử Dụng

### **Bước 1: Xem Tóm Tắt**
```bash
# Mở file tóm tắt
cat doc/dataset_analysis/dataset_summary.md
```

### **Bước 2: Chạy Script Phân Tích**
```bash
# Thông tin tổng quan
python doc/dataset_analysis/show_dataset.py

# Thông tin các tập dữ liệu
python doc/dataset_analysis/show_splits.py

# Mẫu chi tiết
python doc/dataset_analysis/show_samples.py
```

### **Bước 3: Xem Hướng Dẫn Chi Tiết**
```bash
# Mở file README
cat doc/dataset_analysis/README.md
```

## 📊 Kết Quả Mong Đợi

### **Từ `show_dataset.py`:**
- Thông tin tổng quan về 10,000 samples
- Phân loại 7 loại văn bản (Level 1)
- Phân loại 15 domain pháp lý (Level 2)
- Thống kê độ dài văn bản từ 11 đến 85,849 ký tự

### **Từ `show_splits.py`:**
- Train set: 7,000 samples (70%)
- Validation set: 1,500 samples (15%)
- Test set: 1,500 samples (15%)
- Phân bố cân bằng giữa các tập

### **Từ `show_samples.py`:**
- Mẫu cụ thể từ mỗi loại văn bản
- Mẫu từ các domain pháp lý khác nhau
- Thống kê chi tiết về độ dài văn bản

## 💡 Lưu Ý Quan Trọng

1. **Đảm bảo dataset đã được tạo** trước khi chạy các script
2. **Cài đặt dependencies**: `pip install pandas numpy`
3. **Encoding**: Dataset sử dụng UTF-8
4. **Memory**: Dataset có kích thước ~38MB

## 🚀 Tiếp Theo

Sau khi phân tích dataset, bạn có thể:

1. **Training mô hình SVM**: `python src/main.py --mode train_svm --level both`
2. **Training mô hình PhoBERT**: `python src/main.py --mode train_phobert --level both`
3. **Training mô hình BiLSTM**: `python src/main.py --mode train_bilstm --level both`
4. **Training mô hình Ensemble**: `python src/main.py --mode train_ensemble --level both`
5. **Evaluation tổng hợp**: `python src/main.py --mode evaluate_all`

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra dataset đã được tạo thành công chưa
2. Dependencies đã được cài đặt chưa
3. Python version có tương thích không (3.7+)
4. Đường dẫn file có chính xác không

---

**📅 Ngày tạo**: 2025  
**🔄 Phiên bản**: 1.0  
**✅ Trạng thái**: Hoàn thành và sẵn sàng sử dụng 