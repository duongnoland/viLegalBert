# 🚀 **viLegalBert - Google Colab Pipeline**

## 📋 **Tổng Quan**

viLegalBert là hệ thống phân loại văn bản pháp luật Việt Nam với kiến trúc phân cấp 2 tầng:
- **Tầng 1**: Phân loại loại văn bản (LUẬT, NGHỊ ĐỊNH, THÔNG TƯ, QUYẾT ĐỊNH, ...)
- **Tầng 2**: Phân loại domain pháp lý (HÌNH SỰ, DÂN SỰ, HÀNH CHÍNH, TÀI CHÍNH, ...)

## 🎯 **Mục Tiêu**

Xây dựng pipeline hoàn chỉnh có thể chạy trực tiếp trên Google Colab để:
1. ✅ Tạo dataset từ JSON gốc
2. ✅ Training mô hình SVM
3. ✅ Evaluation kết quả
4. ✅ Lưu trữ models và kết quả

## 📁 **Files Có Sẵn**

### 🚀 **Main Pipeline**
- **`main_colab.py`**: Pipeline hoàn chỉnh cho Colab
- **`demo_colab.py`**: Demo đơn giản để test

### 📖 **Hướng Dẫn**
- **`COLAB_USAGE.md`**: Hướng dẫn sử dụng chi tiết
- **`README_COLAB.md`**: File này

## 🚀 **Cách Sử Dụng Nhanh**

### **Bước 1: Chuẩn Bị Colab**
1. Mở [Google Colab](https://colab.research.google.com)
2. Tạo notebook mới
3. Upload file `vbpl_crawl.json`

### **Bước 2: Copy & Chạy**
```python
# Copy toàn bộ nội dung main_colab.py vào cell
# Chạy cell để khởi động pipeline
```

### **Bước 3: Kết Quả**
- ✅ Dataset 10,000 samples
- ✅ Models SVM cho 2 tầng
- ✅ Kết quả evaluation
- ✅ Files được lưu tự động

## 🔧 **Tính Năng Chính**

### 📊 **Dataset Creation**
- Xử lý JSON gốc (515K+ items)
- Phân loại tự động 2 tầng
- Làm sạch và chuẩn hóa text
- Chia train/val/test (70/15/15)

### 🏋️ **SVM Training**
- TF-IDF vectorization
- Feature selection (Chi2)
- Hyperparameter tuning
- Cross-validation
- Model persistence

### 📊 **Evaluation**
- Accuracy metrics
- Classification reports
- Confusion matrices
- Performance comparison

## 📈 **Kết Quả Mong Đợi**

### **Level 1 (Loại văn bản)**
- THÔNG TƯ: ~28%
- QUYẾT ĐỊNH: ~26%
- NGHỊ QUYẾT: ~18%
- NGHỊ ĐỊNH: ~16%
- LUẬT: ~6%

### **Level 2 (Domain pháp lý)**
- HÀNH CHÍNH: ~44%
- KHÁC: ~22%
- DÂN SỰ: ~5%
- XÂY DỰNG: ~5%
- TÀI CHÍNH: ~5%

## 🎯 **Tùy Chỉnh**

### **Dataset Size**
```python
# Thay đổi số lượng samples
df = create_hierarchical_dataset(json_file, output_csv, target_size=5000)
```

### **SVM Parameters**
```python
# Trong SVMTrainer class
self.config['svm']['kernel'] = 'linear'  # rbf, poly, sigmoid
self.config['svm']['C'] = 10.0           # Regularization
self.config['svm']['gamma'] = 'auto'     # Kernel coefficient
```

### **Feature Selection**
```python
# Số features được chọn
self.config['feature_selection']['k_best'] = 3000
```

## 🚀 **Tiếp Theo**

Sau khi hoàn thành SVM, bạn có thể:

### **1. Training PhoBERT**
```python
# Sử dụng transformers library
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```

### **2. Training BiLSTM**
```python
# Sử dụng PyTorch
import torch
import torch.nn as nn
```

### **3. Training Ensemble**
```python
# Kết hợp nhiều models
from sklearn.ensemble import VotingClassifier
```

## 📊 **Cấu Trúc Output**

```
📁 Project Structure
├── 📊 data/
│   ├── 📁 processed/
│   │   ├── 📄 hierarchical_legal_dataset.csv
│   │   └── 📁 dataset_splits/
│   │       ├── 📄 train.csv
│   │       ├── 📄 validation.csv
│   │       └── 📄 test.csv
├── 🤖 models/
│   └── 📁 saved_models/
│       ├── 📁 level1_classifier/
│       │   └── 📁 svm_level1/
│       └── 📁 level2_classifier/
│           └── 📁 svm_level2/
├── 📈 results/
│   ├── 📁 training_results/
│   └── 📁 evaluation_results/
└── 📝 logs/
```

## 🔍 **Troubleshooting**

### **Lỗi Import**
```python
# Đảm bảo đã cài đặt dependencies
install_dependencies()
```

### **Lỗi Memory**
```python
# Giảm target_size
target_size = 5000  # Thay vì 10000
```

### **Lỗi File Not Found**
```python
# Kiểm tra file JSON đã upload
# Tạo cấu trúc thư mục trước
create_project_structure()
```

## 📞 **Hỗ Trợ**

Nếu gặp vấn đề:
1. ✅ Kiểm tra dependencies đã được cài đặt
2. ✅ File JSON đã được upload
3. ✅ Cấu trúc thư mục đã được tạo
4. ✅ Memory Colab đủ (RAM > 8GB)

## 🎉 **Kết Luận**

viLegalBert pipeline cho Google Colab cung cấp:
- 🚀 **Dễ sử dụng**: Copy & paste, chạy trực tiếp
- 📊 **Hoàn chỉnh**: Từ dataset đến evaluation
- 🔧 **Linh hoạt**: Dễ dàng tùy chỉnh parameters
- 💾 **Tự động**: Lưu trữ models và kết quả
- 📈 **Mở rộng**: Sẵn sàng cho các mô hình khác

---

**🚀 Chúc bạn thành công với viLegalBert trên Google Colab!** 