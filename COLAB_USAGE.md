# 🚀 **Hướng Dẫn Sử Dụng viLegalBert trên Google Colab**

## 📋 **Bước 1: Chuẩn Bị**

1. **Mở Google Colab**: Truy cập [colab.research.google.com](https://colab.research.google.com)
2. **Tạo notebook mới**: File → New notebook
3. **Upload file JSON**: Upload file `vbpl_crawl.json` vào Colab

## 📋 **Bước 2: Copy Code**

Copy toàn bộ nội dung file `main_colab.py` vào một cell của Colab và chạy.

## 📋 **Bước 3: Chạy Pipeline**

Sau khi copy code, chạy cell để:
- ✅ Cài đặt dependencies
- ✅ Tạo cấu trúc project
- ✅ Tạo dataset phân cấp 2 tầng
- ✅ Training mô hình SVM
- ✅ Evaluation kết quả

## 🎯 **Kết Quả Mong Đợi**

- **Dataset**: 10,000 samples với phân loại 2 tầng
- **Models**: SVM cho Level 1 và Level 2
- **Accuracy**: Kết quả phân loại cho cả 2 tầng
- **Files**: Models và kết quả được lưu trong thư mục tương ứng

## 🔧 **Tùy Chỉnh**

Bạn có thể thay đổi:
- `target_size`: Số lượng samples (mặc định: 10,000)
- SVM parameters: kernel, C, gamma
- Feature selection: k_best

## 📊 **Cấu Trúc Output**

```
models/
├── saved_models/
│   ├── level1_classifier/svm_level1/
│   └── level2_classifier/svm_level2/
results/
├── training_results/
└── evaluation_results/
data/
├── processed/
│   └── dataset_splits/
└── raw/
```

## 🚀 **Tiếp Theo**

Sau khi hoàn thành SVM, bạn có thể:
1. Training PhoBERT
2. Training BiLSTM  
3. Training Ensemble
4. So sánh kết quả

---

**🎉 Chúc bạn thành công với viLegalBert!** 