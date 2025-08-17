# Phân loại Văn bản Pháp lý

Module này thực hiện so sánh 3 mô hình phân loại văn bản pháp lý tiếng Việt:

## 🎯 Mục tiêu

So sánh hiệu suất của các mô hình:
1. **SVM** (Support Vector Machine) - Mô hình cơ sở với TF-IDF
2. **BiLSTM** - Mô hình Deep Learning
3. **PhoBERT** - Mô hình Transformer được pre-train cho tiếng Việt

## 📊 Metrics đánh giá

- **Accuracy**: Độ chính xác tổng thể
- **Weighted F1-Score**: F1 có trọng số theo số lượng mẫu
- **Macro F1-Score**: F1 trung bình không trọng số

## 🏗️ Cấu trúc

```
src/classification/
├── __init__.py
├── data_loader.py      # Tải và tiền xử lý dữ liệu
├── models.py           # Định nghĩa các mô hình
├── evaluator.py        # Đánh giá và so sánh mô hình  
├── experiment.py       # Script chạy thí nghiệm chính
└── README.md           # Tài liệu này
```

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Dữ liệu cần có định dạng CSV với 2 cột:
- `text`: Văn bản pháp lý
- `label`: Nhãn phân loại

### 3. Chạy thí nghiệm

#### Option A: Sử dụng script

```bash
cd src/classification
python experiment.py
```

#### Option B: Sử dụng Jupyter Notebook

```bash
jupyter notebook notebooks/legal_classification_demo.ipynb
```

### 4. Sử dụng trong code

```python
from src.classification.experiment import LegalTextClassificationExperiment

# Khởi tạo thí nghiệm
experiment = LegalTextClassificationExperiment(
    data_path="data/your_data.csv",
    text_column="text",
    label_column="label"
)

# Chạy thí nghiệm
evaluator, results_df = experiment.run_full_experiment(
    sample_size=10000,
    phobert_epochs=3
)

# Xem kết quả
print(results_df)
```

## 📋 Chi tiết các mô hình

### 1. SVM (Baseline)
- **Vectorization**: TF-IDF với n-gram (1,2)
- **Kernel**: Linear
- **Features**: 5000-10000 từ quan trọng nhất

### 2. BiLSTM
- **Architecture**: Embedding → BiLSTM → Dense
- **Embedding**: 300-dim (có thể dùng pre-trained)
- **LSTM**: 128 hidden units, 2 layers
- **Dropout**: 0.3

### 3. PhoBERT
- **Base model**: vinai/phobert-base
- **Fine-tuning**: Chỉ classifier head
- **Max length**: 256 tokens
- **Learning rate**: 2e-5

## 📈 Kết quả mẫu

| Mô hình | Accuracy | Weighted F1 | Macro F1 |
|---------|----------|-------------|----------|
| SVM     | 0.8234   | 0.8189      | 0.7543   |
| BiLSTM  | 0.8567   | 0.8523      | 0.8012   |
| PhoBERT | 0.9123   | 0.9087      | 0.8876   |

## ⚙️ Cấu hình

### Tham số có thể điều chỉnh:

```python
# Trong experiment.py
SAMPLE_SIZE = 10000      # Số mẫu sử dụng
PHOBERT_EPOCHS = 3       # Số epochs cho PhoBERT
BATCH_SIZE = 16          # Batch size
LEARNING_RATE = 2e-5     # Learning rate

# SVM parameters
MAX_FEATURES = 5000      # Số features TF-IDF
NGRAM_RANGE = (1, 2)     # N-gram range

# PhoBERT parameters
MAX_LENGTH = 256         # Độ dài tối đa input
DROPOUT = 0.3            # Dropout rate
```

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**
   - Giảm `BATCH_SIZE` 
   - Giảm `SAMPLE_SIZE`
   - Giảm `MAX_LENGTH`

2. **File not found**
   - Kiểm tra đường dẫn trong `DATA_PATH`
   - Đảm bảo file CSV có đúng format

3. **Column not found**
   - Kiểm tra tên cột `TEXT_COLUMN` và `LABEL_COLUMN`

4. **Slow training**
   - Sử dụng GPU nếu có
   - Giảm số epochs
   - Giảm kích thước dữ liệu

## 📊 Phân tích kết quả

Module tự động tạo:
- Bảng so sánh các mô hình
- Confusion matrix cho từng mô hình  
- Biểu đồ so sánh hiệu suất
- Báo cáo chi tiết (classification report)
- File CSV với kết quả
- Báo cáo Markdown

## 🎓 Ứng dụng trong báo cáo

### Cấu trúc báo cáo đề xuất:

1. **Giới thiệu**
   - Mục tiêu phân loại văn bản pháp lý
   - So sánh 3 approaches: traditional ML, deep learning, transformer

2. **Phương pháp**
   - Mô tả từng mô hình
   - Metrics đánh giá
   - Cài đặt thí nghiệm

3. **Kết quả**
   - Bảng kết quả
   - Biểu đồ so sánh
   - Confusion matrices

4. **Phân tích**
   - So sánh ưu/nhược điểm
   - Phân tích lỗi
   - Thời gian training vs accuracy

5. **Kết luận**
   - Mô hình tốt nhất
   - Ứng dụng thực tế
   - Hướng phát triển

## 📝 Trích dẫn

Nếu sử dụng PhoBERT:
```
@inproceedings{phobert,
title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
year      = {2020}
}
``` 