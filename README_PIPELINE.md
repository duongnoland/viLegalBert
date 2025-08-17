# 🚀 **viLegalBert - Complete Pipeline cho Google Colab (Dataset Có Sẵn)**

## 📋 **Tổng Quan**

viLegalBert là hệ thống phân loại văn bản pháp luật Việt Nam với kiến trúc phân cấp 2 tầng, được thiết kế để chạy hoàn toàn trên Google Colab **với dataset có sẵn**.

## 🎯 **Kiến Trúc 2 Tầng**

### **🏷️ Tầng 1 (Level 1) - Loại Văn Bản**
- **LUẬT**: Các văn bản luật do Quốc hội ban hành
- **NGHỊ ĐỊNH**: Văn bản do Chính phủ ban hành
- **THÔNG TƯ**: Hướng dẫn của các bộ, ngành
- **QUYẾT ĐỊNH**: Quyết định của các cơ quan có thẩm quyền
- **NGHỊ QUYẾT**: Nghị quyết của Quốc hội
- **PHÁP LỆNH**: Pháp lệnh của Ủy ban thường vụ Quốc hội
- **KHÁC**: Các loại văn bản khác

### **🏷️ Tầng 2 (Level 2) - Domain Pháp Lý**
- **HÌNH SỰ**: Luật hình sự, tội phạm, xử lý vi phạm
- **DÂN SỰ**: Hợp đồng, quyền sở hữu, thừa kế, hôn nhân gia đình
- **HÀNH CHÍNH**: Thủ tục hành chính, xử phạt vi phạm
- **LAO ĐỘNG**: Hợp đồng lao động, tiền lương, bảo hiểm
- **THUẾ**: Thuế thu nhập, thuế giá trị gia tăng, khai thuế
- **DOANH NGHIỆP**: Thành lập, quản lý doanh nghiệp
- **ĐẤT ĐAI**: Quyền sử dụng đất, thủ tục đất đai
- **XÂY DỰNG**: Giấy phép xây dựng, quy hoạch, kiến trúc
- **GIAO THÔNG**: Luật giao thông, vi phạm giao thông
- **Y TẾ**: Khám chữa bệnh, dược phẩm, vệ sinh an toàn thực phẩm
- **GIÁO DỤC**: Đào tạo, chương trình giáo dục, bằng cấp
- **TÀI CHÍNH**: Ngân hàng, tín dụng, tiền tệ, đầu tư
- **MÔI TRƯỜNG**: Bảo vệ môi trường, xử lý chất thải
- **AN NINH**: Quốc phòng, bảo vệ an ninh, trật tự xã hội

## 🏗️ **Các Models Được Hỗ Trợ**

### **1. 🎯 SVM (Support Vector Machine)**
- **Ưu điểm**: Nhanh, hiệu quả với dữ liệu vừa và nhỏ
- **Features**: TF-IDF vectorization, feature selection (Chi2)
- **Kernels**: RBF, Linear, Poly, Sigmoid
- **Tuning**: Grid Search với cross-validation

### **2. 🚀 PhoBERT (Vietnamese BERT)**
- **Ưu điểm**: Hiệu suất cao, hiểu ngữ cảnh tiếng Việt
- **Architecture**: Transformer-based, pre-trained on Vietnamese text
- **Fine-tuning**: Sequence classification với 2 tầng
- **Optimization**: Learning rate scheduling, early stopping

### **3. 🧠 BiLSTM (Bidirectional LSTM)**
- **Ưu điểm**: Xử lý tốt chuỗi dài, attention mechanism
- **Architecture**: Bidirectional LSTM + Attention + Classification head
- **Features**: TF-IDF input, dropout regularization
- **Training**: Adam optimizer, learning rate scheduling

### **4. 🎭 Ensemble Model**
- **Strategy**: Weighted voting của 3 models
- **Weights**: SVM (40%), PhoBERT (30%), BiLSTM (30%)
- **Benefits**: Tăng độ chính xác, giảm overfitting

## 📁 **Files Pipeline (Đã Tối Ưu)**

### **🚀 Core Pipeline Files**
- **`main_colab.py`**: Pipeline cơ bản với SVM (dataset có sẵn)
- **`phobert_colab.py`**: Training PhoBERT models (dataset có sẵn)
- **`bilstm_colab.py`**: Training BiLSTM models (dataset có sẵn)
- **`ensemble_colab.py`**: Tạo và đánh giá ensemble
- **`complete_pipeline_colab.py`**: Pipeline hoàn chỉnh tích hợp tất cả (dataset có sẵn)

### **📖 Documentation**
- **`README_PIPELINE.md`**: File này - Hướng dẫn chi tiết

## 🚀 **Cách Sử Dụng (Dataset Có Sẵn)**

### **Bước 1: Chuẩn Bị Google Colab**
1. Mở [Google Colab](https://colab.research.google.com)
2. Tạo notebook mới
3. **Upload dataset CSV** (không cần JSON gốc)
4. Đảm bảo runtime type là **GPU** (khuyến nghị)

### **Bước 2: Chuẩn Bị Dataset**
Dataset cần có các cột sau:
- **`text`**: Nội dung văn bản
- **`type_level1`**: Loại văn bản (Level 1)
- **`domain_level2`**: Domain pháp lý (Level 2)

### **Bước 3: Chọn Pipeline**

#### **🎯 Option 1: Pipeline Cơ Bản (SVM)**
```python
# Copy toàn bộ main_colab.py vào cell và chạy
# Tự động tìm dataset và training SVM
```

#### **🚀 Option 2: Pipeline Nâng Cao (PhoBERT + BiLSTM)**
```python
# Copy main_colab.py trước
# Sau đó copy phobert_colab.py và chạy
# Cuối cùng copy bilstm_colab.py và chạy
```

#### **🎭 Option 3: Pipeline Hoàn Chỉnh (Tất cả + Ensemble)**
```python
# Copy complete_pipeline_colab.py vào cell và chạy
# Tự động training tất cả models và tạo ensemble
```

### **Bước 4: Chạy Pipeline**
```python
# Chạy cell để khởi động pipeline
# Quá trình sẽ tự động:
# 1. Cài đặt dependencies
# 2. Tạo cấu trúc project
# 3. Tìm và load dataset có sẵn
# 4. Kiểm tra/tạo dataset splits
# 5. Training các models
# 6. Tạo ensemble (nếu chọn)
# 7. Evaluation và báo cáo
```

## 📊 **Dataset Requirements**

### **📋 Format Yêu Cầu**
- **File type**: CSV với encoding UTF-8
- **Required columns**: `text`, `type_level1`, `domain_level2`
- **Optional columns**: `id`, `ministry`, `name`, `chapter`, `article`

### **🔍 Tự Động Tìm Kiếm**
Pipeline sẽ tự động tìm dataset trong các đường dẫn:
1. `data/processed/hierarchical_legal_dataset.csv`
2. `hierarchical_legal_dataset.csv`
3. `data/hierarchical_legal_dataset.csv`
4. `dataset.csv`
5. `legal_dataset.csv`

### **📈 Dataset Splits**
- **Tự động tạo** nếu chưa có
- **Train/Val/Test**: 70/15/15 ratio
- **Stratified sampling** theo Level 1 labels

## 🎯 **Kết Quả Mong Đợi**

### **📈 Dataset Processing**
- **Auto-detection**: Tự động tìm và load dataset
- **Validation**: Kiểm tra columns cần thiết
- **Splits**: Tự động tạo training splits

### **🏆 Performance Metrics**
- **SVM**: Accuracy ~75-85%
- **PhoBERT**: Accuracy ~80-90%
- **BiLSTM**: Accuracy ~75-85%
- **Ensemble**: Accuracy ~85-95%

### **💾 Output Files**
```
models/
├── saved_models/
│   ├── level1_classifier/
│   │   ├── svm_level1/
│   │   ├── phobert_level1/
│   │   └── bilstm_level1/
│   ├── level2_classifier/
│   │   ├── svm_level2/
│   │   ├── phobert_level2/
│   │   └── bilstm_level2/
│   └── hierarchical_models/
│       └── ensemble_model.pkl
results/
├── training_results/
│   └── pipeline_summary_report.pkl
└── evaluation_results/
    └── complete_evaluation_results.pkl
data/
└── processed/
    └── dataset_splits/
        ├── train.csv
        ├── validation.csv
        └── test.csv
```

## 🔧 **Tùy Chỉnh Pipeline**

### **Models Training**
```python
# Trong complete_pipeline_colab.py
self.config = {
    'train_models': ['svm', 'phobert'],  # Chỉ training SVM và PhoBERT
    # ...
}
```

### **Ensemble Configuration**
```python
# Trong ensemble_colab.py
self.config = {
    'weights': [0.5, 0.3, 0.2],  # Thay đổi trọng số
    # ...
}
```

### **Dataset Paths**
```python
# Thêm đường dẫn dataset mới
possible_paths = [
    "your_custom_dataset.csv",
    "data/your_dataset.csv",
    # ...
]
```

## 🎯 **Use Cases**

### **1. 🏛️ Cơ Quan Nhà Nước**
- Phân loại văn bản pháp luật có sẵn
- Tự động routing documents
- Compliance checking

### **2. 💼 Công Ty Luật**
- Phân tích văn bản pháp luật
- Legal research automation
- Document classification

### **3. 🎓 Nghiên Cứu & Giảng Dạy**
- Legal NLP research
- Vietnamese language processing
- Multi-label classification

### **4. 🔍 Công Cụ Tìm Kiếm**
- Legal document search
- Semantic similarity
- Content recommendation

## 🚀 **Tiếp Theo Sau Training**

### **1. 🌐 Web Application**
```python
# Sử dụng Flask/FastAPI
from flask import Flask, request, jsonify
# Load trained models và tạo API endpoints
```

### **2. 📱 Mobile App**
```python
# Export models sang ONNX format
# Tích hợp vào mobile app
```

### **3. 🔌 API Service**
```python
# Deploy models lên cloud
# Tạo RESTful API service
```

### **4. 📊 Dashboard**
```python
# Sử dụng Streamlit/Gradio
# Tạo giao diện người dùng thân thiện
```

## 🔍 **Troubleshooting**

### **Lỗi "Dataset Not Found"**
```python
# Kiểm tra tên file dataset
# Đảm bảo có columns: text, type_level1, domain_level2
# Kiểm tra encoding UTF-8
```

### **Lỗi Memory**
```python
# Giảm batch size trong PhoBERT
'batch_size': 4

# Giảm max_features trong BiLSTM
'max_features': 3000
```

### **Lỗi CUDA**
```python
# Kiểm tra runtime type trong Colab
# Runtime → Change runtime type → GPU
```

### **Lỗi Import**
```python
# Đảm bảo chạy install_dependencies() trước
# Kiểm tra dataset đã upload
```

## 📞 **Hỗ Trợ & Liên Hệ**

### **🔧 Technical Issues**
1. Kiểm tra dataset có đúng format không
2. Đảm bảo columns cần thiết đã có
3. Kiểm tra runtime type (GPU/CPU)
4. Xem logs và error messages

### **📚 Documentation**
- Đọc kỹ README files
- Kiểm tra code comments
- Xem example outputs

### **🚀 Best Practices**
- Sử dụng GPU runtime cho training
- Bắt đầu với dataset nhỏ để test
- Backup models sau khi training thành công
- Monitor training progress

## 🎉 **Lợi Ích Mới (Dataset Có Sẵn)**

- 🚀 **Nhanh chóng**: Không cần tạo dataset từ JSON
- 📊 **Linh hoạt**: Sử dụng dataset có sẵn
- 🔍 **Tự động**: Tự động tìm và validate dataset
- 💾 **Tiết kiệm**: Không cần xử lý dữ liệu gốc
- 📈 **Hiệu quả**: Tập trung vào training models
- 🌍 **Tiếng Việt**: Tối ưu cho văn bản pháp luật

## 🎉 **Kết Luận**

viLegalBert pipeline cho Google Colab (Dataset Có Sẵn) cung cấp:

- 🚀 **Dễ sử dụng**: Copy & paste, chạy trực tiếp
- 📊 **Hoàn chỉnh**: Từ dataset loading đến ensemble
- 🔧 **Linh hoạt**: Dễ dàng tùy chỉnh và mở rộng
- 💾 **Tự động**: Lưu trữ và quản lý models
- 📈 **Hiệu quả**: Kết hợp nhiều approaches
- 🌍 **Tiếng Việt**: Tối ưu cho văn bản pháp luật
- 🔍 **Thông minh**: Tự động tìm và validate dataset

---

**🚀 viLegalBert Pipeline (Dataset Có Sẵn) đã sẵn sàng cho Google Colab!**

**📧 Hỗ trợ**: Kiểm tra logs và documentation  
**🔗 Repository**: Tất cả files đã được tối ưu  
**📅 Version**: 2.0 - Dataset Ready Pipeline 