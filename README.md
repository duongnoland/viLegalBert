# 🚀 viLegalBert - Mô Hình Phân Loại Văn Bản Pháp Luật Việt Nam

## 📋 Tổng Quan Dự Án

**viLegalBert** là một dự án nghiên cứu và phát triển mô hình phân loại văn bản pháp luật Việt Nam sử dụng kiến trúc phân cấp 2 tầng:

- **Tầng 1**: Phân loại loại văn bản cơ bản (Luật, Nghị định, Thông tư, ...)
- **Tầng 2**: Phân loại domain pháp lý chuyên biệt (Hình sự, Dân sự, Hành chính, ...)

## 🏗️ Cấu Trúc Dự Án Hoàn Chỉnh

```
viLegalBert/
├── 📂 config/                          # Cấu hình dự án
│   ├── config.yaml                     # Cấu hình chính
│   ├── model_configs/                  # Cấu hình cho từng loại model
│   │   ├── phobert_config.yaml        # Cấu hình PhoBERT
│   │   ├── bilstm_config.yaml         # Cấu hình BiLSTM
│   │   └── hierarchical_config.yaml   # Cấu hình mô hình phân cấp
│   └── data_config.yaml               # Cấu hình xử lý dữ liệu
│
├── 📂 data/                            # Dữ liệu và dataset
│   ├── raw/                           # Dữ liệu gốc
│   │   └── vbpl_crawl.json            # Dữ liệu văn bản pháp luật gốc (1.6GB)
│   ├── processed/                      # Dữ liệu đã xử lý
│   ├── embeddings/                     # Vector embeddings
│   │   ├── phobert_embeddings/        # Embeddings từ PhoBERT
│   │   └── custom_embeddings/         # Embeddings tùy chỉnh
│   └── external/                       # Dữ liệu bên ngoài
│
├── 📂 models/                          # Mô hình và weights
│   ├── saved_models/                   # Mô hình đã train
│   │   ├── level1_classifier/         # Classifier tầng 1
│   │   ├── level2_classifier/         # Classifier tầng 2
│   │   └── hierarchical_models/        # Mô hình phân cấp hoàn chỉnh
│   ├── checkpoints/                    # Checkpoints trong quá trình training
│   └── pretrained/                     # Mô hình pretrained
│
├── 📂 src/                             # Mã nguồn chính
│   ├── data/                           # Xử lý dữ liệu
│   ├── models/                         # Kiến trúc mô hình
│   ├── training/                       # Training và optimization
│   ├── evaluation/                     # Đánh giá mô hình
│   └── utils/                          # Tiện ích
│
├── 📂 training/                        # Scripts và notebooks training
│   ├── scripts/                        # Scripts training
│   ├── notebooks/                      # Jupyter notebooks
│   └── experiments/                    # Các thí nghiệm
│
├── 📂 results/                         # Kết quả và báo cáo
│   ├── training_results/                # Kết quả training
│   ├── evaluation_results/              # Kết quả đánh giá
│   ├── visualizations/                  # Biểu đồ và hình ảnh
│   └── reports/                         # Báo cáo tổng hợp
│
├── 📂 logs/                            # Log files
├── 📂 tests/                            # Unit tests và integration tests
├── 📂 docs/                             # Tài liệu dự án
├── 📂 deployment/                       # Triển khai mô hình
└── 📂 scripts/                          # Scripts tiện ích
```

## 🎯 Công Dụng Chi Tiết Của Từng Thư Mục

### 📂 config/
**Mục đích**: Quản lý tất cả cấu hình dự án
**Chức năng**: 
- Hyperparameters cho models
- Đường dẫn file và thư mục
- Cấu hình training (batch size, learning rate, epochs)
- Cấu hình data processing
- Model architectures

**Files chính**:
- `config.yaml`: Cấu hình tổng thể dự án
- `model_configs/phobert_config.yaml`: Cấu hình chi tiết cho PhoBERT
- `model_configs/bilstm_config.yaml`: Cấu hình chi tiết cho BiLSTM
- `model_configs/hierarchical_config.yaml`: Cấu hình mô hình phân cấp 2 tầng
- `data_config.yaml`: Cấu hình xử lý dữ liệu

### 📂 data/
**Mục đích**: Quản lý tất cả dữ liệu của dự án
**Chức năng**:
- `raw/`: Dữ liệu gốc chưa xử lý (vbpl_crawl.json)
- `processed/`: Dữ liệu đã được xử lý và chuẩn bị cho training
- `embeddings/`: Vector representations của văn bản
- `external/`: Dữ liệu bổ sung từ nguồn bên ngoài

### 📂 models/
**Mục đích**: Quản lý tất cả mô hình và weights
**Chức năng**:
- `saved_models/`: Mô hình đã train hoàn chỉnh
  - `level1_classifier/`: Classifier cho tầng 1 (loại văn bản)
  - `level2_classifier/`: Classifier cho tầng 2 (domain pháp lý)
  - `hierarchical_models/`: Mô hình phân cấp hoàn chỉnh
- `checkpoints/`: Trạng thái mô hình trong quá trình training
- `pretrained/`: Mô hình pretrained sẵn (PhoBERT, etc.)

### 📂 src/
**Mục đích**: Chứa mã nguồn chính của dự án
**Chức năng**:
- `data/`: Xử lý và chuẩn bị dữ liệu
- `models/`: Kiến trúc và implementation của các mô hình
- `training/`: Logic training, optimization, loss functions
- `evaluation/`: Đánh giá và phân tích hiệu suất
- `utils/`: Các tiện ích và helper functions

### 📂 training/
**Mục đích**: Quản lý quá trình training và development
**Chức năng**:
- `scripts/`: Scripts để chạy training
- `notebooks/`: Jupyter notebooks cho development và analysis
- `experiments/`: Quản lý các thí nghiệm khác nhau

### 📂 results/
**Mục đích**: Lưu trữ tất cả kết quả và báo cáo
**Chức năng**:
- `training_results/`: Kết quả từ quá trình training
- `evaluation_results/`: Kết quả đánh giá mô hình
- `visualizations/`: Biểu đồ và hình ảnh
- `reports/`: Báo cáo tổng hợp

### 📂 logs/
**Mục đích**: Ghi lại tất cả hoạt động của hệ thống
**Chức năng**: Debugging, monitoring, audit trail

### 📂 tests/
**Mục đích**: Đảm bảo chất lượng code
**Chức năng**: Unit tests, integration tests, validation

### 📂 docs/
**Mục đích**: Tài liệu hóa dự án
**Chức năng**: API docs, user guides, developer guides

### 📂 deployment/
**Mục đích**: Triển khai mô hình vào production
**Chức năng**: REST API, web app, Docker, cloud

## 🚀 Workflow Sử Dụng

### 1. Setup Environment
```bash
# Clone repository
git clone <repository_url>
cd viLegalBert

# Cài đặt dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Chỉnh sửa .env với các giá trị thực tế
```

### 2. Data Processing
```bash
# Khám phá dữ liệu gốc
python load_vbpl_crawl.py

# Tạo dataset phân cấp 2 tầng
python create_hierarchical_dataset.py
```

### 3. Model Development
```bash
# Chạy training cho tầng 1
python training/scripts/train_level1.py

# Chạy training cho tầng 2
python training/scripts/train_level2.py

# Chạy training mô hình phân cấp hoàn chỉnh
python training/scripts/train_hierarchical.py
```

### 4. Evaluation & Analysis
```bash
# Đánh giá mô hình
python src/evaluation/evaluator.py

# Phân tích kết quả
python src/evaluation/performance_analysis.py
```

## 📊 Cấu Trúc Dataset

Mỗi sample trong dataset sẽ có các trường:
- `id`: Mã định danh văn bản
- `text`: Văn bản đầy đủ để phân loại
- `type_level1`: Nhãn tầng 1 (loại văn bản)
- `domain_level2`: Nhãn tầng 2 (domain pháp lý)
- `ministry`: Cơ quan ban hành
- `name`: Tên văn bản
- `chapter`: Tên chương
- `article`: Điều khoản
- `content_length`: Độ dài nội dung

## 🏷️ Hệ Thống Nhãn

### Tầng 1: Loại văn bản cơ bản (10 classes)
- **LUẬT**: Các bộ luật, luật
- **NGHỊ ĐỊNH**: Nghị định của Chính phủ
- **THÔNG TƯ**: Thông tư của các bộ, ngành
- **NGHỊ QUYẾT**: Nghị quyết của Quốc hội, Chính phủ
- **QUYẾT ĐỊNH**: Quyết định hành chính
- **CHỈ THỊ**: Chỉ thị của các cơ quan
- **PHÁP LỆNH**: Pháp lệnh
- **NGHỊ QUYẾT LIÊN TỊCH**: Nghị quyết liên tịch
- **THÔNG TƯ LIÊN TỊCH**: Thông tư liên tịch
- **NGHỊ ĐỊNH LIÊN TỊCH**: Nghị định liên tịch

### Tầng 2: Domain pháp lý chuyên biệt (15 classes)
- **HÌNH SỰ**: Luật hình sự, tội phạm
- **DÂN SỰ**: Luật dân sự, hợp đồng, quyền sở hữu
- **HÀNH CHÍNH**: Luật hành chính, xử phạt vi phạm
- **LAO ĐỘNG**: Luật lao động, hợp đồng lao động
- **THUẾ**: Luật thuế, khai thuế
- **DOANH NGHIỆP**: Luật doanh nghiệp, công ty
- **ĐẤT ĐAI**: Luật đất đai, quyền sử dụng đất
- **XÂY DỰNG**: Luật xây dựng, quy hoạch
- **GIAO THÔNG**: Luật giao thông, vi phạm giao thông
- **Y TẾ**: Luật y tế, khám chữa bệnh
- **GIÁO DỤC**: Luật giáo dục, đào tạo
- **TÀI CHÍNH**: Luật tài chính, ngân hàng
- **MÔI TRƯỜNG**: Luật môi trường, bảo vệ môi trường
- **AN NINH**: Luật an ninh, quốc phòng
- **KHÁC**: Các domain khác

## 🔧 Dependencies

Xem file `requirements.txt` để biết các thư viện cần thiết.

## 📝 Ghi Chú

- Dữ liệu gốc từ `vbpl_crawl.json` chứa 515,188 items
- Dataset được tạo bằng cách lấy mẫu ngẫu nhiên 10,000 items
- Các nhãn được tạo tự động dựa trên từ khóa và nội dung
- Workspace đã được dọn dẹp và tổ chức lại hoàn toàn
- Cấu trúc tuân theo best practices của ML projects

## 🤝 Đóng Góp

Dự án này được thiết kế để dễ dàng mở rộng và đóng góp. Hãy đọc `CONTRIBUTING.md` để biết thêm chi tiết.

## 📄 License

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết. 