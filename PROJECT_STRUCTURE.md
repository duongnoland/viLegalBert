# 🏗️ Cấu Trúc Dự Án viLegalBert - Phiên Bản Hoàn Chỉnh

## 📁 Cấu Trúc Thư Mục Chi Tiết

```
viLegalBert/
├── 📂 config/                          # Cấu hình dự án
│   ├── config.yaml                     # Cấu hình chính (hyperparameters, paths)
│   ├── model_configs/                  # Cấu hình cho từng loại model
│   │   ├── phobert_config.yaml        # Cấu hình PhoBERT
│   │   ├── bilstm_config.yaml         # Cấu hình BiLSTM
│   │   ├── svm_config.yaml            # Cấu hình SVM
│   │   └── hierarchical_config.yaml   # Cấu hình mô hình phân cấp
│   └── data_config.yaml               # Cấu hình xử lý dữ liệu
│
├── 📂 data/                            # Dữ liệu và dataset
│   ├── raw/                           # Dữ liệu gốc
│   │   ├── vbpl_crawl.json            # Dữ liệu văn bản pháp luật gốc (1.6GB)
│   │   └── legal_domains/             # Thư mục chứa domain pháp lý
│   ├── processed/                      # Dữ liệu đã xử lý
│   │   ├── hierarchical_dataset.csv   # Dataset phân cấp 2 tầng
│   │   ├── train.csv                  # Tập training
│   │   ├── validation.csv             # Tập validation
│   │   └── test.csv                   # Tập test
│   ├── embeddings/                     # Vector embeddings đã tạo
│   │   ├── phobert_embeddings/        # Embeddings từ PhoBERT
│   │   └── custom_embeddings/         # Embeddings tùy chỉnh
│   └── external/                       # Dữ liệu bên ngoài (nếu có)
│
├── 📂 models/                          # Mô hình và weights
│   ├── saved_models/                   # Mô hình đã train
│   │   ├── level1_classifier/         # Classifier tầng 1
│   │   │   ├── phobert_level1/        # PhoBERT cho tầng 1
│   │   │   ├── bilstm_level1/         # BiLSTM cho tầng 1
│   │   │   ├── svm_level1/            # SVM cho tầng 1
│   │   │   └── ensemble_level1/       # Ensemble model tầng 1
│   │   ├── level2_classifier/         # Classifier tầng 2
│   │   │   ├── phobert_level2/        # PhoBERT cho tầng 2
│   │   │   ├── bilstm_level2/         # BiLSTM cho tầng 2
│   │   │   ├── svm_level2/            # SVM cho tầng 2
│   │   │   └── domain_specific/       # Model chuyên biệt theo domain
│   │   └── hierarchical_models/        # Mô hình phân cấp hoàn chỉnh
│   ├── checkpoints/                    # Checkpoints trong quá trình training
│   │   ├── level1_checkpoints/        # Checkpoints tầng 1
│   │   └── level2_checkpoints/        # Checkpoints tầng 2
│   └── pretrained/                     # Mô hình pretrained
│       ├── phobert_base/               # PhoBERT base model
│       └── custom_pretrained/          # Mô hình pretrained tùy chỉnh
│
├── 📂 src/                             # Mã nguồn chính
│   ├── data/                           # Xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── data_loader.py              # Load và preprocess dữ liệu
│   │   ├── data_processor.py           # Xử lý và chuẩn bị dữ liệu
│   │   ├── text_preprocessing.py       # Tiền xử lý văn bản
│   │   ├── augmentation.py             # Data augmentation
│   │   └── dataset.py                  # Dataset classes
│   ├── models/                         # Kiến trúc mô hình
│   │   ├── __init__.py
│   │   ├── base_model.py               # Base class cho tất cả models
│   │   ├── phobert_classifier.py       # PhoBERT classifier
│   │   ├── bilstm_classifier.py        # BiLSTM classifier
│   │   ├── svm_classifier.py           # SVM classifier
│   │   ├── hierarchical_classifier.py  # Mô hình phân cấp 2 tầng
│   │   ├── ensemble_model.py           # Ensemble methods
│   │   └── attention_mechanisms.py     # Attention mechanisms
│   ├── training/                       # Training và optimization
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Training loop chính
│   │   ├── optimizer.py                # Optimizers và schedulers
│   │   ├── loss_functions.py           # Loss functions
│   │   ├── metrics.py                  # Evaluation metrics
│   │   └── callbacks.py                # Training callbacks
│   ├── evaluation/                     # Đánh giá mô hình
│   │   ├── __init__.py
│   │   ├── evaluator.py                # Evaluator chính
│   │   ├── performance_analysis.py     # Phân tích hiệu suất
│   │   ├── confusion_matrix.py         # Confusion matrix
│   │   └── error_analysis.py           # Phân tích lỗi
│   ├── utils/                          # Tiện ích
│   │   ├── __init__.py
│   │   ├── logger.py                   # Logging system
│   │   ├── visualization.py            # Visualization tools
│   │   ├── metrics_utils.py            # Utility functions cho metrics
│   │   └── file_utils.py               # File operations
│   └── main.py                         # Entry point chính
│
├── 📂 training/                        # Scripts và notebooks training
│   ├── scripts/                        # Scripts training
│   │   ├── train_level1.py             # Train classifier tầng 1
│   │   ├── train_level2.py             # Train classifier tầng 2
│   │   ├── train_svm.py                # Train SVM model
│   │   ├── train_hierarchical.py       # Train mô hình phân cấp hoàn chỉnh
│   │   ├── fine_tune_phobert.py        # Fine-tune PhoBERT
│   │   └── hyperparameter_tuning.py    # Hyperparameter optimization
│   ├── notebooks/                      # Jupyter notebooks
│   │   ├── data_exploration.ipynb      # Khám phá dữ liệu
│   │   ├── model_development.ipynb     # Phát triển mô hình
│   │   ├── training_analysis.ipynb     # Phân tích training
│   │   └── evaluation_results.ipynb    # Kết quả đánh giá
│   └── experiments/                    # Các thí nghiệm
│       ├── experiment_001/              # Thí nghiệm 1
│       ├── experiment_002/              # Thí nghiệm 2
│       └── experiment_configs/          # Cấu hình thí nghiệm
│
├── 📂 results/                         # Kết quả và báo cáo
│   ├── training_results/                # Kết quả training
│   │   ├── level1_results/             # Kết quả tầng 1
│   │   ├── level2_results/             # Kết quả tầng 2
│   │   └── hierarchical_results/       # Kết quả mô hình phân cấp
│   ├── evaluation_results/              # Kết quả đánh giá
│   │   ├── performance_metrics/         # Metrics hiệu suất
│   │   ├── confusion_matrices/         # Confusion matrices
│   │   ├── error_analysis/             # Phân tích lỗi
│   │   └── comparison_reports/         # Báo cáo so sánh
│   ├── visualizations/                  # Biểu đồ và hình ảnh
│   │   ├── training_curves/             # Đường cong training
│   │   ├── performance_charts/          # Biểu đồ hiệu suất
│   │   └── confusion_plots/            # Biểu đồ confusion matrix
│   └── reports/                         # Báo cáo tổng hợp
│       ├── training_summary.md          # Tóm tắt training
│       ├── evaluation_report.md         # Báo cáo đánh giá
│       └── final_report.md              # Báo cáo cuối cùng
│
├── 📂 logs/                            # Log files
│   ├── training_logs/                   # Logs training
│   ├── evaluation_logs/                 # Logs evaluation
│   └── system_logs/                     # Logs hệ thống
│
├── 📂 tests/                            # Unit tests và integration tests
│   ├── unit/                            # Unit tests
│   │   ├── test_data_loader.py
│   │   ├── test_models.py
│   │   ├── test_training.py
│   │   └── test_evaluation.py
│   ├── integration/                     # Integration tests
│   └── test_data/                       # Dữ liệu test
│
├── 📂 docs/                             # Tài liệu dự án
│   ├── api/                             # API documentation
│   ├── user_guide/                      # Hướng dẫn sử dụng
│   ├── developer_guide/                 # Hướng dẫn developer
│   └── research/                        # Tài liệu nghiên cứu
│
├── 📂 deployment/                       # Triển khai mô hình
│   ├── api/                             # REST API
│   ├── web_app/                         # Web application
│   ├── docker/                          # Docker containers
│   └── cloud/                           # Cloud deployment
│
├── 📂 scripts/                          # Scripts tiện ích
│   ├── setup.sh                         # Setup environment
│   ├── run_training.sh                  # Chạy training
│   ├── run_evaluation.sh                # Chạy evaluation
│   └── cleanup.sh                       # Dọn dẹp workspace
│
├── 📄 requirements.txt                  # Python dependencies
├── 📄 setup.py                          # Package setup
├── 📄 .env.example                      # Environment variables template
├── 📄 .gitignore                        # Git ignore rules
├── 📄 README.md                         # Hướng dẫn chính
└── 📄 PROJECT_STRUCTURE.md              # Tài liệu này
```

## 🎯 Công Dụng Chi Tiết Của Từng Thư Mục

### 📂 config/
- **Mục đích**: Quản lý tất cả cấu hình dự án
- **Chức năng**: 
  - Hyperparameters cho models
  - Đường dẫn file và thư mục
  - Cấu hình training (batch size, learning rate, epochs)
  - Cấu hình data processing
  - Model architectures

**Files chính**:
- `config.yaml`: Cấu hình tổng thể dự án
- `model_configs/phobert_config.yaml`: Cấu hình chi tiết cho PhoBERT
- `model_configs/bilstm_config.yaml`: Cấu hình chi tiết cho BiLSTM
- `model_configs/svm_config.yaml`: Cấu hình chi tiết cho SVM
- `model_configs/hierarchical_config.yaml`: Cấu hình mô hình phân cấp 2 tầng
- `data_config.yaml`: Cấu hình xử lý dữ liệu

### 📂 data/
- **raw/**: Dữ liệu gốc chưa xử lý
- **processed/**: Dữ liệu đã được xử lý và chuẩn bị cho training
- **embeddings/**: Vector representations của văn bản
- **external/**: Dữ liệu bổ sung từ nguồn bên ngoài

### 📂 models/
- **saved_models/**: Mô hình đã train hoàn chỉnh
  - `level1_classifier/`: Classifier cho tầng 1 (loại văn bản)
    - `phobert_level1/`: PhoBERT cho tầng 1
    - `bilstm_level1/`: BiLSTM cho tầng 1
    - `svm_level1/`: SVM cho tầng 1
    - `ensemble_level1/`: Ensemble model tầng 1
  - `level2_classifier/`: Classifier cho tầng 2 (domain pháp lý)
    - `phobert_level2/`: PhoBERT cho tầng 2
    - `bilstm_level2/`: BiLSTM cho tầng 2
    - `svm_level2/`: SVM cho tầng 2
    - `domain_specific/`: Model chuyên biệt theo domain
  - `hierarchical_models/`: Mô hình phân cấp hoàn chỉnh
- **checkpoints/**: Trạng thái mô hình trong quá trình training
- **pretrained/**: Mô hình pretrained sẵn (PhoBERT, etc.)

### 📂 src/
- **data/**: Xử lý và chuẩn bị dữ liệu
- **models/**: Kiến trúc và implementation của các mô hình
  - `phobert_classifier.py`: PhoBERT classifier
  - `bilstm_classifier.py`: BiLSTM classifier
  - `svm_classifier.py`: SVM classifier
  - `hierarchical_classifier.py`: Mô hình phân cấp 2 tầng
- **training/**: Logic training, optimization, loss functions
- **evaluation/**: Đánh giá và phân tích hiệu suất
- **utils/**: Các tiện ích và helper functions

### 📂 training/
- **scripts/**: Scripts để chạy training
  - `train_svm.py`: Script training cho SVM
  - `train_level1.py`: Train classifier tầng 1
  - `train_level2.py`: Train classifier tầng 2
- **notebooks/**: Jupyter notebooks cho development và analysis
- **experiments/**: Quản lý các thí nghiệm khác nhau

### 📂 results/
- **training_results/**: Kết quả từ quá trình training
- **evaluation_results/**: Kết quả đánh giá mô hình
- **visualizations/**: Biểu đồ và hình ảnh
- **reports/**: Báo cáo tổng hợp

### 📂 logs/
- **Mục đích**: Ghi lại tất cả hoạt động của hệ thống
- **Chức năng**: Debugging, monitoring, audit trail

### 📂 tests/
- **Mục đích**: Đảm bảo chất lượng code
- **Chức năng**: Unit tests, integration tests, validation

### 📂 docs/
- **Mục đích**: Tài liệu hóa dự án
- **Chức năng**: API docs, user guides, developer guides

### 📂 deployment/
- **Mục đích**: Triển khai mô hình vào production
- **Chức năng**: REST API, web app, Docker, cloud

## 🚀 Workflow Sử Dụng

1. **Setup**: `scripts/setup.sh`
2. **Data Processing**: `src/data/`
3. **Model Development**: `src/models/`
4. **Training**: `training/scripts/`
   - `train_svm.py`: Training SVM
   - `train_level1.py`: Training tầng 1
   - `train_level2.py`: Training tầng 2
5. **Evaluation**: `src/evaluation/`
6. **Results Analysis**: `results/`
7. **Deployment**: `deployment/`

## 📊 Lợi Ích Của Cấu Trúc Này

- **Tách biệt rõ ràng**: Code, data, models, results
- **Dễ maintain**: Mỗi thư mục có mục đích cụ thể
- **Scalable**: Có thể mở rộng dễ dàng
- **Reproducible**: Experiments được quản lý tốt
- **Professional**: Đạt chuẩn industry best practices
- **Đa dạng mô hình**: Hỗ trợ cả Deep Learning (PhoBERT, BiLSTM) và Traditional ML (SVM)
- **Ensemble ready**: Sẵn sàng kết hợp nhiều mô hình 