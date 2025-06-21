# Vietnamese Legal NLP

Hệ thống xử lý ngôn ngữ tự nhiên toàn diện được thiết kế đặc biệt cho văn bản pháp lý tiếng Việt, tích hợp khả năng OCR tiên tiến, xử lý văn bản và các mô hình máy học để phân tích tài liệu pháp lý.

## 🌟 Tính năng chính

### Xử lý OCR
- **Hỗ trợ đa engine OCR**: Tesseract, PaddleOCR, EasyOCR, Google Vision API, Azure Computer Vision
- **Phương pháp Ensemble OCR**: Voting, trọng số theo độ tin cậy, và ensemble dựa trên ML
- **Tiền xử lý nâng cao**: Cải thiện ảnh, loại bỏ nhiễu, sửa độ nghiêng, chuyển đổi nhị phân
- **Hậu xử lý thông minh**: Kiểm tra chính tả, sửa lỗi, lọc theo độ tin cậy, ghép nối văn bản
- **Đánh giá chất lượng**: Metrics độ chính xác, phân tích lỗi, benchmark hiệu suất

### Khả năng NLP


### Xử lý tài liệu

## 🏗️ Kiến trúc hệ thống

```
viLegalBert/
├── src/                    # Mã nguồn chính
│   ├── ocr/               # Module xử lý OCR
│   ├── data_collection/   # Thu thập và crawl dữ liệu
│   ├── data_processing/   # Xử lý và làm sạch văn bản
│   ├── models/           # Mô hình ML/DL
│   ├── training/         # Pipeline huấn luyện mô hình
│   ├── evaluation/       # Đánh giá mô hình
│   ├── inference/        # Suy luận production
│   └── api/              # Dịch vụ REST API
├── data/                  # Lưu trữ và quản lý dữ liệu
    ├── raw/
├── config/               # File cấu hình
├── tests/                # Bộ test toàn diện
├── scripts/              # Scripts tiện ích
├── notebooks/            # Jupyter notebooks phân tích
├── docs/                 # Tài liệu
└── deployment/           # Cấu hình triển khai
```

## 🚀 Bắt đầu nhanh

### Yêu cầu hệ thống

```bash
# Python 3.8+
python --version
```

4. **Thiết lập OCR engines**
```bash
# Cài đặt Tesseract
sudo apt-get install tesseract-ocr tesseract-ocr-vie  # Ubuntu/Debian
# hoặc
brew install tesseract tesseract-lang  # macOS

# Cài đặt thêm dependencies OCR
pip install paddlepaddle paddleocr easyocr
```

5. **Khởi tạo database**
```bash
python -m src.database.migrations.init_db
```

### Thiết lập Docker

```bash
# Build và khởi động services
docker-compose up -d

# Truy cập API tại http://localhost:8000
# Truy cập database tại localhost:5432
```

## 📖 Hướng dẫn sử dụng

### Xử lý OCR

```python
from src.ocr.engines import TesseractOCR, PaddleOCR
from src.ocr.ensemble import VotingEnsemble

# OCR với một engine
tesseract = TesseractOCR()
text = tesseract.extract_text("path/to/document.pdf")

# Ensemble OCR để độ chính xác cao hơn
ensemble = VotingEnsemble([
    TesseractOCR(),
    PaddleOCR(),
    EasyOCR()
])
result = ensemble.extract_text("path/to/document.pdf")
```

### Pipeline xử lý tài liệu

```python
from src.inference.pipelines import PDFToNLPPipeline

# Pipeline hoàn chỉnh từ PDF đến NLP
pipeline = PDFToNLPPipeline()
result = pipeline.process("legal_document.pdf")

print(result.entities)        # Thực thể có tên
print(result.classification)  # Phân loại tài liệu
print(result.summary)        # Tóm tắt tài liệu
```

### Sử dụng API

```bash
# Khởi động API server
uvicorn src.api.app:app --reload

# Upload và xử lý tài liệu
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf"

# Xử lý OCR
curl -X POST "http://localhost:8000/api/v1/ocr/process" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "document.pdf", "engines": ["tesseract", "paddleocr"]}'
```

### Xử lý hàng loạt

```bash
# OCR hàng loạt
python scripts/ocr_processing/batch_ocr.py \
    --input_dir data/raw/pdf/ \
    --output_dir data/ocr_output/ \
    --engines tesseract paddleocr

# So sánh OCR engines
python scripts/ocr_processing/compare_engines.py \
    --test_set data/test_images/ \
    --ground_truth data/annotated/ocr_correction/ground_truth/
```

## 🔧 Cấu hình

### Cấu hình OCR

Chỉnh sửa `config/ocr_config.py`:

```python
OCR_CONFIG = {
    'tesseract': {
        'lang': 'vie',
        'oem': 3,
        'psm': 6,
        'config': '--tessdata-dir /usr/share/tesseract-ocr/5/tessdata'
    },
    'paddleocr': {
        'use_angle_cls': True,
        'lang': 'vi',
        'use_gpu': True
    },
    'ensemble': {
        'method': 'confidence_weighted',
        'min_confidence': 0.7
    }
}
```

### Cấu hình mô hình

Cấu hình mô hình được lưu trong `config/model_configs/`:

- `phobert_config.json`: Cài đặt mô hình PhoBERT
- `ner_config.yaml`: Cấu hình mô hình NER
- `classification_config.yaml`: Cài đặt mô hình phân loại
- `ocr_models_config.yaml`: Cấu hình mô hình sửa lỗi OCR

## 📊 Quản lý dữ liệu

### Cấu trúc dữ liệu

```
data/
├── raw/                    # Tài liệu gốc
│   ├── pdf/               # File PDF (scan, native, hỗn hợp)
│   ├── images/            # Hình ảnh tài liệu
│   └── html/              # Nội dung crawl từ web
├── ocr_output/            # Kết quả xử lý OCR
├── processed/             # Dữ liệu đã làm sạch và cấu trúc hóa
├── annotated/             # Dữ liệu đã gán nhãn để huấn luyện
└── outputs/               # Kết quả cuối cùng và báo cáo
```

### Thu thập dữ liệu

```bash
# Crawl tài liệu pháp lý
python scripts/data_collection/crawl_legal_docs.py \
    --source_urls config/crawl_urls.txt \
    --output_dir data/raw/html/

# Xử lý file upload
python scripts/data_processing/process_uploads.py \
    --input_dir uploads/ \
    --output_dir data/raw/
```

## 🧪 Kiểm thử

### Chạy test

```bash
# Tất cả test
pytest tests/

# Test theo danh mục
pytest tests/unit/test_ocr/           # Unit test OCR
pytest tests/integration/test_api/    # Integration test API
pytest tests/unit/test_models/        # Test mô hình

# Test đặc biệt cho OCR
pytest tests/unit/test_ocr/test_engines/
pytest tests/integration/test_ocr_pipeline/
```

### Dữ liệu test

Test fixtures có sẵn trong `tests/fixtures/`:
- Hình ảnh mẫu để test OCR
- PDF mẫu để xử lý tài liệu
- Dữ liệu mock cho unit test

## 📈 Giám sát và đánh giá

### Đánh giá chất lượng OCR

```bash
# Đánh giá độ chính xác OCR
python scripts/ocr_processing/evaluate_ocr.py \
    --test_images data/test_images/ \
    --ground_truth data/annotated/ocr_correction/ground_truth/ \
    --output_report data/outputs/ocr_reports/

# Phân tích lỗi OCR
jupyter notebook notebooks/05.5_ocr_error_analysis.ipynb
```

### Hiệu suất mô hình

```bash
# Đánh giá mô hình
python -m src.evaluation.evaluate_model \
    --model_type ner \
    --test_data data/annotated/ner/test.json \
    --model_path models/production/ner_v1.0/

# Tạo báo cáo hiệu suất
python scripts/evaluation/generate_reports.py
```

### Tích hợp MLOps

- **MLflow**: Theo dõi thí nghiệm và versioning mô hình
- **Prometheus**: Thu thập metrics hệ thống
- **Grafana**: Dashboard hiệu suất

## 🚀 Triển khai

### Triển khai production

```bash
# Build container production
docker-compose -f deployment/docker/docker-compose.prod.yml build

# Triển khai với Kubernetes
kubectl apply -f deployment/kubernetes/

# Triển khai với Terraform
cd deployment/terraform/
terraform init
terraform plan
terraform apply
```

### API Endpoints

Các endpoint API chính:

- `POST /api/v1/documents/upload` - Upload tài liệu
- `POST /api/v1/ocr/process` - Xử lý OCR
- `POST /api/v1/ner/extract` - Trích xuất thực thể có tên
- `POST /api/v1/classify` - Phân loại tài liệu
- `GET /api/v1/health` - Kiểm tra sức khỏe hệ thống

## 🤝 Đóng góp

### Thiết lập development

1. Fork repository
2. Tạo feature branch
3. Thực hiện thay đổi
4. Thêm test cho tính năng mới
5. Chạy test suite
6. Gửi pull request

### Tiêu chuẩn code

- Tuân theo PEP 8 cho Python code
- Thêm docstring cho tất cả functions và classes
- Viết unit test cho tính năng mới
- Cập nhật tài liệu khi cần thiết

### Hướng dẫn OCR

Khi làm việc với components OCR:

- Test với nhiều OCR engine
- Validate chất lượng OCR output
- Xem xét yêu cầu tiền xử lý ảnh
- Document đặc điểm hiệu suất

## 📚 Tài liệu

Tài liệu toàn diện có sẵn trong thư mục `docs/`:

- [Hướng dẫn thiết lập](docs/setup_guide.md)
- [Hướng dẫn thiết lập OCR](docs/ocr_setup_guide.md)
- [Thực hành tốt nhất OCR](docs/ocr_best_practices.md)
- [Tài liệu API](docs/api_documentation.md)
- [Kiến trúc mô hình](docs/model_architecture.md)
- [Hướng dẫn annotation](docs/annotation_guidelines/)
- [Hướng dẫn triển khai](docs/deployment_guide.md)

## 📝 Giấy phép

Dự án này được cấp phép theo MIT License - xem file [LICENSE](LICENSE) để biết chi tiết.

## 🙏 Lời cảm ơn

- Nhóm PhoBERT cho các mô hình ngôn ngữ tiếng Việt
- Các nhà phát triển OCR engine (Tesseract, PaddleOCR, EasyOCR)
- Nhà cung cấp tài liệu pháp lý Việt Nam
- Cộng đồng đóng góp mã nguồn mở

## 📞 Hỗ trợ

Để được hỗ trợ và giải đáp thắc mắc:

- Tạo issue trên GitHub
- Xem tài liệu trong thư mục `docs/`
- Xem lại phần FAQ
- Liên hệ nhóm phát triển

---

**Lưu ý**: Dự án này được thiết kế đặc biệt cho tài liệu pháp lý tiếng Việt và bao gồm khả năng xử lý OCR chuyên biệt để xử lý các định dạng tài liệu khác nhau thường gặp trong bối cảnh pháp lý.