# 🤝 Hướng Dẫn Đóng Góp - viLegalBert

## 📋 Tổng Quan

Cảm ơn bạn quan tâm đến việc đóng góp cho dự án **viLegalBert**! Dự án này nhằm mục tiêu phát triển mô hình phân loại văn bản pháp luật Việt Nam với kiến trúc phân cấp 2 tầng.

## 🎯 Cách Thức Đóng Góp

### 1. **Báo Cáo Bug**
- Sử dụng [GitHub Issues](https://github.com/yourusername/viLegalBert/issues)
- Mô tả chi tiết bug và cách tái tạo
- Bao gồm thông tin về hệ thống và phiên bản

### 2. **Đề Xuất Tính Năng**
- Tạo issue với label "enhancement"
- Mô tả chi tiết tính năng mong muốn
- Giải thích lý do và lợi ích

### 3. **Đóng Góp Code**
- Fork repository
- Tạo feature branch
- Commit và push thay đổi
- Tạo Pull Request

## 🏗️ Cấu Trúc Dự Án

### **Thư Mục Chính**
```
viLegalBert/
├── 📂 config/          # Cấu hình dự án
├── 📂 data/            # Dữ liệu và dataset
├── 📂 models/          # Mô hình và weights
├── 📂 src/             # Mã nguồn chính
├── 📂 training/        # Scripts training
├── 📂 results/         # Kết quả và báo cáo
├── 📂 logs/            # Log files
├── 📂 tests/           # Unit tests
├── 📂 docs/            # Tài liệu
└── 📂 deployment/      # Triển khai
```

### **Mô Hình Hỗ Trợ**
- **PhoBERT**: Transformer-based model
- **BiLSTM**: Recurrent Neural Network
- **SVM**: Support Vector Machine
- **Ensemble**: Kết hợp nhiều mô hình

## 🚀 Quy Trình Phát Triển

### 1. **Setup Development Environment**
```bash
# Clone repository
git clone https://github.com/yourusername/viLegalBert.git
cd viLegalBert

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt package (development mode)
pip install -e .
```

### 2. **Tạo Feature Branch**
```bash
# Cập nhật main branch
git checkout main
git pull origin main

# Tạo feature branch
git checkout -b feature/your-feature-name
```

### 3. **Phát Triển Tính Năng**
- Tuân thủ coding standards
- Viết tests cho code mới
- Cập nhật documentation
- Commit thường xuyên với message rõ ràng

### 4. **Testing**
```bash
# Chạy unit tests
pytest tests/

# Chạy với coverage
pytest --cov=src tests/

# Kiểm tra code style
black src/
flake8 src/
```

### 5. **Commit và Push**
```bash
# Add changes
git add .

# Commit với message rõ ràng
git commit -m "feat: add new feature description"

# Push to remote
git push origin feature/your-feature-name
```

### 6. **Tạo Pull Request**
- Tạo PR từ feature branch vào main
- Mô tả chi tiết thay đổi
- Liên kết với issues liên quan
- Request review từ maintainers

## 📝 Coding Standards

### **Python Code Style**
- Tuân thủ PEP 8
- Sử dụng type hints
- Viết docstrings cho functions/classes
- Giới hạn độ dài dòng: 88 characters (Black)

### **File Naming**
- Sử dụng snake_case cho Python files
- Sử dụng PascalCase cho class names
- Sử dụng UPPER_CASE cho constants

### **Import Organization**
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np

# Local imports
from .base_model import BaseModel
from .utils import helper_function
```

### **Documentation**
- Viết docstrings cho tất cả public functions/classes
- Sử dụng Google style docstrings
- Cập nhật README.md khi cần thiết

## 🧪 Testing Guidelines

### **Unit Tests**
- Viết tests cho tất cả functions/classes
- Sử dụng pytest framework
- Đặt tên test functions rõ ràng
- Test cả happy path và edge cases

### **Test Structure**
```
tests/
├── unit/                    # Unit tests
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_training.py
├── integration/             # Integration tests
└── test_data/              # Test data files
```

### **Test Examples**
```python
def test_data_loader_loads_data_correctly():
    """Test that DataLoader loads data correctly"""
    loader = DataLoader("test_data.csv")
    data = loader.load()
    
    assert len(data) > 0
    assert "text" in data.columns
    assert "label" in data.columns

def test_model_predicts_correctly():
    """Test that model makes predictions correctly"""
    model = SVMClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert all(isinstance(pred, str) for pred in predictions)
```

## 📊 Performance Guidelines

### **Code Performance**
- Sử dụng vectorized operations khi có thể
- Tránh loops không cần thiết
- Sử dụng appropriate data structures
- Profile code khi cần thiết

### **Memory Management**
- Giải phóng memory khi không cần thiết
- Sử dụng generators cho large datasets
- Batch processing khi có thể

## 🔒 Security Guidelines

### **Data Security**
- Không commit sensitive data
- Sử dụng environment variables cho secrets
- Validate input data
- Sanitize output data

### **Code Security**
- Tránh SQL injection
- Validate file uploads
- Sử dụng secure random generators
- Cập nhật dependencies thường xuyên

## 📚 Documentation Guidelines

### **Code Documentation**
- Viết docstrings cho tất cả public APIs
- Sử dụng Google style docstrings
- Bao gồm examples trong docstrings

### **Project Documentation**
- Cập nhật README.md
- Viết API documentation
- Tạo user guides
- Maintain changelog

## 🚀 Deployment Guidelines

### **Model Deployment**
- Version control cho models
- Model validation trước deployment
- Monitoring model performance
- Rollback strategy

### **API Development**
- RESTful API design
- Input validation
- Error handling
- Rate limiting

## 🤝 Review Process

### **Code Review Checklist**
- [ ] Code follows style guidelines
- [ ] Tests are written and passing
- [ ] Documentation is updated
- [ ] No sensitive data is exposed
- [ ] Performance considerations addressed
- [ ] Security considerations addressed

### **Review Process**
1. **Automated Checks**: CI/CD pipeline
2. **Peer Review**: Code review từ team members
3. **Maintainer Review**: Final review từ maintainers
4. **Merge**: Merge vào main branch

## 📞 Liên Hệ

- **Issues**: [GitHub Issues](https://github.com/yourusername/viLegalBert/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/viLegalBert/discussions)
- **Email**: your.email@example.com

## 🙏 Cảm Ơn

Cảm ơn bạn đã đóng góp cho dự án viLegalBert! Mọi đóng góp, dù nhỏ hay lớn, đều rất quý giá và giúp dự án phát triển tốt hơn.

---

**Happy Contributing! 🎉** 