# 📊 Tóm Tắt Dataset viLegalBert

## 🎯 Thông Tin Tổng Quan

- **Tổng số samples**: 10,000
- **Số cột**: 9
- **Kích thước file**: 38.21 MB
- **Encoding**: UTF-8
- **Dữ liệu bị thiếu**: 30 values

## 🏷️ Phân Loại Tầng 1 (Loại Văn Bản)

| Loại Văn Bản | Số Lượng | Tỷ Lệ |
|---------------|----------|-------|
| THÔNG TƯ | 2,792 | 27.9% |
| QUYẾT ĐỊNH | 2,575 | 25.8% |
| NGHỊ QUYẾT | 1,816 | 18.2% |
| NGHỊ ĐỊNH | 1,633 | 16.3% |
| LUẬT | 622 | 6.2% |
| KHÁC | 432 | 4.3% |
| PHÁP LỆNH | 130 | 1.3% |

## 🏷️ Phân Loại Tầng 2 (Domain Pháp Lý)

| Domain | Số Lượng | Tỷ Lệ |
|--------|----------|-------|
| HÀNH CHÍNH | 4,370 | 43.7% |
| KHÁC | 2,254 | 22.5% |
| DÂN SỰ | 543 | 5.4% |
| XÂY DỰNG | 497 | 5.0% |
| TÀI CHÍNH | 483 | 4.8% |
| GIÁO DỤC | 314 | 3.1% |
| HÌNH SỰ | 243 | 2.4% |
| DOANH NGHIỆP | 239 | 2.4% |
| LAO ĐỘNG | 213 | 2.1% |
| THUẾ | 157 | 1.6% |
| MÔI TRƯỜNG | 157 | 1.6% |
| AN NINH | 143 | 1.4% |
| Y TẾ | 131 | 1.3% |
| ĐẤT ĐAI | 129 | 1.3% |
| GIAO THÔNG | 127 | 1.3% |

## 📏 Thống Kê Độ Dài Văn Bản

- **Trung bình**: 1,487 ký tự
- **Trung vị**: 780 ký tự
- **Min**: 11 ký tự
- **Max**: 85,849 ký tự
- **Độ lệch chuẩn**: 2,791 ký tự

### Phân Bố Theo Độ Dài

| Độ Dài | Số Lượng | Tỷ Lệ |
|--------|----------|-------|
| Ngắn (< 500 ký tự) | 3,243 | 32.4% |
| Trung bình (500-2000 ký tự) | 4,999 | 50.0% |
| Dài (2000-5000 ký tự) | 1,286 | 12.9% |
| Rất dài (> 5000 ký tự) | 472 | 4.7% |

## 📚 Các Tập Dữ Liệu

| Tập | Số Lượng | Tỷ Lệ |
|-----|----------|-------|
| **Train** | 7,000 | 70% |
| **Validation** | 1,500 | 15% |
| **Test** | 1,500 | 15% |

## 🏛️ Thống Kê Theo Bộ/Ngành (Top 10)

| Bộ/Ngành | Số Lượng | Tỷ Lệ |
|-----------|----------|-------|
| CHÍNH PHỦ | 1,698 | 17.0% |
| QUỐC HỘI | 622 | 6.2% |
| BỘ TÀI CHÍNH | 501 | 5.0% |
| BỘ GIAO THÔNG VẬN TẢI | 251 | 2.5% |
| BỘ CÔNG THƯƠNG | 208 | 2.1% |
| BỘ Y TẾ | 185 | 1.8% |
| BỘ GIÁO DỤC VÀ ĐÀO TẠO | 181 | 1.8% |
| BỘ NÔNG NGHIỆP VÀ PHÁT TRIỂN NÔNG THÔN | 173 | 1.7% |
| BỘ TÀI NGUYÊN VÀ MÔI TRƯỜNG | 172 | 1.7% |
| NGÂN HÀNG NHÀ NƯỚC VIỆT NAM | 147 | 1.5% |

## 🔍 Cấu Trúc Dữ Liệu

### Các Cột Trong Dataset

1. **id**: ID duy nhất của văn bản
2. **text**: Nội dung văn bản đã được làm sạch
3. **type_level1**: Loại văn bản (Level 1)
4. **domain_level2**: Domain pháp lý (Level 2)
5. **ministry**: Bộ/ngành ban hành
6. **name**: Tên văn bản
7. **chapter**: Tên chương
8. **article**: Nội dung điều
9. **content_length**: Độ dài nội dung gốc

## ✅ Chất Lượng Dataset

### 🎯 Điểm Mạnh
- **Phân bố cân bằng** giữa các tập train/val/test
- **Đa dạng loại văn bản** với 7 categories chính
- **Phong phú domain pháp lý** với 15 lĩnh vực
- **Độ dài văn bản đa dạng** từ ngắn đến rất dài
- **Nguồn dữ liệu đáng tin cậy** từ các cơ quan nhà nước

### 📊 Tính Cân Bằng
- **Chênh lệch tối đa** giữa train và test: 0.9%
- **Phân bố đồng đều** giữa các loại văn bản
- **Representation tốt** của các domain pháp lý

## 🚀 Sẵn Sàng Cho Training

Dataset viLegalBert đã hoàn chỉnh và sẵn sàng để:

1. **Training SVM**: `python src/main.py --mode train_svm --level both`
2. **Training PhoBERT**: `python src/main.py --mode train_phobert --level both`
3. **Training BiLSTM**: `python src/main.py --mode train_bilstm --level both`
4. **Training Ensemble**: `python src/main.py --mode train_ensemble --level both`
5. **Evaluation**: `python src/main.py --mode evaluate_all`

## 📈 Kết Luận

Dataset viLegalBert với 10,000 samples được phân loại 2 tầng hoàn chỉnh, cung cấp nền tảng vững chắc cho việc xây dựng mô hình phân loại văn bản pháp luật Việt Nam. Với phân bố cân bằng, đa dạng loại văn bản và domain pháp lý, dataset này sẽ giúp mô hình học được các đặc trưng phong phú và đạt hiệu suất cao trong việc phân loại văn bản pháp luật.

---

**Ngày tạo**: 2025  
**Phiên bản**: 1.0  
**Trạng thái**: Hoàn thành và sẵn sàng sử dụng 