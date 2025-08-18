---
title: Chương X: Thực nghiệm
---

## Mục tiêu thí nghiệm
- Phân loại hai tầng: Level 1 (`type_level1`), Level 2 (`domain_level2`).
- So sánh: SVM (TF‑IDF), BiLSTM (TF‑IDF/Embedding), PhoBERT.
- Ablation: `max_length`, `num_epochs`, `class_weights`.

## Dữ liệu
### Nguồn và tiền xử lý
- Mô tả nguồn dữ liệu, quy trình làm sạch và tạo splits.
- Thư mục: `/content/viLegalBert/data/processed/dataset_splits`.

### Thống kê dữ liệu
| Nhãn | Số mẫu |
|-----:|------:|
| (Điền từ `results/report/stats_type_level1.csv`) | |

| Nhãn | Số mẫu |
|-----:|------:|
| (Điền từ `results/report/stats_domain_level2.csv`) | |

![Length distribution](../../results/report/length_distribution_Train.png)

## Mô hình
- Tóm tắt SVM, BiLSTM, PhoBERT: kiến trúc, tham số, ưu/nhược.

## Thiết lập thực nghiệm
- Phần cứng, phần mềm, phiên bản.
- Siêu tham số chủ đạo.

## Quy trình huấn luyện và đánh giá
- Tách tầng, huấn luyện riêng; thước đo: Accuracy, Macro‑F1, Weighted‑F1.

## Kết quả chính
| Mô hình | Tầng | Accuracy | Macro‑F1 | Weighted‑F1 |
|--------|------|---------:|---------:|------------:|
| (Điền từ `results/report/main_results.csv`) ||||

## Ma trận nhầm lẫn
| | |
|-|-|
| ![SVM L1](../../results/report/cm_svm_level1_type_level1.png) | ![SVM L2](../../results/report/cm_svm_level2_domain_level2.png) |
| ![BiLSTM L1](../../results/report/cm_bilstm_level1.png) | ![BiLSTM L2](../../results/report/cm_bilstm_level2.png) |
| ![PhoBERT L1](../../results/report/cm_phobert_level1.png) | ![PhoBERT L2](../../results/report/cm_phobert_level2.png) |

## Learning curves (PhoBERT)
| | |
|-|-|
| ![LC L1](../../results/report/learning_curves_PhoBERT_Level1.png) | ![LC L2](../../results/report/learning_curves_PhoBERT_Level2.png) |

## Ablation (PhoBERT)
| max_length | num_epochs | class_weights | Eval |
|-----------:|-----------:|:-------------:|:-----|
| (Điền từ `results/report/ablation_phobert_level1.csv`) ||||

## Phân tích lỗi
- Mẫu sai nhãn điển hình, nguyên nhân, đề xuất cải thiện.

## So sánh với công trình liên quan
- Đặt cạnh baseline/related works.

## Thời gian và tài nguyên
- Thời gian train/infer, kích thước model.

## Tái lập
- Lệnh chạy, commit hash, phiên bản libraries.


