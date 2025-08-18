🚀 VILEGALBERT PIPELINE - GPU OPTIMIZED
============================================================

🚀 BƯỚC 1: GPU SETUP
🚀 GPU CUDA available!
📊 GPU Device: NVIDIA A100-SXM4-40GB
📊 GPU Memory: 39.6 GB

📦 BƯỚC 2: CÀI ĐẶT DEPENDENCIES
📦 Cài đặt scikit-learn...
✅ scikit-learn đã cài đặt xong
✅ pandas đã có sẵn
✅ numpy đã có sẵn
✅ joblib đã có sẵn

🏗️ BƯỚC 3: TẠO THƯ MỤC
✅ Đã tạo thư mục: /content/viLegalBert/models/saved_models/level1_classifier/svm_level1
✅ Đã tạo thư mục: /content/viLegalBert/models/saved_models/level2_classifier/svm_level2
✅ Đã tạo thư mục: /content/viLegalBert/data/processed/dataset_splits

🔄 BƯỚC 4: KIỂM TRA SPLITS
✅ Dataset splits đã có sẵn:
📊 Train set: 3500 samples
📊 Validation set: 750 samples
📊 Test set: 750 samples

🏋️ BƯỚC 5: TRAINING SVM
🚀 GPU CUDA available!
📊 GPU Device: NVIDIA A100-SXM4-40GB
📊 GPU Memory: 39.6 GB
🚀 SVMTrainer - GPU: ✅
🏷️ Training Level 1...
🏋️ Bắt đầu training SVM...
📊 Progress: Training với Grid Search...
⏳ 0% - Khởi tạo Grid Search...
Fitting 5 folds for each of 12 candidates, totalling 60 fits
✅ 100% - Grid Search hoàn thành!
✅ Best params: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
📊 Progress: Đánh giá model...
⏳ 80% - Prediction trên validation set...
⏳ 90% - Tính toán accuracy...
✅ 100% - Evaluation hoàn thành!
✅ Level 1 Accuracy: 0.7760
              precision    recall  f1-score   support

        KHÁC       0.43      0.08      0.13        39
        LUẬT       0.70      0.46      0.55        50
  NGHỊ QUYẾT       0.76      0.83      0.79       115
   NGHỊ ĐỊNH       0.70      0.72      0.71       123
   PHÁP LỆNH       1.00      0.17      0.29         6
  QUYẾT ĐỊNH       0.86      0.88      0.87       202
    THÔNG TƯ       0.77      0.90      0.83       215

    accuracy                           0.78       750
   macro avg       0.75      0.58      0.60       750
weighted avg       0.76      0.78      0.76       750

💾 Model đã lưu: /content/viLegalBert/models/saved_models/level1_classifier/svm_level1/svm_level1_model.pkl
🏷️ Training Level 2...
🏋️ Bắt đầu training SVM Level 2...
📊 Progress: Training với Grid Search...
⏳ 0% - Khởi tạo Grid Search...
Fitting 5 folds for each of 12 candidates, totalling 60 fits
✅ 100% - Grid Search hoàn thành!
📊 Progress: Đánh giá model...
⏳ 80% - Prediction trên validation set...
⏳ 90% - Tính toán accuracy...
✅ 100% - Evaluation hoàn thành!
✅ Level 2 Accuracy: 0.5893
              precision    recall  f1-score   support

     AN NINH       0.00      0.00      0.00        10
DOANH NGHIỆP       0.67      0.22      0.33        18
      DÂN SỰ       0.50      0.42      0.46        52
  GIAO THÔNG       0.50      0.43      0.46         7
    GIÁO DỤC       0.31      0.21      0.25        19
  HÀNH CHÍNH       0.59      0.86      0.70       332
     HÌNH SỰ       0.60      0.21      0.32        28
        KHÁC       0.63      0.55      0.59       145
    LAO ĐỘNG       0.73      0.33      0.46        24
  MÔI TRƯỜNG       0.67      0.17      0.27        12
        THUẾ       1.00      0.20      0.33        15
   TÀI CHÍNH       0.59      0.28      0.38        36
    XÂY DỰNG       0.58      0.31      0.40        36
        Y TẾ       0.50      0.10      0.17        10
     ĐẤT ĐAI       1.00      0.33      0.50         6

    accuracy                           0.59       750
   macro avg       0.59      0.31      0.37       750
weighted avg       0.59      0.59      0.55       750

💾 Model đã lưu: /content/viLegalBert/models/saved_models/level2_classifier/svm_level2/svm_level2_model.pkl

📊 BƯỚC 6: EVALUATION
📊 Đánh giá models...
🏷️ Level 1 Test Accuracy: 0.7827
🏷️ Level 2 Test Accuracy: 0.6267
❌ Lỗi khi đánh giá: [Errno 2] No such file or directory: '/content/viLegalBert/results/evaluation_results/svm_evaluation_results.pkl'

🎉 PIPELINE HOÀN THÀNH!
📊 Dataset: /content/viLegalBert/data/processed/dataset_splits/train.csv
🏷️ Level 1 Accuracy: 0.7760
🏷️ Level 2 Accuracy: 0.5893
🚀 GPU Status: ✅ Available