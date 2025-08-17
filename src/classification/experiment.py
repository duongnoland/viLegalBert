"""
Script chính để chạy thí nghiệm phân loại văn bản pháp lý
So sánh SVM, BiLSTM, và PhoBERT
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from classification.data_loader import DataLoader
from classification.models import SVMClassifier, PhoBERTClassifier, ModelTrainer
from classification.evaluator import ModelEvaluator
from classification.bilstm_model import BiLSTMPipeline
from classification.phobert_model import PhoBERTPipeline

class LegalTextClassificationExperiment:
    """Thí nghiệm phân loại văn bản pháp lý"""
    
    def __init__(self, data_path, text_column='text', label_column='label'):
        self.data_path = data_path
        self.text_column = text_column
        self.label_column = label_column
        self.results = {}
        
        # Khởi tạo data loader
        self.data_loader = DataLoader(data_path)
        
        # Device cho deep learning
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Sử dụng device: {self.device}")
    
    def load_and_prepare_data(self, sample_size=10000):
        """Tải và chuẩn bị dữ liệu"""
        print("=== CHUẨN BỊ DỮ LIỆU ===")
        
        # Tải dữ liệu
        texts, labels, class_names = self.data_loader.load_data(
            text_column=self.text_column,
            label_column=self.label_column,
            sample_size=sample_size
        )
        
        # Chia dữ liệu
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
            texts, labels, test_size=0.2, val_size=0.1
        )
        
        # Lưu dữ liệu
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        print(f"Số lớp: {self.num_classes}")
        print(f"Tên các lớp: {class_names}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, class_names
    
    def run_svm_experiment(self):
        """Chạy thí nghiệm với SVM"""
        print("\n=== THÍ NGHIỆM SVM ===")
        
        # Khởi tạo mô hình
        svm_model = SVMClassifier(max_features=5000, ngram_range=(1, 2))
        
        # Huấn luyện
        svm_model.train(self.X_train, self.y_train)
        
        # Dự đoán
        y_pred = svm_model.predict(self.X_test)
        y_pred_proba = svm_model.predict_proba(self.X_test)
        
        # Lưu kết quả
        self.results['SVM'] = {
            'model': svm_model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return y_pred, y_pred_proba
    
    def run_phobert_experiment(self, num_epochs=3, batch_size=16, learning_rate=2e-5):
        """Chạy thí nghiệm với PhoBERT"""
        print("\n=== THÍ NGHIỆM PHOBERT ===")
        
        # Khởi tạo PhoBERT pipeline
        phobert_pipeline = PhoBERTPipeline(
            model_name='vinai/phobert-base',
            max_length=256,
            dropout=0.3
        )
        
        # Prepare data - convert numpy arrays to lists if needed
        if hasattr(self.X_train, 'tolist'):
            texts = self.X_train.tolist() + self.X_val.tolist() + self.X_test.tolist()
            labels = self.y_train.tolist() + self.y_val.tolist() + self.y_test.tolist()
        else:
            texts = list(self.X_train) + list(self.X_val) + list(self.X_test)
            labels = list(self.y_train) + list(self.y_val) + list(self.y_test)
        
        X_train_phobert, X_val_phobert, X_test_phobert, y_train_phobert, y_val_phobert, y_test_phobert = \
            phobert_pipeline.prepare_data(texts, labels)
        
        # Create data loaders
        train_loader, val_loader, test_loader = phobert_pipeline.create_data_loaders(
            X_train_phobert, X_val_phobert, X_test_phobert, 
            y_train_phobert, y_val_phobert, y_test_phobert,
            batch_size=batch_size
        )
        
        # Build and train model
        phobert_pipeline.build_model(device=self.device)
        
        # Training
        history = phobert_pipeline.train(train_loader, val_loader, num_epochs, learning_rate)
        
        # Evaluation
        test_loss, test_acc, y_pred, y_true = phobert_pipeline.evaluate(test_loader)
        
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Convert predictions back to original labels
        y_pred_original = phobert_pipeline.label_encoder.inverse_transform(y_pred)
        y_true_original = phobert_pipeline.label_encoder.inverse_transform(y_true)
        
        # Lưu kết quả
        self.results['PhoBERT'] = {
            'model': phobert_pipeline,
            'y_pred': y_pred_original,
            'y_true': y_true_original,
            'training_history': history
        }
        
        return y_pred_original, y_true_original
    
    def run_bilstm_experiment(self, num_epochs=5, batch_size=32, learning_rate=0.001):
        """Chạy thí nghiệm với BiLSTM"""
        print("\n=== THÍ NGHIỆM BILSTM ===")
        
        # Khởi tạo BiLSTM pipeline
        bilstm_pipeline = BiLSTMPipeline(
            max_vocab_size=8000,
            max_seq_length=128,
            embedding_dim=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.3
        )
        
        # Prepare data - convert numpy arrays to lists if needed
        if hasattr(self.X_train, 'tolist'):
            texts = self.X_train.tolist() + self.X_val.tolist() + self.X_test.tolist()
            labels = self.y_train.tolist() + self.y_val.tolist() + self.y_test.tolist()
        else:
            texts = list(self.X_train) + list(self.X_val) + list(self.X_test)
            labels = list(self.y_train) + list(self.y_val) + list(self.y_test)
        
        X_train_bilstm, X_val_bilstm, X_test_bilstm, y_train_bilstm, y_val_bilstm, y_test_bilstm = \
            bilstm_pipeline.prepare_data(texts, labels)
        
        # Create data loaders
        train_loader, val_loader, test_loader = bilstm_pipeline.create_data_loaders(
            X_train_bilstm, X_val_bilstm, X_test_bilstm, 
            y_train_bilstm, y_val_bilstm, y_test_bilstm,
            batch_size=batch_size
        )
        
        # Build and train model
        bilstm_pipeline.build_model(device=self.device)
        
        # Training
        history = bilstm_pipeline.train(train_loader, val_loader, num_epochs, learning_rate)
        
        # Evaluation
        test_loss, test_acc, y_pred, y_true = bilstm_pipeline.evaluate(test_loader)
        
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Convert predictions back to original labels
        y_pred_original = bilstm_pipeline.label_encoder.inverse_transform(y_pred)
        y_true_original = bilstm_pipeline.label_encoder.inverse_transform(y_true)
        
        # Lưu kết quả
        self.results['BiLSTM'] = {
            'model': bilstm_pipeline,
            'y_pred': y_pred_original,
            'y_true': y_true_original,
            'training_history': history
        }
        
        return y_pred_original, y_true_original
    
    def evaluate_all_models(self):
        """Đánh giá tất cả các mô hình"""
        print("\n=== ĐÁNH GIÁ CÁC MÔ HÌNH ===")
        
        # Khởi tạo evaluator
        evaluator = ModelEvaluator(class_names=self.class_names)
        
        # Đánh giá SVM
        if 'SVM' in self.results:
            svm_pred = self.results['SVM']['y_pred']
            evaluator.evaluate_model('SVM', self.y_test, svm_pred)
        
        # Đánh giá PhoBERT
        if 'PhoBERT' in self.results:
            phobert_pred = self.results['PhoBERT']['y_pred']
            phobert_true = self.results['PhoBERT']['y_true']
            evaluator.evaluate_model('PhoBERT', phobert_true, phobert_pred)
        
        # Đánh giá BiLSTM
        if 'BiLSTM' in self.results:
            bilstm_pred = self.results['BiLSTM']['y_pred']
            bilstm_true = self.results['BiLSTM']['y_true']
            evaluator.evaluate_model('BiLSTM', bilstm_true, bilstm_pred)
        
        # So sánh các mô hình
        comparison_df = evaluator.compare_models()
        
        # Tạo báo cáo tổng hợp
        evaluator.generate_summary_report()
        
        # Vẽ biểu đồ so sánh
        evaluator.plot_comparison()
        
        # Confusion matrices
        for model_name in self.results.keys():
            evaluator.plot_confusion_matrix(model_name)
        
        return evaluator, comparison_df
    
    def run_full_experiment(self, sample_size=10000, phobert_epochs=3):
        """Chạy thí nghiệm đầy đủ"""
        print("🚀 BẮT ĐẦU THÍ NGHIỆM PHÂN LOẠI VĂN BẢN PHÁP LÝ")
        print("="*60)
        
        # 1. Chuẩn bị dữ liệu
        self.load_and_prepare_data(sample_size=sample_size)
        
        # 2. Chạy các thí nghiệm
        try:
            # SVM (baseline)
            self.run_svm_experiment()
        except Exception as e:
            print(f"Lỗi khi chạy SVM: {e}")
        
        try:
            # BiLSTM (deep learning baseline)
            self.run_bilstm_experiment(num_epochs=3, batch_size=32)
        except Exception as e:
            print(f"Lỗi khi chạy BiLSTM: {e}")
        
        try:
            # PhoBERT (nếu có GPU/đủ tài nguyên)
            if torch.cuda.is_available() or sample_size <= 5000:
                self.run_phobert_experiment(num_epochs=phobert_epochs)
            else:
                print("Bỏ qua PhoBERT do thiếu GPU và dataset quá lớn")
        except Exception as e:
            print(f"Lỗi khi chạy PhoBERT: {e}")
        
        # 3. Đánh giá và so sánh
        evaluator, results_df = self.evaluate_all_models()
        
        print("\n✅ HOÀN THÀNH THÍ NGHIỆM")
        
        return evaluator, results_df

def main():
    """Hàm chính"""
    # Cấu hình thí nghiệm
    DATA_PATH = "data/sent_truncated_vbpl_legal_only.csv"  # Thay đổi path phù hợp
    TEXT_COLUMN = "text"  # Thay đổi tên cột phù hợp
    LABEL_COLUMN = "label"  # Thay đổi tên cột phù hợp
    SAMPLE_SIZE = 5000  # Giảm để test nhanh
    
    # Kiểm tra file tồn tại
    if not os.path.exists(DATA_PATH):
        print(f"❌ Không tìm thấy file dữ liệu: {DATA_PATH}")
        print("Vui lòng cập nhật đường dẫn file trong biến DATA_PATH")
        return
    
    # Chạy thí nghiệm
    experiment = LegalTextClassificationExperiment(
        data_path=DATA_PATH,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN
    )
    
    evaluator, results_df = experiment.run_full_experiment(
        sample_size=SAMPLE_SIZE,
        phobert_epochs=2  # Giảm epochs để test nhanh
    )
    
    # Lưu kết quả
    results_df.to_csv('classification_results.csv', index=False)
    print(f"\n💾 Đã lưu kết quả vào classification_results.csv")

if __name__ == "__main__":
    main() 