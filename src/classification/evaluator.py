"""
Module đánh giá hiệu suất các mô hình phân loại văn bản pháp lý
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Class đánh giá hiệu suất mô hình"""
    
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.results = {}
    
    def evaluate_model(self, model_name, y_true, y_pred, y_pred_proba=None):
        """
        Đánh giá một mô hình cụ thể
        
        Args:
            model_name: Tên mô hình
            y_true: Nhãn thực tế
            y_pred: Nhãn dự đoán
            y_pred_proba: Xác suất dự đoán (optional)
        """
        # Tính các metrics chính
        accuracy = accuracy_score(y_true, y_pred)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Thêm metrics khác
        weighted_precision = precision_score(y_true, y_pred, average='weighted')
        weighted_recall = recall_score(y_true, y_pred, average='weighted')
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        
        # Lưu kết quả
        self.results[model_name] = {
            'Accuracy': accuracy,
            'Weighted F1-Score': weighted_f1,
            'Macro F1-Score': macro_f1,
            'Weighted Precision': weighted_precision,
            'Weighted Recall': weighted_recall,
            'Macro Precision': macro_precision,
            'Macro Recall': macro_recall,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n=== Kết quả đánh giá {model_name} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1
        }
    
    def print_detailed_report(self, model_name):
        """In báo cáo chi tiết cho một mô hình"""
        if model_name not in self.results:
            print(f"Chưa có kết quả cho mô hình {model_name}")
            return
        
        result = self.results[model_name]
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        print(f"\n=== BÁO CÁO CHI TIẾT - {model_name} ===")
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        ))
    
    def plot_confusion_matrix(self, model_name, figsize=(8, 6)):
        """Vẽ confusion matrix"""
        if model_name not in self.results:
            print(f"Chưa có kết quả cho mô hình {model_name}")
            return
        
        result = self.results[model_name]
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        # Tính confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Vẽ heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """So sánh các mô hình"""
        if not self.results:
            print("Chưa có kết quả nào để so sánh")
            return
        
        # Tạo DataFrame so sánh
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['Accuracy'],
                'Weighted F1-Score': results['Weighted F1-Score'],
                'Macro F1-Score': results['Macro F1-Score'],
                'Weighted Precision': results['Weighted Precision'],
                'Weighted Recall': results['Weighted Recall'],
                'Macro Precision': results['Macro Precision'],
                'Macro Recall': results['Macro Recall']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.round(4)
        
        print("\n=== SO SÁNH CÁC MÔ HÌNH ===")
        print(df.to_string(index=False))
        
        # Tìm mô hình tốt nhất cho từng metric
        print(f"\n=== MÔ HÌNH TỐT NHẤT ===")
        key_metrics = ['Accuracy', 'Weighted F1-Score', 'Macro F1-Score']
        for metric in key_metrics:
            best_model = df.loc[df[metric].idxmax(), 'Model']
            best_score = df[metric].max()
            print(f"{metric}: {best_model} ({best_score:.4f})")
        
        return df
    
    def plot_comparison(self, metrics=['Accuracy', 'Weighted F1-Score', 'Macro F1-Score']):
        """Vẽ biểu đồ so sánh các mô hình"""
        if not self.results:
            print("Chưa có kết quả nào để so sánh")
            return
        
        # Chuẩn bị dữ liệu
        models = list(self.results.keys())
        data = {metric: [self.results[model][metric] for model in models] 
                for metric in metrics}
        
        # Vẽ biểu đồ
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, data[metric], width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('So sánh hiệu suất các mô hình')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Thêm giá trị lên cột
        for i, metric in enumerate(metrics):
            for j, v in enumerate(data[metric]):
                ax.text(j + i*width, v + 0.01, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """Tạo báo cáo tổng hợp"""
        if not self.results:
            print("Chưa có kết quả nào để tạo báo cáo")
            return
        
        print("\n" + "="*60)
        print("           BÁO CÁO TỔNG HỢP KẾT QUẢ PHÂN LOẠI")
        print("="*60)
        
        # So sánh mô hình
        df = self.compare_models()
        
        # Phân tích kết quả
        print(f"\n=== PHÂN TÍCH ===")
        
        # Tìm mô hình tổng quả tốt nhất
        key_metrics = ['Accuracy', 'Weighted F1-Score', 'Macro F1-Score']
        avg_scores = df[key_metrics].mean(axis=1)
        best_overall_idx = avg_scores.idxmax()
        best_overall_model = df.loc[best_overall_idx, 'Model']
        best_overall_score = avg_scores.max()
        
        print(f"Mô hình tổng quát tốt nhất: {best_overall_model} (Điểm TB: {best_overall_score:.4f})")
        
        # So sánh với baseline (SVM)
        if 'SVM' in [model.upper() for model in self.results.keys()]:
            svm_results = None
            for model_name in self.results.keys():
                if 'SVM' in model_name.upper():
                    svm_results = self.results[model_name]
                    break
            
            if svm_results:
                print(f"\n=== SO SÁNH VỚI BASELINE (SVM) ===")
                svm_acc = svm_results['Accuracy']
                
                for model_name, results in self.results.items():
                    if 'SVM' not in model_name.upper():
                        improvement = results['Accuracy'] - svm_acc
                        print(f"{model_name}: {improvement:+.4f} so với SVM")
        
        print("\n" + "="*60)
        
        return df 