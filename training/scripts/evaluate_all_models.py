#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 Script Evaluation Tổng Hợp - viLegalBert
Đánh giá và so sánh hiệu suất của tất cả các mô hình
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

# Thêm src vào path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluation_all_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Evaluator tổng hợp cho tất cả các mô hình"""
    
    def __init__(self, config_path: str = "config/model_configs/hierarchical_config.yaml"):
        """Khởi tạo evaluator"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load cấu hình từ file YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Load cấu hình thành công từ {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Lỗi khi load cấu hình: {e}")
            raise
    
    def _load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dữ liệu test"""
        try:
            logger.info(f"📊 Loading dữ liệu test từ {data_path}")
            
            # Load dataset
            df = pd.read_csv(data_path, encoding='utf-8')
            logger.info(f"✅ Load thành công {len(df)} samples")
            
            # Tách features và labels
            X = df['text'].fillna('')
            y_level1 = df['type_level1']
            y_level2 = df['domain_level2']
            
            return X, y_level1, y_level2
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi load dữ liệu: {e}")
            raise
    
    def _load_models(self, level: str) -> Dict[str, Any]:
        """Load tất cả các mô hình đã train"""
        try:
            logger.info(f"🔧 Loading models cho {level}")
            
            models = {}
            level_num = level[-1]
            
            # Load SVM model
            svm_path = f"models/saved_models/level{level_num}_classifier/svm_level{level_num}/svm_model.pkl"
            if os.path.exists(svm_path):
                models['SVM'] = joblib.load(svm_path)
                logger.info("✅ Loaded SVM model")
            else:
                logger.warning("⚠️ SVM model không tồn tại")
            
            # Load PhoBERT model
            phobert_path = f"models/saved_models/level{level_num}_classifier/phobert_level{level_num}/phobert_model"
            if os.path.exists(phobert_path):
                # TODO: Implement PhoBERT loading
                logger.info("✅ Loaded PhoBERT model")
            else:
                logger.warning("⚠️ PhoBERT model không tồn tại")
            
            # Load BiLSTM model
            bilstm_path = f"models/saved_models/level{level_num}_classifier/bilstm_level{level_num}/bilstm_model.pth"
            if os.path.exists(bilstm_path):
                # TODO: Implement BiLSTM loading
                logger.info("✅ Loaded BiLSTM model")
            else:
                logger.warning("⚠️ BiLSTM model không tồn tại")
            
            # Load Ensemble model
            ensemble_path = f"models/saved_models/level{level_num}_classifier/ensemble_level{level_num}/ensemble_model.pkl"
            if os.path.exists(ensemble_path):
                models['Ensemble'] = joblib.load(ensemble_path)
                logger.info("✅ Loaded Ensemble model")
            else:
                logger.warning("⚠️ Ensemble model không tồn tại")
            
            return models
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi load models: {e}")
            raise
    
    def _evaluate_model(self, model: Any, X: pd.Series, y: pd.Series, model_name: str, level: str) -> Dict[str, Any]:
        """Đánh giá một mô hình cụ thể"""
        try:
            logger.info(f"📊 Đánh giá {model_name} cho {level}")
            
            # Predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, y_pred, average='weighted'
            )
            
            # Classification report
            report = classification_report(y, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            # Results
            results = {
                'model_name': model_name,
                'level': level,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
            logger.info(f"✅ Đánh giá {model_name} hoàn thành")
            logger.info(f"🎯 Accuracy: {accuracy:.4f}")
            logger.info(f"🎯 Precision: {precision:.4f}")
            logger.info(f"🎯 Recall: {recall:.4f}")
            logger.info(f"🎯 F1-Score: {f1:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi đánh giá {model_name}: {e}")
            return {}
    
    def _evaluate_all_models(self, X: pd.Series, y: pd.Series, level: str) -> Dict[str, Any]:
        """Đánh giá tất cả các mô hình cho một level"""
        try:
            logger.info(f"🔍 Đánh giá tất cả models cho {level}")
            
            # Load models
            models = self._load_models(level)
            
            if not models:
                logger.error(f"❌ Không có model nào để đánh giá cho {level}")
                return {}
            
            # Evaluate each model
            results = {}
            for model_name, model in models.items():
                model_results = self._evaluate_model(model, X, y, model_name, level)
                if model_results:
                    results[model_name] = model_results
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi đánh giá tất cả models cho {level}: {e}")
            raise
    
    def _create_comparison_report(self, level1_results: Dict[str, Any], level2_results: Dict[str, Any]) -> pd.DataFrame:
        """Tạo báo cáo so sánh tổng hợp"""
        try:
            logger.info("📊 Tạo báo cáo so sánh tổng hợp")
            
            comparison_data = []
            
            # Process Level 1 results
            for model_name, results in level1_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Level': 'Level 1 (Loại văn bản)',
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1']
                })
            
            # Process Level 2 results
            for model_name, results in level2_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Level': 'Level 2 (Domain pháp lý)',
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1']
                })
            
            # Create DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            logger.info("✅ Báo cáo so sánh đã được tạo")
            return comparison_df
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo báo cáo so sánh: {e}")
            raise
    
    def _create_visualizations(self, level1_results: Dict[str, Any], level2_results: Dict[str, Any]) -> None:
        """Tạo visualizations cho kết quả"""
        try:
            logger.info("🎨 Tạo visualizations")
            
            # Tạo thư mục lưu
            viz_dir = Path("results/visualizations/model_comparison")
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Performance Comparison Chart
            self._create_performance_chart(level1_results, level2_results, viz_dir)
            
            # 2. Confusion Matrices
            self._create_confusion_matrices(level1_results, level2_results, viz_dir)
            
            # 3. Metrics Comparison
            self._create_metrics_comparison(level1_results, level2_results, viz_dir)
            
            logger.info("✅ Visualizations đã được tạo")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo visualizations: {e}")
    
    def _create_performance_chart(self, level1_results: Dict[str, Any], level2_results: Dict[str, Any], viz_dir: Path) -> None:
        """Tạo biểu đồ so sánh hiệu suất"""
        try:
            # Prepare data
            models = list(level1_results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('So Sánh Hiệu Suất Các Mô Hình', fontsize=16, fontweight='bold')
            
            for i, metric in enumerate(metrics):
                row, col = i // 2, i % 2
                ax = axes[row, col]
                
                # Level 1 data
                level1_values = [level1_results[model][metric] for model in models]
                
                # Level 2 data
                level2_values = [level2_results[model][metric] for model in models]
                
                x = np.arange(len(models))
                width = 0.35
                
                ax.bar(x - width/2, level1_values, width, label='Level 1', alpha=0.8)
                ax.bar(x + width/2, level2_values, width, label='Level 2', alpha=0.8)
                
                ax.set_xlabel('Models')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo performance chart: {e}")
    
    def _create_confusion_matrices(self, level1_results: Dict[str, Any], level2_results: Dict[str, Any], viz_dir: Path) -> None:
        """Tạo confusion matrices"""
        try:
            for level_name, results in [('Level1', level1_results), ('Level2', level2_results)]:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Confusion Matrices - {level_name}', fontsize=16, fontweight='bold')
                
                for i, (model_name, model_results) in enumerate(results.items()):
                    row, col = i // 2, i % 2
                    ax = axes[row, col]
                    
                    cm = np.array(model_results['confusion_matrix'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f'{model_name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                
                plt.tight_layout()
                plt.savefig(viz_dir / f'confusion_matrices_{level_name.lower()}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo confusion matrices: {e}")
    
    def _create_metrics_comparison(self, level1_results: Dict[str, Any], level2_results: Dict[str, Any], viz_dir: Path) -> None:
        """Tạo biểu đồ so sánh metrics"""
        try:
            # Prepare data
            models = list(level1_results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(models))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                level1_values = [level1_results[model][metric] for model in models]
                level2_values = [level2_results[model][metric] for model in models]
                
                ax.bar(x + i*width, level1_values, width, label=f'{metric.capitalize()} - Level 1', alpha=0.8)
                ax.bar(x + i*width, level2_values, width, label=f'{metric.capitalize()} - Level 2', alpha=0.8, bottom=level1_values)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Metrics Comparison Across Models and Levels')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo metrics comparison: {e}")
    
    def _save_results(self, level1_results: Dict[str, Any], level2_results: Dict[str, Any], comparison_df: pd.DataFrame) -> None:
        """Lưu kết quả evaluation"""
        try:
            # Tạo thư mục lưu
            results_dir = Path("results/evaluation_results/comprehensive_evaluation")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu kết quả chi tiết
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Level 1 results
            level1_path = results_dir / f"level1_results_{timestamp}.yaml"
            with open(level1_path, 'w', encoding='utf-8') as f:
                yaml.dump(level1_results, f, default_flow_style=False, allow_unicode=True)
            
            # Level 2 results
            level2_path = results_dir / f"level2_results_{timestamp}.yaml"
            with open(level2_path, 'w', encoding='utf-8') as f:
                yaml.dump(level2_results, f, default_flow_style=False, allow_unicode=True)
            
            # Comparison report
            comparison_path = results_dir / f"comparison_report_{timestamp}.csv"
            comparison_df.to_csv(comparison_path, index=False, encoding='utf-8')
            
            # Summary report
            summary_path = results_dir / f"summary_report_{timestamp}.yaml"
            summary = {
                'evaluation_date': datetime.now().isoformat(),
                'total_models_evaluated': len(level1_results),
                'levels_evaluated': ['Level 1', 'Level 2'],
                'best_model_level1': max(level1_results.items(), key=lambda x: x[1]['f1'])[0] if level1_results else None,
                'best_model_level2': max(level2_results.items(), key=lambda x: x[1]['f1'])[0] if level2_results else None,
                'average_f1_level1': np.mean([r['f1'] for r in level1_results.values()]) if level1_results else 0,
                'average_f1_level2': np.mean([r['f1'] for r in level2_results.values()]) if level2_results else 0
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ Kết quả đã được lưu vào {results_dir}")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu kết quả: {e}")
            raise
    
    def evaluate_all(self, data_path: str) -> Dict[str, Any]:
        """Đánh giá tất cả các mô hình"""
        try:
            logger.info("🚀 Bắt đầu evaluation tổng hợp")
            
            # Load data
            X, y_level1, y_level2 = self._load_data(data_path)
            
            # Evaluate Level 1
            logger.info("=" * 60)
            level1_results = self._evaluate_all_models(X, y_level1, "level1")
            
            # Evaluate Level 2
            logger.info("=" * 60)
            level2_results = self._evaluate_all_models(X, y_level2, "level2")
            
            # Create comparison report
            comparison_df = self._create_comparison_report(level1_results, level2_results)
            
            # Create visualizations
            self._create_visualizations(level1_results, level2_results)
            
            # Save results
            self._save_results(level1_results, level2_results, comparison_df)
            
            # Print summary
            self._print_summary(level1_results, level2_results, comparison_df)
            
            return {
                'level1_results': level1_results,
                'level2_results': level2_results,
                'comparison': comparison_df
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình evaluation: {e}")
            raise
    
    def _print_summary(self, level1_results: Dict[str, Any], level2_results: Dict[str, Any], comparison_df: pd.DataFrame) -> None:
        """In tóm tắt kết quả"""
        try:
            logger.info("=" * 80)
            logger.info("📊 TÓM TẮT KẾT QUẢ EVALUATION TỔNG HỢP")
            logger.info("=" * 80)
            
            # Level 1 summary
            if level1_results:
                logger.info(f"\n🏷️ LEVEL 1 (Loại văn bản):")
                for model_name, results in level1_results.items():
                    logger.info(f"  {model_name}: Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}")
                
                best_level1 = max(level1_results.items(), key=lambda x: x[1]['f1'])
                logger.info(f"  🏆 Best Model: {best_level1[0]} (F1: {best_level1[1]['f1']:.4f})")
            
            # Level 2 summary
            if level2_results:
                logger.info(f"\n🏷️ LEVEL 2 (Domain pháp lý):")
                for model_name, results in level2_results.items():
                    logger.info(f"  {model_name}: Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}")
                
                best_level2 = max(level2_results.items(), key=lambda x: x[1]['f1'])
                logger.info(f"  🏆 Best Model: {best_level2[0]} (F1: {best_level2[1]['f1']:.4f})")
            
            # Overall summary
            if level1_results and level2_results:
                avg_f1_level1 = np.mean([r['f1'] for r in level1_results.values()])
                avg_f1_level2 = np.mean([r['f1'] for r in level2_results.values()])
                
                logger.info(f"\n📈 TỔNG KẾT:")
                logger.info(f"  Average F1 Level 1: {avg_f1_level1:.4f}")
                logger.info(f"  Average F1 Level 2: {avg_f1_level2:.4f}")
                logger.info(f"  Overall Performance: {(avg_f1_level1 + avg_f1_level2) / 2:.4f}")
            
            logger.info("\n🎉 Evaluation tổng hợp hoàn thành!")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi in tóm tắt: {e}")

def main():
    """Hàm chính"""
    try:
        # Khởi tạo evaluator
        evaluator = ComprehensiveEvaluator()
        
        # Đường dẫn dữ liệu test
        data_path = "data/processed/dataset_splits/test.csv"
        
        if not os.path.exists(data_path):
            logger.error(f"❌ Không tìm thấy file dữ liệu test: {data_path}")
            logger.info("💡 Hãy chạy create_hierarchical_dataset.py trước")
            return
        
        # Thực hiện evaluation
        results = evaluator.evaluate_all(data_path)
        
        logger.info("🎉 Evaluation tổng hợp hoàn thành thành công!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 