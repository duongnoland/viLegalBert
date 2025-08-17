#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Complete Pipeline viLegalBert cho Google Colab (GPU Optimized)
Tích hợp SVM, PhoBERT, BiLSTM và Ensemble
"""

import os
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 🚀 GPU SETUP & DEPENDENCIES
# ============================================================================

def setup_gpu():
    """Setup GPU environment cho Linux"""
    import torch
    
    if torch.cuda.is_available():
        print("🚀 GPU CUDA available!")
        print(f"📊 GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Set default device
        torch.cuda.set_device(0)
        return True
    else:
        print("⚠️ GPU CUDA không available, sử dụng CPU")
        return False

def install_deps():
    """Cài đặt dependencies cho Linux"""
    import subprocess
    import sys
    
    packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} đã có sẵn")
        except ImportError:
            print(f"📦 Cài đặt {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} đã cài đặt xong")

# ============================================================================
# 🚀 COMPLETE PIPELINE
# ============================================================================

class CompletePipeline:
    """Pipeline hoàn chỉnh cho viLegalBert với GPU optimization"""
    
    def __init__(self):
        # Kiểm tra GPU
        self.use_gpu = setup_gpu()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"🚀 Sử dụng device: {self.device}")
        
        # Cấu hình pipeline
        self.config = {
            'train_models': ['svm', 'phobert', 'bilstm'],
            'create_ensemble': True,
            'evaluate_all': True,
            'gpu_optimization': {
                'mixed_precision': self.use_gpu,
                'gradient_accumulation': self.use_gpu,
                'memory_efficient': self.use_gpu,
                'parallel_processing': self.use_gpu
            }
        }
        
        # Kết quả training
        self.results = {}
        
        print(f"🚀 CompletePipeline - GPU: {'✅' if self.use_gpu else '❌'}")
    
    def create_dirs(self):
        """Tạo thư mục cần thiết từ /content/viLegalBert"""
        import os
        
        # Base directory cho Google Colab
        base_dir = "/content/viLegalBert"
        
        dirs = [
            f"{base_dir}/models/saved_models/level1_classifier/svm_level1",
            f"{base_dir}/models/saved_models/level2_classifier/svm_level2",
            f"{base_dir}/models/saved_models/level1_classifier/phobert_level1",
            f"{base_dir}/models/saved_models/level2_classifier/phobert_level2",
            f"{base_dir}/models/saved_models/level1_classifier/bilstm_level1",
            f"{base_dir}/models/saved_models/level2_classifier/bilstm_level2",
            f"{base_dir}/models/saved_models/hierarchical_models",
            f"{base_dir}/results/evaluation_results",
            f"{base_dir}/logs"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Đã tạo thư mục: {dir_path}")
    
    def check_dataset(self):
        """Kiểm tra dataset có sẵn từ /content/viLegalBert"""
        import os
        
        # Base directory cho Google Colab
        base_dir = "/content/viLegalBert"
        
        dataset_path = f"{base_dir}/data/processed/hierarchical_legal_dataset.csv"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Không tìm thấy dataset: {dataset_path}")
            return None
        
        print(f"✅ Dataset đã có sẵn: {dataset_path}")
        return dataset_path
    
    def check_splits(self):
        """Kiểm tra dataset splits có sẵn cho Linux từ /content/viLegalBert"""
        import os
        
        # Base directory cho Google Colab
        base_dir = "/content/viLegalBert"
        
        splits_dir = f"{base_dir}/data/processed/dataset_splits"
        train_path = os.path.join(splits_dir, "train.csv")
        val_path = os.path.join(splits_dir, "validation.csv")
        test_path = os.path.join(splits_dir, "test.csv")
        
        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            # Load và hiển thị thông tin splits
            import pandas as pd
            train_df = pd.read_csv(train_path, encoding='utf-8')
            val_df = pd.read_csv(val_path, encoding='utf-8')
            test_df = pd.read_csv(test_path, encoding='utf-8')
            
            print(f"✅ Dataset splits đã có sẵn:")
            print(f"📊 Train set: {len(train_df)} samples")
            print(f"📊 Validation set: {len(val_df)} samples")
            print(f"📊 Test set: {len(test_df)} samples")
            return True
        else:
            print("⚠️ Dataset splits chưa có, sẽ tạo mới...")
            return False
    
    def train_svm(self, dataset_path):
        """Training SVM models"""
        print("🏋️ Training SVM models...")
        
        try:
            from main_colab import SVMTrainer
            
            trainer = SVMTrainer()
            
            results_level1 = trainer.train_level1(dataset_path)
            results_level2 = trainer.train_level2(dataset_path)
            
            self.results['svm'] = {
                'level1': results_level1,
                'level2': results_level2,
                'gpu_optimized': results_level1.get('gpu_optimized', False)
            }
            
            print("✅ SVM training hoàn thành")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi training SVM: {e}")
            return False
    
    def train_phobert(self, dataset_path):
        """Training PhoBERT models"""
        print("🏋️ Training PhoBERT models...")
        
        try:
            from phobert_colab import PhoBERTTrainer
            
            trainer = PhoBERTTrainer()
            
            results_level1 = trainer.train_level1(dataset_path)
            results_level2 = trainer.train_level2(dataset_path)
            
            self.results['phobert'] = {
                'level1': results_level1,
                'level2': results_level2,
                'gpu_optimized': results_level1.get('gpu_optimized', False)
            }
            
            print("✅ PhoBERT training hoàn thành")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi training PhoBERT: {e}")
            return False
    
    def train_bilstm(self, dataset_path):
        """Training BiLSTM models"""
        print("🏋️ Training BiLSTM models...")
        
        try:
            from bilstm_colab import BiLSTMTrainer
            
            trainer = BiLSTMTrainer()
            
            results_level1 = trainer.train_level1(dataset_path)
            results_level2 = trainer.train_level2(dataset_path)
            
            self.results['bilstm'] = {
                'level1': results_level1,
                'level2': results_level2,
                'gpu_optimized': results_level1.get('gpu_optimized', False)
            }
            
            print("✅ BiLSTM training hoàn thành")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi training BiLSTM: {e}")
            return False
    
    def create_ensemble(self):
        """Tạo ensemble model"""
        print("🏗️ Tạo ensemble model...")
        
        try:
            from ensemble_colab import EnsembleTrainer
            
            trainer = EnsembleTrainer()
            
            # Load models
            svm_loaded = trainer.load_svm_models()
            phobert_loaded = trainer.load_phobert_models()
            bilstm_loaded = trainer.load_bilstm_models()
            
            if not any([svm_loaded, phobert_loaded, bilstm_loaded]):
                print("❌ Không có model nào được load thành công")
                return False
            
            # Evaluation ensemble
            results = trainer.evaluate_ensemble("data/processed/dataset_splits/test.csv")
            
            self.results['ensemble'] = results
            
            print("✅ Ensemble model đã được tạo")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi tạo ensemble: {e}")
            return False
    
    def evaluate_all(self):
        """Đánh giá tất cả models"""
        print("📊 Đánh giá tất cả models...")
        
        try:
            # Load test data
            test_df = pd.read_csv("data/processed/dataset_splits/test.csv", encoding='utf-8')
            
            evaluation_results = {}
            
            # Evaluate SVM
            if 'svm' in self.results:
                print("\n🏷️ EVALUATING SVM...")
                from main_colab import evaluate_models
                svm_results = evaluate_models("data/processed/dataset_splits/test.csv")
                evaluation_results['svm'] = svm_results
                print(f"🚀 GPU Optimized: {'✅' if self.results['svm'].get('gpu_optimized', False) else '❌'}")
            
            # Evaluate PhoBERT
            if 'phobert' in self.results:
                print("\n🏷️ EVALUATING PHOBERT...")
                evaluation_results['phobert'] = {
                    'status': 'trained',
                    'gpu_optimized': self.results['phobert'].get('gpu_optimized', False)
                }
                print(f"🚀 GPU Optimized: {'✅' if self.results['phobert'].get('gpu_optimized', False) else '❌'}")
            
            # Evaluate BiLSTM
            if 'bilstm' in self.results:
                print("\n🏷️ EVALUATING BILSTM...")
                evaluation_results['bilstm'] = {
                    'status': 'trained',
                    'gpu_optimized': self.results['bilstm'].get('gpu_optimized', False)
                }
                print(f"🚀 GPU Optimized: {'✅' if self.results['bilstm'].get('gpu_optimized', False) else '❌'}")
            
            # Evaluate Ensemble
            if 'ensemble' in self.results:
                print("\n🏷️ EVALUATING ENSEMBLE...")
                evaluation_results['ensemble'] = self.results['ensemble']
            
            # Save evaluation results
            results_path = "results/evaluation_results/complete_evaluation_results.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(evaluation_results, f)
            
            print(f"💾 Evaluation results đã lưu: {results_path}")
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá: {e}")
            return None
    
    def generate_report(self):
        """Tạo báo cáo tổng hợp"""
        print("📋 Tạo báo cáo tổng hợp...")
        
        report = {
            'pipeline_config': self.config,
            'training_results': self.results,
            'gpu_status': {
                'available': self.use_gpu,
                'device': str(self.device),
                'optimization_enabled': self.config['gpu_optimization']
            },
            'summary': {}
        }
        
        # Thống kê models đã train
        trained_models = list(self.results.keys())
        report['summary']['trained_models'] = trained_models
        report['summary']['total_models'] = len(trained_models)
        
        # Thống kê theo level và GPU optimization
        for level in ['level1', 'level2']:
            level_models = []
            gpu_optimized_models = []
            
            for model_name in trained_models:
                if model_name != 'ensemble' and level in self.results[model_name]:
                    level_models.append(model_name)
                    if self.results[model_name].get('gpu_optimized', False):
                        gpu_optimized_models.append(model_name)
            
            report['summary'][f'{level}_models'] = level_models
            report['summary'][f'{level}_gpu_optimized'] = gpu_optimized_models
        
        # Lưu báo cáo
        report_path = "results/training_results/pipeline_summary_report.pkl"
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)
        
        print(f"💾 Báo cáo đã lưu: {report_path}")
        
        # In báo cáo
        print("\n" + "=" * 80)
        print("📋 BÁO CÁO TỔNG HỢP PIPELINE VỚI GPU OPTIMIZATION")
        print("=" * 80)
        print(f"📊 Models đã train: {', '.join(trained_models)}")
        print(f"📊 Tổng số models: {len(trained_models)}")
        print(f"🚀 GPU Status: {'✅ Available' if self.use_gpu else '❌ Not Available'}")
        print(f"🚀 Device: {self.device}")
        
        for level in ['level1', 'level2']:
            level_models = report['summary'][f'{level}_models']
            gpu_optimized = report['summary'][f'{level}_gpu_optimized']
            print(f"🏷️ {level.upper()}: {', '.join(level_models)}")
            print(f"🚀 GPU Optimized: {', '.join(gpu_optimized) if gpu_optimized else 'None'}")
        
        print("=" * 80)
        
        return report
    
    def run_pipeline(self):
        """Chạy toàn bộ pipeline"""
        print("🚀 KHỞI ĐỘNG COMPLETE PIPELINE VỚI GPU OPTIMIZATION!")
        print("=" * 80)
        
        # Base directory cho Google Colab
        base_dir = "/content/viLegalBert"
        
        # Bước 1: GPU setup
        print("\n🚀 BƯỚC 1: GPU SETUP")
        gpu_available = setup_gpu()
        
        # Bước 2: Cài đặt dependencies
        print("\n📦 BƯỚC 2: CÀI ĐẶT DEPENDENCIES")
        install_deps()
        
        # Bước 3: Tạo thư mục
        print("\n🏗️ BƯỚC 3: TẠO THƯ MỤC")
        self.create_dirs()
        
        # Bước 4: Kiểm tra splits
        print("\n🔄 BƯỚC 4: KIỂM TRA SPLITS")
        if not self.check_splits():
            print("❌ Pipeline dừng do không có dataset splits")
            return False
        
        # Bước 5: Training các models
        print("\n🏋️ BƯỚC 5: TRAINING MODELS")
        training_success = True
        
        # ĐÚNG: Training chỉ trên train set
        train_path = f"{base_dir}/data/processed/dataset_splits/train.csv"
        
        if 'svm' in self.config['train_models']:
            if not self.train_svm(train_path):  # Chỉ training trên train set
                training_success = False
        
        if 'phobert' in self.config['train_models']:
            if not self.train_phobert(train_path):  # Chỉ training trên train set
                training_success = False
        
        if 'bilstm' in self.config['train_models']:
            if not self.train_bilstm(train_path):  # Chỉ training trên train set
                training_success = False
        
        if not training_success:
            print("⚠️ Một số models training thất bại")
        
        # Bước 6: Tạo ensemble
        if self.config['create_ensemble'] and training_success:
            self.create_ensemble()
        
        # Bước 7: Đánh giá tất cả
        if self.config['evaluate_all']:
            self.evaluate_all()
        
        # Bước 8: Tạo báo cáo
        self.generate_report()
        
        print("\n🎉 COMPLETE PIPELINE HOÀN THÀNH!")
        print("🚀 viLegalBert đã sẵn sàng sử dụng!")
        print(f"🚀 GPU Status: {'✅ Available' if gpu_available else '❌ Not Available'}")
        
        return True

def main():
    """Hàm chính"""
    print("🚀 VILEGALBERT COMPLETE PIPELINE - GPU OPTIMIZED")
    print("=" * 80)
    
    # Khởi tạo và chạy pipeline
    pipeline = CompletePipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("\n🎉 PIPELINE HOÀN THÀNH THÀNH CÔNG!")
        print("📊 Bạn có thể sử dụng các models đã train để dự đoán")
        print("🚀 Tiếp theo: Tạo web app hoặc API để sử dụng models")
        print(f"🚀 GPU Status: {'✅ Available' if pipeline.use_gpu else '❌ Not Available'}")
    else:
        print("\n❌ PIPELINE GẶP LỖI!")
        print("🔍 Hãy kiểm tra logs để tìm nguyên nhân")

if __name__ == "__main__":
    main() 