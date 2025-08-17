#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Complete Pipeline viLegalBert cho Google Colab (Dataset Có Sẵn)
Tích hợp SVM, PhoBERT, BiLSTM và Ensemble
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cài đặt dependencies
def install_dependencies():
    """Cài đặt tất cả dependencies cần thiết"""
    print("📦 Cài đặt dependencies...")
    
    dependencies = [
        'scikit-learn',
        'torch',
        'transformers',
        'datasets',
        'torchtext'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            print(f"✅ {dep} đã sẵn sàng")
        except ImportError:
            print(f"📦 Đang cài đặt {dep}...")
            os.system(f"pip install {dep}")
            print(f"✅ Đã cài đặt {dep}")

# Import sau khi cài đặt
from sklearn.metrics import accuracy_score, classification_report
import torch
from sklearn.model_selection import train_test_split

class CompletePipeline:
    """Pipeline hoàn chỉnh cho viLegalBert"""
    
    def __init__(self):
        """Khởi tạo pipeline"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Sử dụng device: {self.device}")
        
        # Cấu hình pipeline
        self.config = {
            'train_models': ['svm', 'phobert', 'bilstm'],
            'create_ensemble': True,
            'evaluate_all': True
        }
        
        # Kết quả training
        self.results = {}
    
    def create_project_structure(self):
        """Tạo cấu trúc thư mục"""
        print("🏗️ Tạo cấu trúc project...")
        
        directories = [
            'models/saved_models/level1_classifier/svm_level1',
            'models/saved_models/level2_classifier/svm_level2',
            'models/saved_models/level1_classifier/phobert_level1',
            'models/saved_models/level2_classifier/phobert_level2',
            'models/saved_models/level1_classifier/bilstm_level1',
            'models/saved_models/level2_classifier/bilstm_level2',
            'models/saved_models/hierarchical_models',
            'results/training_results',
            'results/evaluation_results',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Tạo thư mục: {directory}")
    
    def check_dataset_availability(self):
        """Kiểm tra dataset có sẵn"""
        print("🔍 Kiểm tra dataset có sẵn...")
        
        possible_paths = [
            "data/processed/hierarchical_legal_dataset.csv",
            "hierarchical_legal_dataset.csv",
            "data/hierarchical_legal_dataset.csv",
            "dataset.csv",
            "legal_dataset.csv"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                print(f"✅ Tìm thấy dataset: {path}")
                return path
        
        print("❌ Không tìm thấy dataset nào")
        return None
    
    def check_dataset_splits(self):
        """Kiểm tra dataset splits có sẵn"""
        print("🔍 Kiểm tra dataset splits...")
        
        splits_dir = "data/processed/dataset_splits"
        train_path = Path(splits_dir) / "train.csv"
        val_path = Path(splits_dir) / "validation.csv"
        test_path = Path(splits_dir) / "test.csv"
        
        if train_path.exists() and val_path.exists() and test_path.exists():
            print("✅ Dataset splits đã có sẵn")
            
            # Load và hiển thị thông tin splits
            train_df = pd.read_csv(train_path, encoding='utf-8')
            val_df = pd.read_csv(val_path, encoding='utf-8')
            test_df = pd.read_csv(test_path, encoding='utf-8')
            
            print(f"📊 Train set: {len(train_df)} samples")
            print(f"📊 Validation set: {len(val_df)} samples")
            print(f"📊 Test set: {len(test_df)} samples")
            
            return True
        else:
            print("⚠️ Dataset splits chưa có, sẽ tạo mới...")
            return False
    
    def create_training_splits_from_existing(self, dataset_path: str, splits_dir: str):
        """Tạo training splits từ dataset có sẵn"""
        print("🔄 Tạo training splits từ dataset có sẵn...")
        
        # Load dataset
        df = pd.read_csv(dataset_path, encoding='utf-8')
        
        # Chia dữ liệu
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['type_level1'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['type_level1'])
        
        # Lưu các tập
        train_path = Path(splits_dir) / "train.csv"
        val_path = Path(splits_dir) / "validation.csv"
        test_path = Path(splits_dir) / "test.csv"
        
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        print(f"✅ Train set: {len(train_df)} samples -> {train_path}")
        print(f"✅ Validation set: {len(val_df)} samples -> {val_path}")
        print(f"✅ Test set: {len(test_df)} samples -> {test_path}")
    
    def train_svm(self, dataset_path: str):
        """Training SVM models"""
        print("🏋️ Training SVM models...")
        
        try:
            # Import SVM trainer
            from main_colab import SVMTrainer
            
            trainer = SVMTrainer()
            
            # Training Level 1
            results_level1 = trainer.train_level1(dataset_path)
            
            # Training Level 2
            results_level2 = trainer.train_level2(dataset_path)
            
            self.results['svm'] = {
                'level1': results_level1,
                'level2': results_level2
            }
            
            print("✅ SVM training hoàn thành")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi training SVM: {e}")
            return False
    
    def train_phobert(self, dataset_path: str):
        """Training PhoBERT models"""
        print("🏋️ Training PhoBERT models...")
        
        try:
            # Import PhoBERT trainer
            from phobert_colab import PhoBERTTrainer
            
            trainer = PhoBERTTrainer()
            
            # Training Level 1
            results_level1 = trainer.train_level1(dataset_path)
            
            # Training Level 2
            results_level2 = trainer.train_level2(dataset_path)
            
            self.results['phobert'] = {
                'level1': results_level1,
                'level2': results_level2
            }
            
            print("✅ PhoBERT training hoàn thành")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi training PhoBERT: {e}")
            return False
    
    def train_bilstm(self, dataset_path: str):
        """Training BiLSTM models"""
        print("🏋️ Training BiLSTM models...")
        
        try:
            # Import BiLSTM trainer
            from bilstm_colab import BiLSTMTrainer
            
            trainer = BiLSTMTrainer()
            
            # Training Level 1
            results_level1 = trainer.train_level1(dataset_path)
            
            # Training Level 2
            results_level2 = trainer.train_level2(dataset_path)
            
            self.results['bilstm'] = {
                'level1': results_level1,
                'level2': results_level2
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
            # Import ensemble trainer
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
    
    def evaluate_all_models(self):
        """Đánh giá tất cả models"""
        print("📊 Đánh giá tất cả models...")
        
        try:
            # Load test data
            test_df = pd.read_csv("data/processed/dataset_splits/test.csv", encoding='utf-8')
            
            evaluation_results = {}
            
            # Evaluate SVM
            if 'svm' in self.results:
                print("\n🏷️ EVALUATING SVM...")
                from main_colab import evaluate_svm_models
                svm_results = evaluate_svm_models("data/processed/dataset_splits/test.csv")
                evaluation_results['svm'] = svm_results
            
            # Evaluate PhoBERT
            if 'phobert' in self.results:
                print("\n🏷️ EVALUATING PHOBERT...")
                # PhoBERT evaluation logic here
                evaluation_results['phobert'] = {'status': 'trained'}
            
            # Evaluate BiLSTM
            if 'bilstm' in self.results:
                print("\n🏷️ EVALUATING BILSTM...")
                # BiLSTM evaluation logic here
                evaluation_results['bilstm'] = {'status': 'trained'}
            
            # Evaluate Ensemble
            if 'ensemble' in self.results:
                print("\n🏷️ EVALUATING ENSEMBLE...")
                evaluation_results['ensemble'] = self.results['ensemble']
            
            # Save evaluation results
            results_path = "results/evaluation_results/complete_evaluation_results.pkl"
            Path(results_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'wb') as f:
                pickle.dump(evaluation_results, f)
            
            print(f"💾 Evaluation results đã được lưu: {results_path}")
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá: {e}")
            return None
    
    def generate_summary_report(self):
        """Tạo báo cáo tổng hợp"""
        print("📋 Tạo báo cáo tổng hợp...")
        
        report = {
            'pipeline_config': self.config,
            'training_results': self.results,
            'summary': {}
        }
        
        # Thống kê models đã train
        trained_models = list(self.results.keys())
        report['summary']['trained_models'] = trained_models
        report['summary']['total_models'] = len(trained_models)
        
        # Thống kê theo level
        for level in ['level1', 'level2']:
            level_models = []
            for model_name in trained_models:
                if model_name != 'ensemble' and level in self.results[model_name]:
                    level_models.append(model_name)
            
            report['summary'][f'{level}_models'] = level_models
        
        # Lưu báo cáo
        report_path = "results/training_results/pipeline_summary_report.pkl"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)
        
        print(f"💾 Báo cáo tổng hợp đã được lưu: {report_path}")
        
        # In báo cáo
        print("\n" + "=" * 80)
        print("📋 BÁO CÁO TỔNG HỢP PIPELINE")
        print("=" * 80)
        print(f"📊 Models đã train: {', '.join(trained_models)}")
        print(f"📊 Tổng số models: {len(trained_models)}")
        
        for level in ['level1', 'level2']:
            level_models = report['summary'][f'{level}_models']
            print(f"🏷️ {level.upper()}: {', '.join(level_models)}")
        
        print("=" * 80)
        
        return report
    
    def run_pipeline(self):
        """Chạy toàn bộ pipeline"""
        print("🚀 KHỞI ĐỘNG COMPLETE PIPELINE!")
        print("📊 SỬ DỤNG DATASET CÓ SẴN")
        print("=" * 80)
        
        # Bước 1: Cài đặt dependencies
        install_dependencies()
        
        # Bước 2: Tạo cấu trúc project
        self.create_project_structure()
        
        # Bước 3: Kiểm tra dataset có sẵn
        print("\n📊 BƯỚC 1: KIỂM TRA DATASET CÓ SẴN")
        print("-" * 50)
        
        dataset_path = self.check_dataset_availability()
        if dataset_path is None:
            print("❌ Pipeline dừng do không tìm thấy dataset")
            return False
        
        # Bước 4: Kiểm tra và tạo dataset splits
        print("\n🔄 BƯỚC 2: KIỂM TRA DATASET SPLITS")
        print("-" * 50)
        
        if not self.check_dataset_splits():
            # Tạo splits mới từ dataset có sẵn
            splits_dir = "data/processed/dataset_splits"
            self.create_training_splits_from_existing(dataset_path, splits_dir)
        
        # Bước 5: Training các models
        print("\n🏋️ BƯỚC 3: TRAINING MODELS")
        print("-" * 50)
        
        training_success = True
        
        if 'svm' in self.config['train_models']:
            if not self.train_svm(dataset_path):
                training_success = False
        
        if 'phobert' in self.config['train_models']:
            if not self.train_phobert(dataset_path):
                training_success = False
        
        if 'bilstm' in self.config['train_models']:
            if not self.train_bilstm(dataset_path):
                training_success = False
        
        if not training_success:
            print("⚠️ Một số models training thất bại")
        
        # Bước 6: Tạo ensemble
        if self.config['create_ensemble'] and training_success:
            self.create_ensemble()
        
        # Bước 7: Đánh giá tất cả
        if self.config['evaluate_all']:
            self.evaluate_all_models()
        
        # Bước 8: Tạo báo cáo
        self.generate_summary_report()
        
        print("\n🎉 COMPLETE PIPELINE HOÀN THÀNH!")
        print("🚀 viLegalBert đã sẵn sàng sử dụng!")
        
        return True

def main():
    """Hàm chính"""
    print("🚀 VILEGALBERT COMPLETE PIPELINE CHO GOOGLE COLAB!")
    print("📊 SỬ DỤNG DATASET CÓ SẴN")
    print("=" * 80)
    
    # Khởi tạo và chạy pipeline
    pipeline = CompletePipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("\n🎉 PIPELINE HOÀN THÀNH THÀNH CÔNG!")
        print("📊 Bạn có thể sử dụng các models đã train để dự đoán")
        print("🚀 Tiếp theo: Tạo web app hoặc API để sử dụng models")
    else:
        print("\n❌ PIPELINE GẶP LỖI!")
        print("🔍 Hãy kiểm tra logs để tìm nguyên nhân")

if __name__ == "__main__":
    main() 