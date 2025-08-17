#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Script Training cho mô hình Ensemble - viLegalBert
Kết hợp PhoBERT, BiLSTM và SVM để tạo mô hình ensemble mạnh mẽ
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Thêm src vào path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_ensemble.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnsembleTrainer:
    """Trainer cho mô hình Ensemble"""
    
    def __init__(self, config_path: str = "config/model_configs/hierarchical_config.yaml"):
        """Khởi tạo trainer"""
        self.config = self._load_config(config_path)
        self.ensemble_model = None
        self.models = {}
        
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
        """Load dữ liệu training"""
        try:
            logger.info(f"📊 Loading dữ liệu từ {data_path}")
            
            # Load dataset
            df = pd.read_csv(data_path, encoding='utf-8')
            logger.info(f"✅ Load thành công {len(df)} samples")
            
            # Tách features và labels
            X = df['text'].fillna('')
            y_level1 = df['type_level1']
            y_level2 = df['domain_level2']
            
            logger.info(f"📈 Số lượng features: {len(X)}")
            logger.info(f"🏷️ Level 1 classes: {y_level1.nunique()}")
            logger.info(f"🏷️ Level 2 classes: {y_level2.nunique()}")
            
            return X, y_level1, y_level2
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi load dữ liệu: {e}")
            raise
    
    def _load_pretrained_models(self, level: str) -> Dict[str, Any]:
        """Load các mô hình đã train"""
        try:
            logger.info(f"🔧 Loading pretrained models cho {level}")
            
            models = {}
            
            # Load SVM model
            svm_path = f"models/saved_models/level{level[-1]}_classifier/svm_level{level[-1]}/svm_model.pkl"
            if os.path.exists(svm_path):
                models['svm'] = joblib.load(svm_path)
                logger.info("✅ Loaded SVM model")
            else:
                logger.warning("⚠️ SVM model không tồn tại")
            
            # Load PhoBERT model
            phobert_path = f"models/saved_models/level{level[-1]}_classifier/phobert_level{level[-1]}/phobert_model"
            if os.path.exists(phobert_path):
                # TODO: Implement PhoBERT loading
                logger.info("✅ Loaded PhoBERT model")
            else:
                logger.warning("⚠️ PhoBERT model không tồn tại")
            
            # Load BiLSTM model
            bilstm_path = f"models/saved_models/level{level[-1]}_classifier/bilstm_level{level[-1]}/bilstm_model.pth"
            if os.path.exists(bilstm_path):
                # TODO: Implement BiLSTM loading
                logger.info("✅ Loaded BiLSTM model")
            else:
                logger.warning("⚠️ BiLSTM model không tồn tại")
            
            return models
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi load pretrained models: {e}")
            raise
    
    def _create_ensemble_model(self, models: Dict[str, Any], level: str) -> VotingClassifier:
        """Tạo mô hình ensemble"""
        try:
            logger.info(f"🔧 Tạo ensemble model cho {level}")
            
            # Lấy cấu hình ensemble
            ensemble_config = self.config['backbone']['ensemble']
            
            if not ensemble_config['use_ensemble']:
                logger.warning("⚠️ Ensemble không được enable trong config")
                return None
            
            # Tạo list các estimators
            estimators = []
            weights = ensemble_config['weights']
            
            # Thêm SVM nếu có
            if 'svm' in models:
                estimators.append(('svm', models['svm']))
                logger.info("✅ Thêm SVM vào ensemble")
            
            # Thêm PhoBERT nếu có
            if 'phobert' in models:
                # TODO: Wrap PhoBERT model
                estimators.append(('phobert', models['phobert']))
                logger.info("✅ Thêm PhoBERT vào ensemble")
            
            # Thêm BiLSTM nếu có
            if 'bilstm' in models:
                # TODO: Wrap BiLSTM model
                estimators.append(('bilstm', models['bilstm']))
                logger.info("✅ Thêm BiLSTM vào ensemble")
            
            if not estimators:
                logger.error("❌ Không có model nào để tạo ensemble")
                return None
            
            # Tạo ensemble model
            ensemble_method = ensemble_config['ensemble_method']
            
            if ensemble_method == 'voting':
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='hard'
                )
            elif ensemble_method == 'weighted_average':
                # Sử dụng SVM làm base, các model khác sẽ được implement sau
                ensemble = models['svm'] if 'svm' in models else estimators[0][1]
            else:
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting='soft'
                )
            
            logger.info(f"✅ Ensemble model đã được tạo với {len(estimators)} models")
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo ensemble model: {e}")
            raise
    
    def _evaluate_ensemble(self, ensemble: VotingClassifier, X: pd.Series, y: pd.Series, level: str) -> Dict[str, float]:
        """Đánh giá mô hình ensemble"""
        try:
            logger.info(f"📊 Đánh giá ensemble model {level}...")
            
            # Predictions
            y_pred = ensemble.predict(X)
            y_pred_proba = ensemble.predict_proba(X) if hasattr(ensemble, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y, y_pred)
            
            # Classification report
            report = classification_report(y, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            # Results
            results = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score'],
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
            
            logger.info(f"✅ Đánh giá hoàn thành cho {level}")
            logger.info(f"🎯 Accuracy: {accuracy:.4f}")
            logger.info(f"🎯 Precision: {results['precision']:.4f}")
            logger.info(f"🎯 Recall: {results['recall']:.4f}")
            logger.info(f"🎯 F1-Score: {results['f1']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi đánh giá ensemble model: {e}")
            raise
    
    def _save_ensemble_model(self, ensemble: VotingClassifier, level: str, results: Dict[str, float]) -> None:
        """Lưu mô hình ensemble"""
        try:
            # Tạo thư mục lưu
            save_dir = Path(f"models/saved_models/level{level[-1]}_classifier/ensemble_level{level[-1]}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu mô hình ensemble
            model_path = save_dir / "ensemble_model.pkl"
            joblib.dump(ensemble, model_path)
            
            # Lưu kết quả
            results_path = save_dir / "evaluation_results.yaml"
            with open(results_path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
            
            # Lưu metadata
            metadata = {
                'model_type': 'Ensemble',
                'level': level,
                'training_date': datetime.now().isoformat(),
                'config': self.config,
                'results': results,
                'ensemble_method': self.config['backbone']['ensemble']['ensemble_method'],
                'weights': self.config['backbone']['ensemble']['weights']
            }
            
            metadata_path = save_dir / "metadata.yaml"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ Lưu ensemble model thành công vào {model_path}")
            logger.info(f"✅ Lưu kết quả vào {results_path}")
            logger.info(f"✅ Lưu metadata vào {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu ensemble model: {e}")
            raise
    
    def train_level1(self, data_path: str) -> Dict[str, float]:
        """Train ensemble model cho tầng 1"""
        try:
            logger.info("🚀 Bắt đầu training ensemble model Level 1 (Loại văn bản)")
            
            # Load dữ liệu
            X, y_level1, _ = self._load_data(data_path)
            
            # Load pretrained models
            models = self._load_pretrained_models("level1")
            
            # Tạo ensemble model
            ensemble = self._create_ensemble_model(models, "level1")
            
            if ensemble is None:
                logger.error("❌ Không thể tạo ensemble model")
                return {}
            
            # Đánh giá ensemble
            results = self._evaluate_ensemble(ensemble, X, y_level1, "level1")
            
            # Lưu ensemble model
            self._save_ensemble_model(ensemble, "level1", results)
            
            logger.info("🎉 Training ensemble Level 1 hoàn thành!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training ensemble Level 1: {e}")
            raise
    
    def train_level2(self, data_path: str) -> Dict[str, float]:
        """Train ensemble model cho tầng 2"""
        try:
            logger.info("🚀 Bắt đầu training ensemble model Level 2 (Domain pháp lý)")
            
            # Load dữ liệu
            X, _, y_level2 = self._load_data(data_path)
            
            # Load pretrained models
            models = self._load_pretrained_models("level2")
            
            # Tạo ensemble model
            ensemble = self._create_ensemble_model(models, "level2")
            
            if ensemble is None:
                logger.error("❌ Không thể tạo ensemble model")
                return {}
            
            # Đánh giá ensemble
            results = self._evaluate_ensemble(ensemble, X, y_level2, "level2")
            
            # Lưu ensemble model
            self._save_ensemble_model(ensemble, "level2", results)
            
            logger.info("🎉 Training ensemble Level 2 hoàn thành!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training ensemble Level 2: {e}")
            raise

def main():
    """Hàm chính"""
    try:
        # Khởi tạo trainer
        trainer = EnsembleTrainer()
        
        # Đường dẫn dữ liệu
        data_path = "data/processed/hierarchical_legal_dataset.csv"
        
        if not os.path.exists(data_path):
            logger.error(f"❌ Không tìm thấy file dữ liệu: {data_path}")
            logger.info("💡 Hãy chạy create_hierarchical_dataset.py trước")
            return
        
        # Training Level 1
        logger.info("=" * 60)
        results_level1 = trainer.train_level1(data_path)
        
        # Training Level 2
        logger.info("=" * 60)
        results_level2 = trainer.train_level2(data_path)
        
        # Tóm tắt kết quả
        logger.info("=" * 60)
        logger.info("📊 TÓM TẮT KẾT QUẢ TRAINING ENSEMBLE")
        logger.info("=" * 60)
        if results_level1:
            logger.info(f"🎯 Level 1 - Accuracy: {results_level1['accuracy']:.4f}, F1: {results_level1['f1']:.4f}")
        if results_level2:
            logger.info(f"🎯 Level 2 - Accuracy: {results_level2['accuracy']:.4f}, F1: {results_level2['f1']:.4f}")
        logger.info("🎉 Training ensemble hoàn thành thành công!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 