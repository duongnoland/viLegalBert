#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Script Training cho mô hình SVM - viLegalBert
Phân loại văn bản pháp luật Việt Nam sử dụng Support Vector Machine
"""

import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any

# Thêm src vào path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_svm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SVMTrainer:
    """Trainer cho mô hình SVM"""
    
    def __init__(self, config_path: str = "config/model_configs/svm_config.yaml"):
        """Khởi tạo trainer"""
        self.config = self._load_config(config_path)
        self.model = None
        self.vectorizer = None
        self.feature_selector = None
        self.pipeline = None
        
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
    
    def _create_pipeline(self, level: str) -> Pipeline:
        """Tạo pipeline cho SVM"""
        try:
            # TF-IDF Vectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.config['feature_extraction']['tfidf']['max_features'],
                min_df=self.config['feature_extraction']['tfidf']['min_df'],
                max_df=self.config['feature_extraction']['tfidf']['max_df'],
                ngram_range=tuple(self.config['feature_extraction']['tfidf']['ngram_range']),
                stop_words=self.config['feature_extraction']['tfidf']['stop_words']
            )
            
            # Feature Selection
            if self.config['feature_extraction']['feature_selection']['enable']:
                feature_selector = SelectKBest(
                    score_func=chi2,
                    k=self.config['feature_extraction']['feature_selection']['k_best']
                )
            else:
                feature_selector = None
            
            # SVM Classifier
            svm_config = self.config['svm']
            svm = SVC(
                kernel=svm_config['kernel'],
                C=svm_config['rbf']['C'] if svm_config['kernel'] == 'rbf' else svm_config['linear']['C'],
                gamma=svm_config['rbf']['gamma'] if svm_config['kernel'] == 'rbf' else 'scale',
                probability=True,
                class_weight='balanced',
                random_state=42,
                max_iter=2000,
                cache_size=200
            )
            
            # Tạo pipeline
            if feature_selector:
                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('feature_selector', feature_selector),
                    ('classifier', svm)
                ])
            else:
                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', svm)
                ])
            
            logger.info(f"✅ Tạo pipeline thành công cho {level}")
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo pipeline: {e}")
            raise
    
    def _hyperparameter_tuning(self, pipeline: Pipeline, X: pd.Series, y: pd.Series) -> Pipeline:
        """Tối ưu hyperparameters"""
        try:
            if not self.config['training']['hyperparameter_tuning']['enable']:
                logger.info("⏭️ Bỏ qua hyperparameter tuning")
                return pipeline
            
            logger.info("🔍 Bắt đầu hyperparameter tuning...")
            
            # Grid search parameters
            param_grid = self.config['training']['hyperparameter_tuning']['grid_search']['param_grid']
            
            # Điều chỉnh param_grid dựa trên pipeline
            if 'feature_selector' in [step[0] for step in pipeline.steps]:
                # Nếu có feature selector, chỉ tune SVM parameters
                svm_params = {f'classifier__{k}': v for k, v in param_grid.items()}
                grid_search = GridSearchCV(
                    pipeline,
                    svm_params,
                    cv=self.config['training']['hyperparameter_tuning']['grid_search']['cv'],
                    scoring=self.config['training']['hyperparameter_tuning']['grid_search']['scoring'],
                    n_jobs=self.config['training']['hyperparameter_tuning']['grid_search']['n_jobs']
                )
            else:
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=self.config['training']['hyperparameter_tuning']['grid_search']['cv'],
                    scoring=self.config['training']['hyperparameter_tuning']['grid_search']['scoring'],
                    n_jobs=self.config['training']['hyperparameter_tuning']['grid_search']['n_jobs']
                )
            
            # Thực hiện grid search
            grid_search.fit(X, y)
            
            logger.info(f"✅ Hyperparameter tuning hoàn thành")
            logger.info(f"🎯 Best parameters: {grid_search.best_params_}")
            logger.info(f"🏆 Best score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tuning hyperparameters: {e}")
            logger.warning("⚠️ Sử dụng pipeline gốc")
            return pipeline
    
    def _cross_validation(self, pipeline: Pipeline, X: pd.Series, y: pd.Series) -> None:
        """Thực hiện cross-validation"""
        try:
            if not self.config['training']['cross_validation']['enable']:
                logger.info("⏭️ Bỏ qua cross-validation")
                return
            
            logger.info("🔄 Thực hiện cross-validation...")
            
            cv_scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=self.config['training']['cross_validation']['cv'],
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            logger.info(f"✅ Cross-validation hoàn thành")
            logger.info(f"📊 CV Scores: {cv_scores}")
            logger.info(f"📈 Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi thực hiện cross-validation: {e}")
    
    def _evaluate_model(self, pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series, level: str) -> Dict[str, float]:
        """Đánh giá mô hình"""
        try:
            logger.info(f"📊 Đánh giá mô hình {level}...")
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Results
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
            
            logger.info(f"✅ Đánh giá hoàn thành cho {level}")
            logger.info(f"🎯 Accuracy: {accuracy:.4f}")
            logger.info(f"🎯 Precision: {precision:.4f}")
            logger.info(f"🎯 Recall: {recall:.4f}")
            logger.info(f"🎯 F1-Score: {f1:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi đánh giá mô hình: {e}")
            raise
    
    def _save_model(self, pipeline: Pipeline, level: str, results: Dict[str, float]) -> None:
        """Lưu mô hình và kết quả"""
        try:
            # Tạo thư mục lưu
            save_dir = Path(f"models/saved_models/level{level[-1]}_classifier/svm_level{level[-1]}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu mô hình
            model_path = save_dir / "svm_model.pkl"
            joblib.dump(pipeline, model_path)
            
            # Lưu kết quả
            results_path = save_dir / "evaluation_results.yaml"
            with open(results_path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
            
            # Lưu metadata
            metadata = {
                'model_type': 'SVM',
                'level': level,
                'training_date': datetime.now().isoformat(),
                'config': self.config,
                'results': results
            }
            
            metadata_path = save_dir / "metadata.yaml"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ Lưu mô hình thành công vào {model_path}")
            logger.info(f"✅ Lưu kết quả vào {results_path}")
            logger.info(f"✅ Lưu metadata vào {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu mô hình: {e}")
            raise
    
    def train_level1(self, data_path: str) -> Dict[str, float]:
        """Train mô hình cho tầng 1"""
        try:
            logger.info("🚀 Bắt đầu training mô hình Level 1 (Loại văn bản)")
            
            # Load dữ liệu
            X, y_level1, _ = self._load_data(data_path)
            
            # Tạo pipeline
            pipeline = self._create_pipeline("level1")
            
            # Hyperparameter tuning
            pipeline = self._hyperparameter_tuning(pipeline, X, y_level1)
            
            # Cross-validation
            self._cross_validation(pipeline, X, y_level1)
            
            # Train mô hình
            logger.info("🏋️ Training mô hình...")
            pipeline.fit(X, y_level1)
            
            # Đánh giá
            results = self._evaluate_model(pipeline, X, y_level1, "Level 1")
            
            # Lưu mô hình
            self._save_model(pipeline, "level1", results)
            
            logger.info("🎉 Training Level 1 hoàn thành!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training Level 1: {e}")
            raise
    
    def train_level2(self, data_path: str) -> Dict[str, float]:
        """Train mô hình cho tầng 2"""
        try:
            logger.info("🚀 Bắt đầu training mô hình Level 2 (Domain pháp lý)")
            
            # Load dữ liệu
            X, _, y_level2 = self._load_data(data_path)
            
            # Tạo pipeline
            pipeline = self._create_pipeline("level2")
            
            # Hyperparameter tuning
            pipeline = self._hyperparameter_tuning(pipeline, X, y_level2)
            
            # Cross-validation
            self._cross_validation(pipeline, X, y_level2)
            
            # Train mô hình
            logger.info("🏋️ Training mô hình...")
            pipeline.fit(X, y_level2)
            
            # Đánh giá
            results = self._evaluate_model(pipeline, X, y_level2, "Level 2")
            
            # Lưu mô hình
            self._save_model(pipeline, "level2", results)
            
            logger.info("🎉 Training Level 2 hoàn thành!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training Level 2: {e}")
            raise

def main():
    """Hàm chính"""
    try:
        # Khởi tạo trainer
        trainer = SVMTrainer()
        
        # Đường dẫn dữ liệu
        data_path = "data/processed/hierarchical_dataset.csv"
        
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
        logger.info("📊 TÓM TẮT KẾT QUẢ TRAINING")
        logger.info("=" * 60)
        logger.info(f"🎯 Level 1 - Accuracy: {results_level1['accuracy']:.4f}, F1: {results_level1['f1']:.4f}")
        logger.info(f"🎯 Level 2 - Accuracy: {results_level2['accuracy']:.4f}, F1: {results_level2['f1']:.4f}")
        logger.info("🎉 Training hoàn thành thành công!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 