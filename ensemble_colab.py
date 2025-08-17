#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
🏋️ Ensemble Trainer cho Google Colab
Kết hợp SVM, PhoBERT và BiLSTM cho phân loại văn bản pháp luật
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
        "scikit-learn",
        "pandas",
        "numpy",
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
# 🏋️ ENSEMBLE TRAINER
# ============================================================================

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
import numpy as np

class EnsembleTrainer:
    """Trainer cho ensemble model"""
    
    def __init__(self):
        """Khởi tạo trainer"""
        self.models = {}
        self.ensemble_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Sử dụng device: {self.device}")
        
        # Cấu hình ensemble
        self.config = {
            'voting_method': 'soft',  # 'hard' hoặc 'soft'
            'weights': [0.4, 0.3, 0.3]  # SVM, PhoBERT, BiLSTM
        }
    
    def load_svm_models(self):
        """Load SVM models"""
        print("📥 Loading SVM models...")
        
        try:
            # Level 1 SVM
            with open("models/saved_models/level1_classifier/svm_level1/svm_level1_model.pkl", 'rb') as f:
                svm_level1_data = pickle.load(f)
            
            # Level 2 SVM
            with open("models/saved_models/level2_classifier/svm_level2/svm_level2_model.pkl", 'rb') as f:
                svm_level2_data = pickle.load(f)
            
            self.models['svm'] = {
                'level1': svm_level1_data,
                'level2': svm_level2_data
            }
            
            print("✅ SVM models loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi load SVM models: {e}")
            return False
    
    def load_phobert_models(self):
        """Load PhoBERT models"""
        print("📥 Loading PhoBERT models...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Level 1 PhoBERT
            phobert_level1_path = "models/saved_models/level1_classifier/phobert_level1/phobert_level1_model"
            if Path(phobert_level1_path).exists():
                tokenizer_level1 = AutoTokenizer.from_pretrained(phobert_level1_path)
                model_level1 = AutoModelForSequenceClassification.from_pretrained(phobert_level1_path)
                
                with open(f"{phobert_level1_path}/label_encoder.pkl", 'rb') as f:
                    label_encoder_level1 = pickle.load(f)
                
                self.models['phobert'] = {
                    'level1': {
                        'model': model_level1,
                        'tokenizer': tokenizer_level1,
                        'label_encoder': label_encoder_level1
                    }
                }
            
            # Level 2 PhoBERT
            phobert_level2_path = "models/saved_models/level2_classifier/phobert_level2/phobert_level2_model"
            if Path(phobert_level2_path).exists():
                tokenizer_level2 = AutoTokenizer.from_pretrained(phobert_level2_path)
                model_level2 = AutoModelForSequenceClassification.from_pretrained(phobert_level2_path)
                
                with open(f"{phobert_level2_path}/label_encoder.pkl", 'rb') as f:
                    label_encoder_level2 = pickle.load(f)
                
                if 'phobert' not in self.models:
                    self.models['phobert'] = {}
                
                self.models['phobert']['level2'] = {
                    'model': model_level2,
                    'tokenizer': tokenizer_level2,
                    'label_encoder': label_encoder_level2
                }
            
            print("✅ PhoBERT models loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi load PhoBERT models: {e}")
            return False
    
    def load_bilstm_models(self):
        """Load BiLSTM models"""
        print("📥 Loading BiLSTM models...")
        
        try:
            # Level 1 BiLSTM
            with open("models/saved_models/level1_classifier/bilstm_level1/bilstm_level1_model.pkl", 'rb') as f:
                bilstm_level1_data = pickle.load(f)
            
            # Level 2 BiLSTM
            with open("models/saved_models/level2_classifier/bilstm_level2/bilstm_level2_model.pkl", 'rb') as f:
                bilstm_level2_data = pickle.load(f)
            
            self.models['bilstm'] = {
                'level1': bilstm_level1_data,
                'level2': bilstm_level2_data
            }
            
            print("✅ BiLSTM models loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi load BiLSTM models: {e}")
            return False
    
    def create_ensemble_predictor(self, level='level1'):
        """Tạo ensemble predictor cho một level"""
        print(f"🏗️ Tạo ensemble predictor cho {level}...")
        
        estimators = []
        
        # SVM predictor
        if 'svm' in self.models and level in self.models['svm']:
            svm_data = self.models['svm'][level]
            svm_model = svm_data['model']
            svm_vectorizer = svm_data['vectorizer']
            svm_feature_selector = svm_data['feature_selector']
            
            def svm_predict(texts):
                X = svm_vectorizer.transform(texts)
                X_selected = svm_feature_selector.transform(X)
                return svm_model.predict_proba(X_selected)
            
            estimators.append(('svm', svm_predict))
            print("✅ SVM predictor added")
        
        # PhoBERT predictor
        if 'phobert' in self.models and level in self.models['phobert']:
            phobert_data = self.models['phobert'][level]
            phobert_model = phobert_data['model']
            phobert_tokenizer = phobert_data['tokenizer']
            phobert_label_encoder = phobert_data['label_encoder']
            
            def phobert_predict(texts):
                phobert_model.eval()
                predictions = []
                
                for text in texts:
                    inputs = phobert_tokenizer(
                        text, 
                        truncation=True, 
                        padding=True, 
                        max_length=512, 
                        return_tensors="pt"
                    )
                    
                    with torch.no_grad():
                        outputs = phobert_model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                        predictions.append(probs.cpu().numpy()[0])
                
                return np.array(predictions)
            
            estimators.append(('phobert', phobert_predict))
            print("✅ PhoBERT predictor added")
        
        # BiLSTM predictor
        if 'bilstm' in self.models and level in self.models['bilstm']:
            bilstm_data = self.models['bilstm'][level]
            bilstm_model_state = bilstm_data['model_state_dict']
            bilstm_vectorizer = bilstm_data['vectorizer']
            bilstm_label_encoder = bilstm_data['label_encoder']
            bilstm_config = bilstm_data['config']
            
            # Import BiLSTM model class
            from bilstm_colab import BiLSTMClassifier
            
            bilstm_model = BiLSTMClassifier(
                input_size=bilstm_config['max_features'],
                hidden_size=bilstm_config['hidden_size'],
                num_layers=bilstm_config['num_layers'],
                num_classes=len(bilstm_label_encoder.classes_),
                dropout=bilstm_config['dropout']
            )
            
            bilstm_model.load_state_dict(bilstm_model_state)
            bilstm_model.to(self.device)
            bilstm_model.eval()
            
            def bilstm_predict(texts):
                predictions = []
                
                for text in texts:
                    # Vectorize text
                    features = bilstm_vectorizer.transform([text]).toarray()[0]
                    
                    # Truncate/pad
                    if len(features) > bilstm_config['max_length']:
                        features = features[:bilstm_config['max_length']]
                    else:
                        features = np.pad(
                            features, 
                            (0, bilstm_config['max_length'] - len(features)), 
                            'constant'
                        )
                    
                    # Predict
                    with torch.no_grad():
                        inputs = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                        outputs = bilstm_model(inputs)
                        probs = torch.softmax(outputs, dim=1)
                        predictions.append(probs.cpu().numpy()[0])
                
                return np.array(predictions)
            
            estimators.append(('bilstm', bilstm_predict))
            print("✅ BiLSTM predictor added")
        
        if not estimators:
            print("❌ Không có model nào được load")
            return None
        
        print(f"✅ Ensemble predictor created với {len(estimators)} models")
        return estimators
    
    def ensemble_predict(self, texts, level='level1'):
        """Dự đoán với ensemble"""
        print(f"🔮 Ensemble prediction cho {level}...")
        
        estimators = self.create_ensemble_predictor(level)
        if not estimators:
            return None
        
        # Lấy predictions từ từng model
        predictions = {}
        for name, predictor in estimators:
            try:
                pred = predictor(texts)
                predictions[name] = pred
                print(f"✅ {name}: {pred.shape}")
            except Exception as e:
                print(f"❌ Lỗi {name}: {e}")
                continue
        
        if not predictions:
            print("❌ Không có prediction nào thành công")
            return None
        
        # Tính weighted average
        weights = self.config['weights'][:len(predictions)]
        weights = [w/sum(weights) for w in weights]  # Normalize
        
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for i, (name, pred) in enumerate(predictions.items()):
            ensemble_pred += weights[i] * pred
        
        # Lấy class có probability cao nhất
        final_predictions = np.argmax(ensemble_pred, axis=1)
        
        return {
            'predictions': final_predictions,
            'probabilities': ensemble_pred,
            'individual_predictions': predictions
        }
    
    def evaluate_ensemble(self, test_data_path: str):
        """Đánh giá ensemble model"""
        print("📊 Đánh giá ensemble model...")
        
        # Load test data
        df = pd.read_csv(test_data_path, encoding='utf-8')
        texts = df['text'].fillna('').tolist()
        y_true_level1 = df['type_level1'].tolist()
        y_true_level2 = df['domain_level2'].tolist()
        
        # Level 1 prediction
        print("\n🏷️ EVALUATION LEVEL 1...")
        level1_results = self.ensemble_predict(texts, 'level1')
        
        if level1_results:
            y_pred_level1 = level1_results['predictions']
            
            # Decode predictions
            if 'svm' in self.models and 'level1' in self.models['svm']:
                label_encoder = self.models['svm']['level1']['label_encoder']
                y_pred_level1_decoded = label_encoder.inverse_transform(y_pred_level1)
                
                # Calculate accuracy
                accuracy_level1 = accuracy_score(y_true_level1, y_pred_level1_decoded)
                print(f"📊 Level 1 Accuracy: {accuracy_level1:.4f}")
                print(f"📊 Classification Report:")
                print(classification_report(y_true_level1, y_pred_level1_decoded))
        
        # Level 2 prediction
        print("\n🏷️ EVALUATION LEVEL 2...")
        level2_results = self.ensemble_predict(texts, 'level2')
        
        if level2_results:
            y_pred_level2 = level2_results['predictions']
            
            # Decode predictions
            if 'svm' in self.models and 'level2' in self.models['svm']:
                label_encoder = self.models['svm']['level2']['label_encoder']
                y_pred_level2_decoded = label_encoder.inverse_transform(y_pred_level2)
                
                # Calculate accuracy
                accuracy_level2 = accuracy_score(y_true_level2, y_pred_level2_decoded)
                print(f"📊 Level 2 Accuracy: {accuracy_level2:.4f}")
                print(f"📊 Classification Report:")
                print(classification_report(y_true_level2, y_pred_level2_decoded))
        
        # Save ensemble results
        ensemble_path = "models/saved_models/hierarchical_models/ensemble_model.pkl"
        Path(ensemble_path).parent.mkdir(parents=True, exist_ok=True)
        
        ensemble_data = {
            'models': self.models,
            'config': self.config,
            'level1_results': level1_results,
            'level2_results': level2_results
        }
        
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        print(f"💾 Ensemble model đã được lưu: {ensemble_path}")
        
        return {
            'level1_results': level1_results,
            'level2_results': level2_results,
            'ensemble_path': ensemble_path
        }

def main():
    """Hàm chính"""
    print("🏋️ ENSEMBLE TRAINER - GPU OPTIMIZED")
    print("=" * 50)
    
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
    import os
    os.makedirs(f"{base_dir}/models/saved_models/hierarchical_models", exist_ok=True)
    
    # Bước 4: Kiểm tra splits
    print("\n🔄 BƯỚC 4: KIỂM TRA SPLITS")
    splits_dir = f"{base_dir}/data/processed/dataset_splits"
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("❌ Dataset splits chưa có, vui lòng chạy main pipeline trước")
        return
    
    # Load và hiển thị thông tin splits
    import pandas as pd
    train_df = pd.read_csv(train_path, encoding='utf-8')
    val_df = pd.read_csv(val_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')
    
    print(f"✅ Dataset splits đã có sẵn:")
    print(f"📊 Train set: {len(train_df)} samples")
    print(f"📊 Validation set: {len(val_df)} samples")
    print(f"📊 Test set: {len(test_df)} samples")
    
    # Bước 5: Khởi tạo trainer
    print("\n🏋️ BƯỚC 5: KHỞI TẠO TRAINER")
    trainer = EnsembleTrainer()
    
    # Bước 6: Tạo ensemble
    print("\n🔄 TẠO ENSEMBLE...")
    ensemble_results = trainer.create_ensemble()
    
    # Bước 7: Đánh giá ensemble
    print("\n📊 ĐÁNH GIÁ ENSEMBLE...")
    evaluation_results = trainer.evaluate_ensemble(f"{base_dir}/data/processed/dataset_splits/test.csv")
    
    # Tóm tắt kết quả
    print("\n🎉 ENSEMBLE TRAINING HOÀN THÀNH!")
    print("=" * 80)
    print(f"📊 Ensemble model: {ensemble_results['ensemble_path']}")
    print(f"📊 Level 1 Accuracy: {evaluation_results['level1_accuracy']:.4f}")
    print(f"📊 Level 2 Accuracy: {evaluation_results['level2_accuracy']:.4f}")
    print(f"🚀 GPU Status: {'✅ Available' if gpu_available else '❌ Not Available'}")

if __name__ == "__main__":
    main() 