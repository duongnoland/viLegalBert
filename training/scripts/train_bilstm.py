#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Script Training cho mô hình BiLSTM - viLegalBert
Phân loại văn bản pháp luật Việt Nam sử dụng BiLSTM
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Thêm src vào path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_bilstm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BiLSTMClassifier(nn.Module):
    """BiLSTM Classifier cho phân loại văn bản"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, num_classes: int, dropout: float = 0.2):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x, lengths):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Get last hidden state
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate forward and backward
        
        # Classification
        output = self.dropout(last_hidden)
        output = self.fc(output)
        
        return output

class BiLSTMTrainer:
    """Trainer cho mô hình BiLSTM"""
    
    def __init__(self, config_path: str = "config/model_configs/bilstm_config.yaml"):
        """Khởi tạo trainer"""
        self.config = self._load_config(config_path)
        self.model = None
        self.vectorizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🔧 Sử dụng device: {self.device}")
        
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
    
    def _setup_vectorizer(self, texts: pd.Series) -> None:
        """Thiết lập TF-IDF vectorizer"""
        try:
            logger.info("🔧 Thiết lập TF-IDF vectorizer")
            
            self.vectorizer = TfidfVectorizer(
                max_features=self.config['embedding']['max_features'],
                min_df=self.config['embedding']['min_df'],
                max_df=self.config['embedding']['max_df'],
                ngram_range=tuple(self.config['embedding']['ngram_range'])
            )
            
            # Fit vectorizer
            self.vectorizer.fit(texts)
            
            logger.info(f"✅ Vectorizer đã sẵn sàng với {len(self.vectorizer.vocabulary_)} features")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi thiết lập vectorizer: {e}")
            raise
    
    def _text_to_sequence(self, texts: pd.Series, max_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chuyển đổi text thành sequences"""
        try:
            if max_length is None:
                max_length = self.config['model']['max_length']
            
            logger.info(f"🔧 Chuyển đổi {len(texts)} texts thành sequences")
            
            # Transform texts to TF-IDF features
            features = self.vectorizer.transform(texts)
            
            # Convert to dense array and then to tensor
            features_dense = features.toarray()
            
            # Pad sequences
            padded_features = []
            lengths = []
            
            for feature in features_dense:
                if len(feature) > max_length:
                    padded_features.append(feature[:max_length])
                    lengths.append(max_length)
                else:
                    padded = np.pad(feature, (0, max_length - len(feature)), 'constant')
                    padded_features.append(padded)
                    lengths.append(len(feature))
            
            # Convert to tensors
            features_tensor = torch.tensor(padded_features, dtype=torch.long)
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            
            logger.info(f"✅ Chuyển đổi hoàn thành: {features_tensor.shape}")
            
            return features_tensor, lengths_tensor
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi chuyển đổi text: {e}")
            raise
    
    def _setup_model(self, num_classes: int, level: str) -> None:
        """Thiết lập mô hình BiLSTM"""
        try:
            model_config = self.config['model']
            
            self.model = BiLSTMClassifier(
                vocab_size=self.config['embedding']['max_features'],
                embedding_dim=model_config['embedding_dim'],
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                num_classes=num_classes,
                dropout=model_config['dropout']
            )
            
            # Move to device
            self.model.to(self.device)
            
            logger.info(f"✅ Mô hình BiLSTM đã sẵn sàng cho {level}")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi thiết lập mô hình: {e}")
            raise
    
    def _prepare_data(self, texts: pd.Series, labels: pd.Series, level: str) -> Tuple[DataLoader, DataLoader]:
        """Chuẩn bị dữ liệu cho training"""
        try:
            logger.info(f"🔧 Chuẩn bị dữ liệu cho {level}")
            
            # Convert text to sequences
            features, lengths = self._text_to_sequence(texts)
            
            # Convert labels to tensor
            labels_tensor = torch.tensor(labels.astype('category').cat.codes.values, dtype=torch.long)
            
            # Split train/validation
            X_train, X_val, y_train, y_val, len_train, len_val = train_test_split(
                features, labels_tensor, lengths, test_size=0.2, random_state=42, stratify=labels_tensor
            )
            
            # Create datasets
            train_dataset = TensorDataset(X_train, y_train, len_train)
            val_dataset = TensorDataset(X_val, y_val, len_val)
            
            # Create dataloaders
            batch_size = self.config['training']['batch_size']
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            logger.info(f"✅ Dữ liệu đã sẵn sàng: {len(train_loader)} train batches, {len(val_loader)} val batches")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi chuẩn bị dữ liệu: {e}")
            raise
    
    def _train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                     optimizer: optim.Optimizer) -> float:
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target, lengths) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data, lengths)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate mô hình"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target, lengths in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data, lengths)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def _train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                     num_classes: int, level: str) -> Dict[str, Any]:
        """Train mô hình"""
        try:
            logger.info(f"🏋️ Bắt đầu training mô hình {level}")
            
            # Setup training components
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.config['training']['scheduler_step_size'],
                gamma=self.config['training']['scheduler_gamma']
            )
            
            # Training loop
            best_val_acc = 0
            train_losses = []
            val_losses = []
            val_accuracies = []
            
            for epoch in range(self.config['training']['num_epochs']):
                logger.info(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
                
                # Train
                train_loss = self._train_epoch(train_loader, criterion, optimizer)
                train_losses.append(train_loss)
                
                # Validate
                val_loss, val_acc = self._validate(val_loader, criterion)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                # Update scheduler
                scheduler.step()
                
                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
            
            # Load best model
            self.model.load_state_dict(best_model_state)
            
            # Final evaluation
            final_val_loss, final_val_acc = self._validate(val_loader, criterion)
            
            results = {
                'best_val_accuracy': best_val_acc,
                'final_val_accuracy': final_val_acc,
                'final_val_loss': final_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }
            
            logger.info(f"✅ Training hoàn thành cho {level}")
            logger.info(f"📊 Best validation accuracy: {best_val_acc:.4f}")
            logger.info(f"📊 Final validation accuracy: {final_val_acc:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training mô hình: {e}")
            raise
    
    def _save_model(self, level: str, results: Dict[str, Any]) -> None:
        """Lưu mô hình và kết quả"""
        try:
            # Tạo thư mục lưu
            save_dir = Path(f"models/saved_models/level{level[-1]}_classifier/bilstm_level{level[-1]}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu mô hình
            model_path = save_dir / "bilstm_model.pth"
            torch.save(self.model.state_dict(), model_path)
            
            # Lưu vectorizer
            vectorizer_path = save_dir / "vectorizer.pkl"
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Lưu kết quả
            results_path = save_dir / "evaluation_results.yaml"
            with open(results_path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
            
            # Lưu metadata
            metadata = {
                'model_type': 'BiLSTM',
                'level': level,
                'training_date': datetime.now().isoformat(),
                'config': self.config,
                'results': results,
                'device': str(self.device)
            }
            
            metadata_path = save_dir / "metadata.yaml"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ Lưu mô hình thành công vào {model_path}")
            logger.info(f"✅ Lưu vectorizer vào {vectorizer_path}")
            logger.info(f"✅ Lưu kết quả vào {results_path}")
            logger.info(f"✅ Lưu metadata vào {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu mô hình: {e}")
            raise
    
    def train_level1(self, data_path: str) -> Dict[str, Any]:
        """Train mô hình cho tầng 1"""
        try:
            logger.info("🚀 Bắt đầu training mô hình Level 1 (Loại văn bản)")
            
            # Load dữ liệu
            X, y_level1, _ = self._load_data(data_path)
            
            # Setup vectorizer
            self._setup_vectorizer(X)
            
            # Setup model
            num_classes = y_level1.nunique()
            self._setup_model(num_classes, "level1")
            
            # Prepare data
            train_loader, val_loader = self._prepare_data(X, y_level1, "level1")
            
            # Train model
            results = self._train_model(train_loader, val_loader, num_classes, "level1")
            
            # Save model
            self._save_model("level1", results)
            
            logger.info("🎉 Training Level 1 hoàn thành!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training Level 1: {e}")
            raise
    
    def train_level2(self, data_path: str) -> Dict[str, Any]:
        """Train mô hình cho tầng 2"""
        try:
            logger.info("🚀 Bắt đầu training mô hình Level 2 (Domain pháp lý)")
            
            # Load dữ liệu
            X, _, y_level2 = self._load_data(data_path)
            
            # Setup vectorizer (nếu chưa có)
            if self.vectorizer is None:
                self._setup_vectorizer(X)
            
            # Setup model
            num_classes = y_level2.nunique()
            self._setup_model(num_classes, "level2")
            
            # Prepare data
            train_loader, val_loader = self._prepare_data(X, y_level2, "level2")
            
            # Train model
            results = self._train_model(train_loader, val_loader, num_classes, "level2")
            
            # Save model
            self._save_model("level2", results)
            
            logger.info("🎉 Training Level 2 hoàn thành!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training Level 2: {e}")
            raise

def main():
    """Hàm chính"""
    try:
        # Khởi tạo trainer
        trainer = BiLSTMTrainer()
        
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
        logger.info("📊 TÓM TẮT KẾT QUẢ TRAINING BILSTM")
        logger.info("=" * 60)
        logger.info(f"🎯 Level 1 - Best Val Acc: {results_level1['best_val_accuracy']:.4f}")
        logger.info(f"🎯 Level 2 - Best Val Acc: {results_level2['best_val_accuracy']:.4f}")
        logger.info("🎉 Training BiLSTM hoàn thành thành công!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 