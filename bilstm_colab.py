#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏋️ BiLSTM Trainer cho Google Colab (GPU Optimized)
Phân loại văn bản pháp luật Việt Nam với BiLSTM
"""

import os
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sys

# Ignore Jupyter/Colab injected arguments like: -f /path/to/kernel-XXXX.json
if any(arg == '-f' or arg.endswith('.json') for arg in sys.argv[1:]):
    sys.argv = [sys.argv[0]]

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
        "pandas",
        "numpy",
        "scikit-learn"
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
# 🏋️ BILSTM TRAINER
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification with optional class weights."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        gather_logp = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        gather_p = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_factor = (1 - gather_p) ** self.gamma
        loss = -focal_factor * gather_logp
        if self.weight is not None:
            class_w = self.weight[targets]
            loss = loss * class_w
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class DataUtils:
    """Nhóm helper để giảm lặp lại code cho DataLoader, class weights, tokenizer"""
    
    @staticmethod
    def create_tokenizer_and_sequences(texts_train, texts_val, max_features, max_length):
        tokenizer = SimpleTokenizer(max_vocab_size=max_features)
        tokenizer.fit(texts_train)
        train_seq = tokenizer.texts_to_ids(texts_train, max_length)
        val_seq = tokenizer.texts_to_ids(texts_val, max_length)
        return tokenizer, train_seq, val_seq

    @staticmethod
    def create_balanced_train_loader(dataset, labels_train, batch_size, use_gpu):
        from torch.utils.data import DataLoader
        from torch.utils.data.sampler import WeightedRandomSampler
        class_sample_count = np.array([sum(labels_train == t) for t in np.unique(labels_train)])
        class_weights_np = 1.0 / np.clip(class_sample_count, 1, None)
        samples_weight = np.array([class_weights_np[t] for t in labels_train])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          pin_memory=True if use_gpu else False, num_workers=4 if use_gpu else 2)

    @staticmethod
    def create_val_loader(dataset, batch_size, use_gpu):
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          pin_memory=True if use_gpu else False, num_workers=4 if use_gpu else 2)

    @staticmethod
    def compute_class_weights(labels_train, num_classes):
        from collections import Counter
        class_counts = Counter(labels_train)
        weights = torch.ones(num_classes, dtype=torch.float32)
        for cls_idx, cnt in class_counts.items():
            weights[cls_idx] = 1.0 / max(cnt, 1)
        weights = weights * (len(labels_train) / weights.sum())
        return weights

    @staticmethod
    def build_artifacts_dict(model, tokenizer, label_encoder, config, use_gpu):
        return {
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'tokenizer_state': {
                'word_to_idx': tokenizer.word_to_idx,
                'pad_token': tokenizer.pad_token,
                'unk_token': tokenizer.unk_token,
                'start_token': tokenizer.start_token,
                'end_token': tokenizer.end_token,
                'vocab_size': tokenizer.vocab_size,
                'max_vocab_size': tokenizer.max_vocab_size
            },
            'label_encoder': label_encoder,
            'config': config,
            'gpu_optimized': use_gpu
        }

class SimpleTokenizer:
    """Tokenizer đơn giản dùng cho BiLSTM-Embedding"""
    
    def __init__(self, max_vocab_size=20000, pad_token='<PAD>', unk_token='<UNK>', start_token='<START>', end_token='<END>'):
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
    
    def _clean(self, text: str) -> str:
        import re
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _tokenize(self, text: str):
        return text.split()
    
    def fit(self, texts):
        from collections import Counter
        counter = Counter()
        for t in texts:
            toks = self._tokenize(self._clean(t))
            counter.update(toks)
        specials = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        self.word_to_idx = {tok: idx for idx, tok in enumerate(specials)}
        self.idx_to_word = {idx: tok for idx, tok in enumerate(specials)}
        limit = max(0, self.max_vocab_size - len(specials))
        for i, (w, _) in enumerate(counter.most_common(limit)):
            idx = len(specials) + i
            self.word_to_idx[w] = idx
            self.idx_to_word[idx] = w
        self.vocab_size = len(self.word_to_idx)
        return self
    
    def text_to_ids(self, text: str, max_length: int):
        toks = self._tokenize(self._clean(text))
        ids = [self.word_to_idx[self.start_token]]
        for w in toks:
            ids.append(self.word_to_idx.get(w, self.word_to_idx[self.unk_token]))
        ids.append(self.word_to_idx[self.end_token])
        if len(ids) > max_length:
            ids = ids[:max_length]
        if len(ids) < max_length:
            ids.extend([self.word_to_idx[self.pad_token]] * (max_length - len(ids)))
        return ids
    
    def texts_to_ids(self, texts, max_length: int):
        return [self.text_to_ids(t, max_length) for t in texts]

class TokenDataset(Dataset):
    """Dataset cho token id sequences"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class BiLSTMTokenClassifier(nn.Module):
    """BiLSTM với Embedding cho phân loại văn bản"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout=0.5):
        super(BiLSTMTokenClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        w = torch.softmax(self.attention(out), dim=1)
        attended = torch.sum(w * out, dim=1)
        return self.classifier(attended)

class BiLSTMTrainer:
    """Trainer cho mô hình BiLSTM với GPU optimization"""
    
    def __init__(self):
        # Kiểm tra GPU
        self.use_gpu = setup_gpu()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"🚀 Sử dụng device: {self.device}")
        
        # Cấu hình training (token IDs + embedding)
        if self.use_gpu:
            self.config = {
                'max_features': 30000,
                'max_length': 512,
                'embedding_dim': 300,
                'hidden_size': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'num_epochs': 16,
                'batch_size': 64,
                'early_stopping_patience': 6,
                'gradient_clip': 1.0,
                'scheduler_patience': 3,
                'scheduler_factor': 0.5,
                'use_focal_loss': False,
                'weight_decay': 1e-4,
                'use_balanced_sampler': False,
                'use_class_weights': False
            }
        else:
            self.config = {
                'max_features': 20000,
                'max_length': 512,
                'embedding_dim': 200,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'num_epochs': 12,
                'batch_size': 32,
                'early_stopping_patience': 6,
                'gradient_clip': None,
                'scheduler_patience': 3,
                'scheduler_factor': 0.5,
                'use_focal_loss': False,
                'weight_decay': 1e-4,
                'use_balanced_sampler': False,
                'use_class_weights': False
            }
        
        self.vectorizer = None
        self.label_encoder = None
        self.model = None
        
        print(f"🚀 BiLSTMTrainer - GPU: {'✅' if self.use_gpu else '❌'}")
    
    def prepare_data(self, texts, labels):
        """Chuẩn bị data (token IDs) cho training"""
        print("🔄 Chuẩn bị data (token IDs)...")
        self.tokenizer = SimpleTokenizer(max_vocab_size=self.config['max_features'])
        self.tokenizer.fit(texts)
        sequences = self.tokenizer.texts_to_ids(texts, self.config['max_length'])
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        print(f"📊 Vocab size: {self.tokenizer.vocab_size}")
        return sequences, y_encoded, num_classes
    
    def create_model(self, vocab_size, num_classes):
        """Tạo BiLSTM model (Embedding)"""
        print("🏗️ Tạo BiLSTM model...")
        
        self.model = BiLSTMTokenClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'], num_classes=num_classes,
            dropout=self.config['dropout']
        )
        
        # Chuyển model lên device
        self.model.to(self.device)
        
        # GPU optimization
        if self.use_gpu:
            torch.cuda.empty_cache()
            print(f"🚀 Model đã được tối ưu cho GPU (AMP)")
        
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"🚀 Device: {self.device}")
        
        return self.model
    
    def train_model(self, train_loader, val_loader, num_classes, class_weights=None):
        """Training model"""
        print("🏋️ Bắt đầu training...")
        
        # Loss function và optimizer
        if self.config.get('use_focal_loss', False):
            criterion = FocalLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config.get('weight_decay', 0.0))
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.config['scheduler_factor'], 
            patience=self.config['scheduler_patience'], verbose=True
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"🏋️ Bắt đầu training BiLSTM với {self.config['num_epochs']} epochs...")
        print("📊 Progress: Khởi tạo training loop...")
        
        # AMP scaler
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_gpu)

        for epoch in range(self.config['num_epochs']):
            # Progress tracking (training portion at 100%)
            training_progress = ((epoch + 1) / self.config['num_epochs']) * 100
            print(f"⏳ {training_progress:.1f}% - Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.use_gpu):
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                
                if self.use_gpu:
                    scaler.scale(loss).backward()
                    if self.config['gradient_clip']:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
                
                # Mini-batch progress
                if batch_idx % 10 == 0:
                    print(f"    ⏳ Batch {batch_idx}/{len(train_loader)}")
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.squeeze().to(self.device)
                    
                    with torch.cuda.amp.autocast(enabled=self.use_gpu):
                        outputs = self.model(batch_features)
                        loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                base_dir = "/content/viLegalBert"
                torch.save(self.model.state_dict(), f'{base_dir}/best_bilstm_model.pth')
                print(f"    ✅ Best model saved!")
            else:
                patience_counter += 1
            
            # Print epoch results
            print(f"    📊 Epoch {epoch+1} Results:")
            print(f"      Train Loss: {avg_train_loss:.4f}")
            print(f"      Val Loss: {avg_val_loss:.4f}")
            print(f"      Val Accuracy: {val_accuracy:.4f}")
            print(f"      Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            if self.use_gpu:
                print(f"      GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            # Early stopping check
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"    🛑 Early stopping at epoch {epoch+1}")
                break
        
        print("✅ 100% - Training epochs hoàn thành!")
        
        # Load best model
        print("📊 Progress: Load best model...")
        print("⏳ Loading best model...")
        base_dir = "/content/viLegalBert"
        best_model_path = f'{base_dir}/best_bilstm_model.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"✅ 95% - Loaded best model")
        
        print("✅ 100% - BiLSTM training hoàn thành!")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def train_level1(self, data_path, val_path):
        """Training cho Level 1"""
        print("🏷️ Training Level 1...")
        
        # Load training data
        df_train = pd.read_csv(data_path, encoding='utf-8')
        texts_train = df_train['text'].fillna('').tolist()
        
        # Load validation data
        df_val = pd.read_csv(val_path, encoding='utf-8')
        texts_val = df_val['text'].fillna('').tolist()
        
        # Chuẩn bị dữ liệu
        print("📊 Chuẩn bị dữ liệu (token IDs)...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels_train = self.label_encoder.fit_transform(df_train['type_level1'])
        labels_val = self.label_encoder.transform(df_val['type_level1'])
        
        num_classes = len(self.label_encoder.classes_)
        print(f"📊 Số classes: {num_classes}")
        print(f"📊 Train samples: {len(texts_train)}")
        print(f"📊 Validation samples: {len(texts_val)}")
        
        # Tokenizer & sequences
        self.tokenizer, train_seq, val_seq = DataUtils.create_tokenizer_and_sequences(
            texts_train, texts_val, self.config['max_features'], self.config['max_length']
        )
        
        # Datasets
        train_dataset = TokenDataset(train_seq, labels_train)
        val_dataset = TokenDataset(val_seq, labels_val)
        
        # Create model
        self.create_model(self.tokenizer.vocab_size, num_classes)
        
        # Dùng trực tiếp train/val datasets đã có thay vì tự split lại
        
        # Create dataloaders
        if self.config.get('use_balanced_sampler', False):
            train_loader = DataUtils.create_balanced_train_loader(
                train_dataset, labels_train, self.config['batch_size'], self.use_gpu
            )
        else:
            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                pin_memory=True if self.use_gpu else False, num_workers=4 if self.use_gpu else 2
            )
        val_loader = DataUtils.create_val_loader(
            val_dataset, self.config['batch_size'], self.use_gpu
        )
        
        # Class weights to handle imbalance
        weights = DataUtils.compute_class_weights(labels_train, num_classes) if self.config.get('use_class_weights', False) else None

        # Training
        print("📊 Progress: Bắt đầu training pipeline...")
        print("⏳ 0% - Chuẩn bị training...")
        history = self.train_model(train_loader, val_loader, num_classes, class_weights=weights)
        print("⏳ 85% - Training model hoàn thành!")
        
        # Save model
        print("📊 Progress: Lưu model...")
        print("⏳ 90% - Chuẩn bị lưu model...")
        base_dir = "/content/viLegalBert"
        model_path = f"{base_dir}/models/saved_models/level1_classifier/bilstm_level1/bilstm_level1_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        print("⏳ 95% - Lưu model state và config...")
        model_data = DataUtils.build_artifacts_dict(self.model, self.tokenizer, self.label_encoder, self.config, self.use_gpu)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ 100% - BiLSTM Level 1 training hoàn thành!")
        print(f"💾 Model đã lưu: {model_path}")
        return {
            'model_path': model_path,
            'history': history,
            'num_classes': num_classes,
            'gpu_optimized': self.use_gpu
        }
    
    def train_level2(self, data_path, val_path):
        """Training cho Level 2"""
        print("🏷️ Training Level 2...")
        
        # Load training data
        df_train = pd.read_csv(data_path, encoding='utf-8')
        texts_train = df_train['text'].fillna('').tolist()
        
        # Load validation data
        df_val = pd.read_csv(val_path, encoding='utf-8')
        texts_val = df_val['text'].fillna('').tolist()
        
        # Chuẩn bị dữ liệu
        print("📊 Chuẩn bị dữ liệu (token IDs)...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels_train = self.label_encoder.fit_transform(df_train['domain_level2'])
        labels_val = self.label_encoder.transform(df_val['domain_level2'])
        
        num_classes = len(self.label_encoder.classes_)
        print(f"📊 Số classes: {num_classes}")
        print(f"📊 Train samples: {len(texts_train)}")
        print(f"📊 Validation samples: {len(texts_val)}")
        
        # Tokenizer & sequences
        self.tokenizer, train_seq, val_seq = DataUtils.create_tokenizer_and_sequences(
            texts_train, texts_val, self.config['max_features'], self.config['max_length']
        )
        
        # Datasets
        train_dataset = TokenDataset(train_seq, labels_train)
        val_dataset = TokenDataset(val_seq, labels_val)
        
        # Create model
        self.create_model(self.tokenizer.vocab_size, num_classes)
        
        # Dùng trực tiếp train/val datasets đã có thay vì tự split lại
        
        # Create dataloaders
        if self.config.get('use_balanced_sampler', False):
            train_loader = DataUtils.create_balanced_train_loader(
                train_dataset, labels_train, self.config['batch_size'], self.use_gpu
            )
        else:
            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                pin_memory=True if self.use_gpu else False, num_workers=4 if self.use_gpu else 2
            )
        val_loader = DataUtils.create_val_loader(
            val_dataset, self.config['batch_size'], self.use_gpu
        )
        
        # Class weights to handle imbalance
        weights = DataUtils.compute_class_weights(labels_train, num_classes) if self.config.get('use_class_weights', False) else None

        # Training
        print("📊 Progress: Bắt đầu training pipeline Level 2...")
        print("⏳ 0% - Chuẩn bị training...")
        history = self.train_model(train_loader, val_loader, num_classes, class_weights=weights)
        print("⏳ 85% - Training model hoàn thành!")
        
        # Save model
        print("📊 Progress: Lưu model...")
        print("⏳ 90% - Chuẩn bị lưu model...")
        base_dir = "/content/viLegalBert"
        model_path = f"{base_dir}/models/saved_models/level2_classifier/bilstm_level2/bilstm_level2_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        print("⏳ 95% - Lưu model state và config...")
        model_data = DataUtils.build_artifacts_dict(self.model, self.tokenizer, self.label_encoder, self.config, self.use_gpu)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ 100% - BiLSTM Level 2 training hoàn thành!")
        print(f"💾 Model đã lưu: {model_path}")
        return {
            'model_path': model_path,
            'history': history,
            'num_classes': num_classes,
            'gpu_optimized': self.use_gpu
        }

def main():
    """Hàm chính"""
    print("🏋️ BILSTM TRAINER - GPU OPTIMIZED")
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
    os.makedirs(f"{base_dir}/models/saved_models/level1_classifier/bilstm_level1", exist_ok=True)
    os.makedirs(f"{base_dir}/models/saved_models/level2_classifier/bilstm_level2", exist_ok=True)
    
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
    trainer = BiLSTMTrainer()
    
    # Bước 6: Training Level 1
    print("\n🏷️ TRAINING LEVEL 1...")
    train_path = f"{base_dir}/data/processed/dataset_splits/train.csv"
    val_path = f"{base_dir}/data/processed/dataset_splits/validation.csv"
    results_level1 = trainer.train_level1(train_path, val_path)  # Truyền cả train và val
    
    # Bước 7: Training Level 2
    print("\n🏷️ TRAINING LEVEL 2...")
    results_level2 = trainer.train_level2(train_path, val_path)  # Truyền cả train và val
    
    # Tóm tắt kết quả
    print("\n🎉 BILSTM TRAINING HOÀN THÀNH!")
    print("=" * 80)
    print(f"📊 Level 1 model: {results_level1['model_path']}")
    print(f"📊 Level 2 model: {results_level2['model_path']}")
    print(f"🚀 GPU Status: {'✅ Available' if gpu_available else '❌ Not Available'}")

if __name__ == "__main__":
    main() 