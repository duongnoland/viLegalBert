#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏋️ BiLSTM Trainer cho Google Colab
Phân loại văn bản pháp luật Việt Nam với BiLSTM
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
def install_deps():
    try:
        import torch
        print("✅ PyTorch đã sẵn sàng")
    except:
        os.system("pip install torch")
        print("📦 Đã cài đặt PyTorch")
    
    try:
        import torchtext
        print("✅ torchtext đã sẵn sàng")
    except:
        os.system("pip install torchtext")
        print("📦 Đã cài đặt torchtext")

# Import sau khi cài đặt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class TextDataset(Dataset):
    """Dataset cho text classification"""
    
    def __init__(self, texts, labels, vectorizer, max_length=1000):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Vectorize text
        features = self.vectorizer.transform([text]).toarray()[0]
        
        # Truncate/pad to max_length
        if len(features) > self.max_length:
            features = features[:self.max_length]
        else:
            features = np.pad(features, (0, self.max_length - len(features)), 'constant')
        
        return torch.FloatTensor(features), torch.LongTensor([label])

class BiLSTMClassifier(nn.Module):
    """BiLSTM model cho text classification"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        # attended_output shape: (batch_size, hidden_size * 2)
        
        # Classification
        output = self.classifier(attended_output)
        return output

class BiLSTMTrainer:
    """Trainer cho mô hình BiLSTM"""
    
    def __init__(self):
        """Khởi tạo trainer"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Sử dụng device: {self.device}")
        
        # Cấu hình training
        self.config = {
            'max_features': 5000,
            'max_length': 1000,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'num_epochs': 10,
            'batch_size': 32,
            'early_stopping_patience': 5
        }
        
        self.vectorizer = None
        self.label_encoder = None
        self.model = None
    
    def prepare_data(self, texts, labels):
        """Chuẩn bị data cho training"""
        print("🔄 Chuẩn bị data...")
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        # Fit vectorizer
        X_tfidf = self.vectorizer.fit_transform(texts)
        print(f"📊 TF-IDF features: {X_tfidf.shape[1]}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        print(f"📊 Số classes: {num_classes}")
        print(f"📊 Classes: {self.label_encoder.classes_}")
        
        return X_tfidf, y_encoded, num_classes
    
    def create_model(self, input_size, num_classes):
        """Tạo BiLSTM model"""
        print("🏗️ Tạo BiLSTM model...")
        
        self.model = BiLSTMClassifier(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            num_classes=num_classes,
            dropout=self.config['dropout']
        )
        
        self.model.to(self.device)
        
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train_model(self, train_loader, val_loader, num_classes):
        """Training model"""
        print("🏋️ Bắt đầu training...")
        
        # Loss function và optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.squeeze().to(self.device)
                    
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
                # Save best model
                torch.save(self.model.state_dict(), 'best_bilstm_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping check
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_bilstm_model.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def train_level1(self, data_path: str):
        """Training cho Level 1 (Loại văn bản)"""
        print("🏷️ Training Level 1 (Loại văn bản) với BiLSTM...")
        
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        texts = df['text'].fillna('').tolist()
        labels = df['type_level1'].tolist()
        
        # Prepare data
        X_tfidf, y_encoded, num_classes = self.prepare_data(texts, labels)
        
        # Create model
        self.create_model(X_tfidf.shape[1], num_classes)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Create datasets
        train_dataset = TextDataset(
            [texts[i] for i in np.where(X_train.toarray().sum(axis=1) > 0)[0]],
            y_train,
            self.vectorizer,
            self.config['max_length']
        )
        
        val_dataset = TextDataset(
            [texts[i] for i in np.where(X_val.toarray().sum(axis=1) > 0)[0]],
            y_val,
            self.vectorizer,
            self.config['max_length']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Training
        history = self.train_model(train_loader, val_loader, num_classes)
        
        # Save model
        model_path = "models/saved_models/level1_classifier/bilstm_level1/bilstm_level1_model.pkl"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'config': self.config
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model đã được lưu: {model_path}")
        
        return {
            'model_path': model_path,
            'history': history,
            'num_classes': num_classes
        }
    
    def train_level2(self, data_path: str):
        """Training cho Level 2 (Domain pháp lý)"""
        print("🏷️ Training Level 2 (Domain pháp lý) với BiLSTM...")
        
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        texts = df['text'].fillna('').tolist()
        labels = df['domain_level2'].tolist()
        
        # Prepare data
        X_tfidf, y_encoded, num_classes = self.prepare_data(texts, labels)
        
        # Create model
        self.create_model(X_tfidf.shape[1], num_classes)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Create datasets
        train_dataset = TextDataset(
            [texts[i] for i in np.where(X_train.toarray().sum(axis=1) > 0)[0]],
            y_train,
            self.vectorizer,
            self.config['max_length']
        )
        
        val_dataset = TextDataset(
            [texts[i] for i in np.where(X_val.toarray().sum(axis=1) > 0)[0]],
            y_val,
            self.vectorizer,
            self.config['max_length']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Training
        history = self.train_model(train_loader, val_loader, num_classes)
        
        # Save model
        model_path = "models/saved_models/level2_classifier/bilstm_level2/bilstm_level2_model.pkl"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'config': self.config
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model đã được lưu: {model_path}")
        
        return {
            'model_path': model_path,
            'history': history,
            'num_classes': num_classes
        }

def main():
    """Hàm chính"""
    print("🏋️ BILSTM TRAINER CHO GOOGLE COLAB!")
    print("=" * 50)
    
    # Cài đặt dependencies
    install_deps()
    
    # Tạo cấu trúc thư mục
    from pathlib import Path
    Path("models/saved_models/level1_classifier/bilstm_level1").mkdir(parents=True, exist_ok=True)
    Path("models/saved_models/level2_classifier/bilstm_level2").mkdir(parents=True, exist_ok=True)
    
    # Khởi tạo trainer
    trainer = BiLSTMTrainer()
    
    # Training Level 1
    print("\n🏷️ TRAINING LEVEL 1...")
    results_level1 = trainer.train_level1("data/processed/hierarchical_legal_dataset.csv")
    
    # Training Level 2
    print("\n🏷️ TRAINING LEVEL 2...")
    results_level2 = trainer.train_level2("data/processed/hierarchical_legal_dataset.csv")
    
    print("\n🎉 BILSTM TRAINING HOÀN THÀNH!")
    print(f"📊 Level 1 model: {results_level1['model_path']}")
    print(f"📊 Level 2 model: {results_level2['model_path']}")

if __name__ == "__main__":
    main() 