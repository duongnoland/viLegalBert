"""
BiLSTM Classifier cho phân loại văn bản pháp lý tiếng Việt
Implementation đầy đủ với text preprocessing và training pipeline
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re
from collections import Counter
import pickle
from tqdm import tqdm

class TextPreprocessor:
    """Xử lý text cho BiLSTM"""
    
    def __init__(self, max_vocab_size=10000, max_seq_length=256, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.min_freq = min_freq
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def clean_text(self, text):
        """Làm sạch text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Xóa các ký tự đặc biệt, giữ lại chữ, số và khoảng trắng
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Xóa số
        text = re.sub(r'\d+', '', text)
        
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text đơn giản bằng space"""
        return text.split()
    
    def build_vocab(self, texts):
        """Xây dựng vocabulary từ corpus"""
        print("🔤 Đang xây dựng vocabulary...")
        
        # Đếm tần suất từ
        word_counter = Counter()
        
        for text in tqdm(texts, desc="Processing texts"):
            clean_text = self.clean_text(text)
            words = self.tokenize(clean_text)
            word_counter.update(words)
        
        # Lọc từ theo frequency
        filtered_words = {word: count for word, count in word_counter.items() 
                         if count >= self.min_freq}
        
        # Sắp xếp theo frequency
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        # Tạo vocabulary
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]
        
        self.word_to_idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx_to_word = {idx: token for idx, token in enumerate(special_tokens)}
        
        # Thêm top words
        vocab_words = sorted_words[:self.max_vocab_size - len(special_tokens)]
        
        for idx, (word, _) in enumerate(vocab_words):
            word_idx = len(special_tokens) + idx
            self.word_to_idx[word] = word_idx
            self.idx_to_word[word_idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        
        print(f"✅ Đã xây dựng vocabulary:")
        print(f"   - Tổng từ unique: {len(word_counter):,}")
        print(f"   - Từ sau filter (freq >= {self.min_freq}): {len(filtered_words):,}")
        print(f"   - Vocab size cuối: {self.vocab_size:,}")
        
        return self.vocab_size
    
    def text_to_sequence(self, text):
        """Chuyển text thành sequence của indices"""
        clean_text = self.clean_text(text)
        words = self.tokenize(clean_text)
        
        # Chuyển thành indices
        sequence = [self.word_to_idx[self.START_TOKEN]]
        
        for word in words:
            if word in self.word_to_idx:
                sequence.append(self.word_to_idx[word])
            else:
                sequence.append(self.word_to_idx[self.UNK_TOKEN])
        
        sequence.append(self.word_to_idx[self.END_TOKEN])
        
        # Truncate hoặc pad
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            # Pad với PAD_TOKEN
            pad_length = self.max_seq_length - len(sequence)
            sequence.extend([self.word_to_idx[self.PAD_TOKEN]] * pad_length)
        
        return sequence
    
    def texts_to_sequences(self, texts):
        """Chuyển list of texts thành sequences"""
        sequences = []
        for text in tqdm(texts, desc="Converting to sequences"):
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def save(self, filepath):
        """Lưu preprocessor"""
        data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """Load preprocessor"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = data['idx_to_word'] 
        self.vocab_size = data['vocab_size']
        self.max_seq_length = data['max_seq_length']
        self.max_vocab_size = data['max_vocab_size']
        self.min_freq = data['min_freq']

class TextDataset(Dataset):
    """Dataset cho BiLSTM"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }

class BiLSTMClassifier(nn.Module):
    """BiLSTM Model cho phân loại văn bản"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, 
                 num_classes=2, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout và classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 vì bidirectional
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Lấy output từ timestep cuối cùng
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        # Classification
        x = self.dropout(last_output)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class BiLSTMTrainer:
    """Trainer cho BiLSTM"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, optimizer):
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            sequences = batch['sequence'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """Đánh giá model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                sequences = batch['sequence'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)
    
    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        """Training loop hoàn chỉnh"""
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        best_val_acc = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"🚀 Bắt đầu training BiLSTM...")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Validation
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_bilstm_model.pth')
                print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print(f"\n🎉 Training hoàn thành!")
        print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        
        return history

class BiLSTMPipeline:
    """Pipeline hoàn chỉnh cho BiLSTM"""
    
    def __init__(self, max_vocab_size=8000, max_seq_length=128, 
                 embedding_dim=128, hidden_dim=64, num_layers=2, dropout=0.3):
        
        self.preprocessor = TextPreprocessor(
            max_vocab_size=max_vocab_size,
            max_seq_length=max_seq_length
        )
        
        self.label_encoder = LabelEncoder()
        self.model = None
        self.trainer = None
        
        # Model params
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
    def prepare_data(self, texts, labels, test_size=0.2, val_size=0.1):
        """Chuẩn bị dữ liệu"""
        print("📊 Chuẩn bị dữ liệu cho BiLSTM...")
        
        # Build vocabulary
        self.preprocessor.build_vocab(texts)
        
        # Convert texts to sequences
        sequences = self.preprocessor.texts_to_sequences(texts)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"📝 Classes: {list(self.label_encoder.classes_)}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, encoded_labels, test_size=test_size, 
            random_state=42, stratify=encoded_labels
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=42, stratify=y_temp
        )
        
        print(f"📏 Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test)
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """Tạo DataLoaders"""
        train_dataset = TextDataset(X_train, y_train)
        val_dataset = TextDataset(X_val, y_val)
        test_dataset = TextDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def build_model(self, device='cpu'):
        """Xây dựng model"""
        self.model = BiLSTMClassifier(
            vocab_size=self.preprocessor.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        self.trainer = BiLSTMTrainer(self.model, device)
        
        print(f"🏗️  Đã xây dựng BiLSTM model:")
        print(f"   - Vocab size: {self.preprocessor.vocab_size:,}")
        print(f"   - Embedding dim: {self.embedding_dim}")
        print(f"   - Hidden dim: {self.hidden_dim}")
        print(f"   - Num classes: {self.num_classes}")
        print(f"   - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        """Training"""
        if self.trainer is None:
            raise ValueError("Model chưa được build. Gọi build_model() trước.")
        
        return self.trainer.train(train_loader, val_loader, num_epochs, learning_rate)
    
    def evaluate(self, test_loader):
        """Evaluation"""
        if self.trainer is None:
            raise ValueError("Model chưa được build.")
        
        return self.trainer.evaluate(test_loader)
    
    def predict(self, texts):
        """Predict on new texts"""
        if self.model is None:
            raise ValueError("Model chưa được train.")
        
        # Preprocess texts
        sequences = self.preprocessor.texts_to_sequences(texts)
        
        # Create dataset and loader
        dummy_labels = np.zeros(len(texts))  # Dummy labels
        dataset = TextDataset(sequences, dummy_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Predict
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                sequences = batch['sequence'].to(self.trainer.device)
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels 