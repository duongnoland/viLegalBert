"""
Các mô hình phân loại văn bản pháp lý: SVM, BiLSTM, PhoBERT
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class SVMClassifier:
    """Mô hình SVM với TF-IDF (baseline)"""
    
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words=None,  # Có thể thêm stopwords tiếng Việt
                lowercase=True
            )),
            ('svm', SVC(kernel='linear', random_state=42, probability=True))
        ])
    
    def train(self, X_train, y_train):
        """Huấn luyện mô hình SVM"""
        print("Đang huấn luyện SVM...")
        self.model.fit(X_train, y_train)
        print("Hoàn thành huấn luyện SVM")
    
    def predict(self, X):
        """Dự đoán nhãn"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Dự đoán xác suất"""
        return self.model.predict_proba(X)

class BiLSTMClassifier(nn.Module):
    """Mô hình BiLSTM cho phân loại văn bản"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, 
                 num_layers=2, dropout=0.3, pretrained_embeddings=None):
        super(BiLSTMClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout và classification layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 vì bidirectional
        
    def forward(self, x, lengths=None):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        if lengths is not None:
            # Pack padded sequence để xử lý hiệu quả
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Lấy hidden state cuối cùng từ cả 2 hướng
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        forward_hidden = hidden[-2]  # Hidden state từ forward direction
        backward_hidden = hidden[-1]  # Hidden state từ backward direction
        
        # Concatenate
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Dropout và classification
        output = self.dropout(final_hidden)
        output = self.fc(output)
        
        return output

class PhoBERTClassifier(nn.Module):
    """Mô hình PhoBERT cho phân loại văn bản pháp lý"""
    
    def __init__(self, model_name='vinai/phobert-base', num_classes=3, dropout=0.3):
        super(PhoBERTClassifier, self).__init__()
        
        # Load PhoBERT
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # Freeze một số layer đầu (optional)
        # for param in self.phobert.embeddings.parameters():
        #     param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # PhoBERT forward
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Lấy [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Dropout và classification
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits

class ModelTrainer:
    """Class để huấn luyện các mô hình deep learning"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, optimizer):
        """Huấn luyện một epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Move to device
            if isinstance(batch, dict):
                if 'input_ids' in batch:  # PhoBERT
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                    
                else:  # BiLSTM
                    # Cần implement tokenization cho BiLSTM
                    pass
            
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
        """Đánh giá mô hình"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    if 'input_ids' in batch:  # PhoBERT
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = self.model(input_ids, attention_mask)
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
    
    def train(self, train_loader, val_loader, num_epochs=5, learning_rate=2e-5):
        """Huấn luyện mô hình hoàn chỉnh"""
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        best_val_acc = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        } 