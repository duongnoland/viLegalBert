"""
PhoBERT Classifier cho phân loại văn bản pháp lý tiếng Việt
Implementation hoàn chỉnh với text preprocessing và training pipeline
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PhoBERTDataset(Dataset):
    """Dataset cho PhoBERT"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class PhoBERTClassifier(nn.Module):
    """PhoBERT Model cho phân loại văn bản"""
    
    def __init__(self, model_name='vinai/phobert-base', num_classes=2, dropout=0.3):
        super(PhoBERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load PhoBERT
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_classes)
        
        # Freeze một số layers nếu cần
        self._freeze_layers(freeze_embeddings=False)
        
    def _freeze_layers(self, freeze_embeddings=False, freeze_n_layers=0):
        """Freeze một số layers để tránh overfitting"""
        if freeze_embeddings:
            for param in self.phobert.embeddings.parameters():
                param.requires_grad = False
        
        if freeze_n_layers > 0:
            for layer in self.phobert.encoder.layer[:freeze_n_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # PhoBERT forward
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Lấy [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Classification
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits

class PhoBERTTrainer:
    """Trainer cho PhoBERT"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
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
    
    def train(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5, 
              warmup_steps=0, weight_decay=0.01):
        """Training loop hoàn chỉnh"""
        
        # Optimizer và scheduler
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1 if warmup_steps > 0 else 1.0,
            total_iters=warmup_steps if warmup_steps > 0 else 1
        )
        
        best_val_acc = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"🚀 Bắt đầu training PhoBERT...")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.model.model_name}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validation
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
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
                torch.save(self.model.state_dict(), 'best_phobert_model.pth')
                print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print(f"\n🎉 Training hoàn thành!")
        print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        
        return history

class PhoBERTPipeline:
    """Pipeline hoàn chỉnh cho PhoBERT"""
    
    def __init__(self, model_name='vinai/phobert-base', max_length=256, dropout=0.3):
        self.model_name = model_name
        self.max_length = max_length
        self.dropout = dropout
        
        # Load tokenizer
        print(f"📥 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.label_encoder = LabelEncoder()
        self.model = None
        self.trainer = None
        
    def prepare_data(self, texts, labels, test_size=0.2, val_size=0.1):
        """Chuẩn bị dữ liệu"""
        print("📊 Chuẩn bị dữ liệu cho PhoBERT...")
        
        # Convert to lists if needed
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"📝 Classes: {list(self.label_encoder.classes_)}")
        print(f"🎭 Số classes: {self.num_classes}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, encoded_labels, test_size=test_size, 
            random_state=42, stratify=encoded_labels
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=42, stratify=y_temp
        )
        
        print(f"📏 Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test)
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16):
        """Tạo DataLoaders"""
        train_dataset = PhoBERTDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = PhoBERTDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = PhoBERTDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def build_model(self, device='cpu'):
        """Xây dựng model"""
        print(f"🏗️  Đang xây dựng PhoBERT model...")
        
        self.model = PhoBERTClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes,
            dropout=self.dropout
        )
        
        self.trainer = PhoBERTTrainer(self.model, device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Đã xây dựng PhoBERT model:")
        print(f"   - Model: {self.model_name}")
        print(f"   - Num classes: {self.num_classes}")
        print(f"   - Max length: {self.max_length}")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
        """Training"""
        if self.trainer is None:
            raise ValueError("Model chưa được build. Gọi build_model() trước.")
        
        return self.trainer.train(train_loader, val_loader, num_epochs, learning_rate)
    
    def evaluate(self, test_loader):
        """Evaluation"""
        if self.trainer is None:
            raise ValueError("Model chưa được build.")
        
        return self.trainer.evaluate(test_loader)
    
    def predict(self, texts, batch_size=16):
        """Predict on new texts"""
        if self.model is None:
            raise ValueError("Model chưa được train.")
        
        # Create dataset
        dummy_labels = np.zeros(len(texts))
        dataset = PhoBERTDataset(texts, dummy_labels, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Predict
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.trainer.device)
                attention_mask = batch['attention_mask'].to(self.trainer.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels

def test_phobert_pipeline():
    """Test PhoBERT pipeline với dữ liệu mẫu"""
    print("🧪 TESTING PHOBERT PIPELINE")
    print("="*50)
    
    # Sample data
    sample_texts = [
        "Điều này quy định về quyền và nghĩa vụ của công dân",
        "Thủ tục đăng ký kinh doanh phải thực hiện theo quy định",
        "Xử phạt vi phạm hành chính trong lĩnh vực thuế",
        "Định nghĩa về doanh nghiệp theo luật hiện hành"
    ] * 10  # Duplicate để có đủ data
    
    sample_labels = [
        "Quyền nghĩa vụ", "Thủ tục", "Xử phạt", "Định nghĩa"
    ] * 10
    
    # Initialize pipeline
    pipeline = PhoBERTPipeline(max_length=128)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(
        sample_texts, sample_labels
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = pipeline.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=4
    )
    
    # Build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline.build_model(device)
    
    # Training (1 epoch để test)
    history = pipeline.train(train_loader, val_loader, num_epochs=1, learning_rate=5e-5)
    
    # Evaluation
    test_loss, test_acc, predictions, true_labels = pipeline.evaluate(test_loader)
    
    print(f"\n✅ Test completed!")
    print(f"   Test accuracy: {test_acc:.2f}%")
    
    return True

if __name__ == "__main__":
    test_phobert_pipeline() 