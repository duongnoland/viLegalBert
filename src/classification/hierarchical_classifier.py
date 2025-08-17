"""
Hierarchical Legal Text Classification System
Hệ thống phân loại văn bản pháp lý theo tầng
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import json
import pickle
from pathlib import Path

class HierarchicalLabelEncoder:
    """Encoder cho nhãn hierarchical"""
    
    def __init__(self):
        self.coarse_encoder = LabelEncoder()
        self.fine_encoders = {}  # Mỗi coarse label có 1 fine encoder riêng
        self.label_hierarchy = {}  # Mapping coarse -> list of fine labels
        
    def create_hierarchy_from_data(self, texts: List[str], labels: List[str]):
        """Tạo hierarchy từ dữ liệu có sẵn"""
        print("🏗️ Tạo label hierarchy...")
        
        # Parse labels thành coarse và fine
        coarse_labels = []
        fine_labels = []
        
        for text, label in zip(texts, labels):
            coarse, fine = self._extract_coarse_fine_labels(text, label)
            coarse_labels.append(coarse)
            fine_labels.append(fine)
        
        # Build hierarchy
        hierarchy = {}
        for coarse, fine in zip(coarse_labels, fine_labels):
            if coarse not in hierarchy:
                hierarchy[coarse] = set()
            hierarchy[coarse].add(fine)
        
        # Convert sets to lists
        for coarse in hierarchy:
            hierarchy[coarse] = list(hierarchy[coarse])
        
        self.label_hierarchy = hierarchy
        
        # Fit encoders
        self.coarse_encoder.fit(coarse_labels)
        
        for coarse_label in self.coarse_encoder.classes_:
            fine_labels_for_coarse = [fine for c, fine in zip(coarse_labels, fine_labels) 
                                    if c == coarse_label]
            if fine_labels_for_coarse:
                self.fine_encoders[coarse_label] = LabelEncoder()
                self.fine_encoders[coarse_label].fit(fine_labels_for_coarse)
        
        print(f"📊 Hierarchy created:")
        for coarse, fines in hierarchy.items():
            print(f"   {coarse}: {len(fines)} fine classes")
        
        return coarse_labels, fine_labels
    
    def _extract_coarse_fine_labels(self, text: str, original_label: str) -> Tuple[str, str]:
        """Extract coarse và fine label từ text và label gốc"""
        
        # Mapping các keyword để tạo coarse labels
        coarse_mapping = {
            'Luật': ['luật', 'law', 'bộ luật'],
            'Nghị định': ['nghị định', 'decree', 'quyết định'],
            'Thông tư': ['thông tư', 'circular', 'hướng dẫn'],
            'Quyết định': ['quyết định', 'decision'],
            'Hợp đồng': ['hợp đồng', 'contract', 'thỏa thuận'],
            'Bản án': ['bản án', 'judgment', 'phán quyết'],
            'Báo cáo': ['báo cáo', 'report'],
            'Văn bản khác': ['khác', 'other']
        }
        
        # Tìm coarse label
        text_lower = text.lower()
        label_lower = original_label.lower()
        
        coarse_label = 'Văn bản khác'  # Default
        for coarse, keywords in coarse_mapping.items():
            if any(keyword in text_lower or keyword in label_lower for keyword in keywords):
                coarse_label = coarse
                break
        
        # Fine label dựa trên nội dung cụ thể
        fine_label = self._extract_fine_label(text, original_label, coarse_label)
        
        return coarse_label, fine_label
    
    def _extract_fine_label(self, text: str, original_label: str, coarse_label: str) -> str:
        """Extract fine-grained label"""
        
        fine_mappings = {
            'Luật': {
                'Dân sự': ['dân sự', 'civil', 'hôn nhân', 'gia đình'],
                'Hình sự': ['hình sự', 'criminal', 'tội phạm'],
                'Thương mại': ['thương mại', 'commercial', 'kinh doanh'],
                'Lao động': ['lao động', 'labor', 'việc làm'],
                'Khác': ['khác']
            },
            'Nghị định': {
                'Hướng dẫn thi hành': ['hướng dẫn', 'thi hành', 'implementation'],
                'Quy định chi tiết': ['quy định', 'chi tiết', 'detailed'],
                'Khác': ['khác']
            },
            'Thông tư': {
                'Hướng dẫn': ['hướng dẫn', 'guidance'],
                'Quy trình': ['quy trình', 'process', 'thủ tục'],
                'Khác': ['khác']
            },
            'Hợp đồng': {
                'Mua bán': ['mua bán', 'purchase', 'sale'],
                'Dịch vụ': ['dịch vụ', 'service'],
                'NDA': ['bảo mật', 'confidential', 'nda'],
                'Khác': ['khác']
            },
            'Bản án': {
                'Dân sự': ['dân sự', 'civil'],
                'Hình sự': ['hình sự', 'criminal'],
                'Hành chính': ['hành chính', 'administrative'],
                'Khác': ['khác']
            }
        }
        
        text_lower = text.lower()
        label_lower = original_label.lower()
        
        if coarse_label in fine_mappings:
            for fine, keywords in fine_mappings[coarse_label].items():
                if any(keyword in text_lower or keyword in label_lower for keyword in keywords):
                    return fine
        
        return 'Khác'  # Default fine label
    
    def encode_hierarchical(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode labels thành hierarchical format"""
        coarse_labels, fine_labels = self.create_hierarchy_from_data(texts, labels)
        
        coarse_encoded = self.coarse_encoder.transform(coarse_labels)
        
        # Encode fine labels
        fine_encoded = []
        for coarse_label, fine_label in zip(coarse_labels, fine_labels):
            if coarse_label in self.fine_encoders:
                fine_enc = self.fine_encoders[coarse_label].transform([fine_label])[0]
            else:
                fine_enc = 0  # Default
            fine_encoded.append(fine_enc)
        
        return coarse_encoded, np.array(fine_encoded)
    
    def decode_hierarchical(self, coarse_encoded: np.ndarray, fine_encoded: np.ndarray) -> List[Tuple[str, str]]:
        """Decode hierarchical predictions"""
        results = []
        
        for coarse_enc, fine_enc in zip(coarse_encoded, fine_encoded):
            coarse_label = self.coarse_encoder.inverse_transform([coarse_enc])[0]
            
            if coarse_label in self.fine_encoders:
                try:
                    fine_label = self.fine_encoders[coarse_label].inverse_transform([fine_enc])[0]
                except:
                    fine_label = 'Khác'
            else:
                fine_label = 'Khác'
            
            results.append((coarse_label, fine_label))
        
        return results
    
    def save(self, filepath: str):
        """Save encoder"""
        data = {
            'coarse_encoder': self.coarse_encoder,
            'fine_encoders': self.fine_encoders,
            'label_hierarchy': self.label_hierarchy
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load encoder"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.coarse_encoder = data['coarse_encoder']
        self.fine_encoders = data['fine_encoders']
        self.label_hierarchy = data['label_hierarchy']

class ChunkProcessor:
    """Xử lý văn bản thành chunks để phân loại document-level"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_into_chunks(self, text: str) -> List[str]:
        """Chia văn bản thành chunks với overlap"""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end == len(words):
                break
            
            start = end - self.overlap
        
        return chunks if chunks else [text]
    
    def aggregate_chunk_predictions(self, chunk_logits: List[torch.Tensor], strategy: str = 'mean') -> torch.Tensor:
        """Aggregate predictions từ chunks"""
        if not chunk_logits:
            return torch.zeros(1)
        
        stacked = torch.stack(chunk_logits)
        
        if strategy == 'mean':
            return torch.mean(stacked, dim=0)
        elif strategy == 'max':
            return torch.max(stacked, dim=0)[0]
        elif strategy == 'attention':
            # Simple attention mechanism
            attention_weights = torch.softmax(torch.sum(stacked, dim=-1), dim=0)
            return torch.sum(stacked * attention_weights.unsqueeze(-1), dim=0)
        else:
            return torch.mean(stacked, dim=0)

class HierarchicalDataset(Dataset):
    """Dataset cho hierarchical classification"""
    
    def __init__(self, texts: List[str], coarse_labels: np.ndarray, fine_labels: np.ndarray, 
                 tokenizer, max_length: int = 512, use_chunks: bool = True):
        self.texts = texts
        self.coarse_labels = coarse_labels
        self.fine_labels = fine_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chunks = use_chunks
        
        if use_chunks:
            self.chunk_processor = ChunkProcessor(chunk_size=max_length//2)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        coarse_label = self.coarse_labels[idx]
        fine_label = self.fine_labels[idx]
        
        if self.use_chunks:
            chunks = self.chunk_processor.split_into_chunks(text)
            # Lấy chunk đầu tiên cho đơn giản (có thể mở rộng sau)
            text = chunks[0] if chunks else text
        
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
            'coarse_label': torch.tensor(coarse_label, dtype=torch.long),
            'fine_label': torch.tensor(fine_label, dtype=torch.long)
        }

class HierarchicalPhoBERT(nn.Module):
    """PhoBERT model cho hierarchical classification"""
    
    def __init__(self, model_name: str = 'vinai/phobert-base', 
                 num_coarse_classes: int = 5, num_fine_classes: int = 10, 
                 dropout: float = 0.3):
        super(HierarchicalPhoBERT, self).__init__()
        
        self.model_name = model_name
        self.num_coarse_classes = num_coarse_classes
        self.num_fine_classes = num_fine_classes
        
        # Load PhoBERT
        self.phobert = AutoModel.from_pretrained(model_name)
        hidden_size = self.phobert.config.hidden_size
        
        # Shared layers
        self.dropout = nn.Dropout(dropout)
        self.shared_layer = nn.Linear(hidden_size, hidden_size // 2)
        
        # Coarse classifier
        self.coarse_classifier = nn.Linear(hidden_size // 2, num_coarse_classes)
        
        # Fine classifier (shared for all coarse classes for now)
        self.fine_classifier = nn.Linear(hidden_size // 2, num_fine_classes)
        
    def forward(self, input_ids, attention_mask):
        # PhoBERT encoding
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Shared representation
        shared_repr = self.dropout(pooled_output)
        shared_repr = torch.relu(self.shared_layer(shared_repr))
        shared_repr = self.dropout(shared_repr)
        
        # Predictions
        coarse_logits = self.coarse_classifier(shared_repr)
        fine_logits = self.fine_classifier(shared_repr)
        
        return coarse_logits, fine_logits

class HierarchicalTrainer:
    """Trainer cho hierarchical model"""
    
    def __init__(self, model, device='cpu', coarse_weight=1.0, fine_weight=1.0):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.coarse_criterion = nn.CrossEntropyLoss()
        self.fine_criterion = nn.CrossEntropyLoss()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
    
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """Train một epoch"""
        self.model.train()
        total_loss = 0
        coarse_correct = 0
        fine_correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            coarse_labels = batch['coarse_label'].to(self.device)
            fine_labels = batch['fine_label'].to(self.device)
            
            optimizer.zero_grad()
            
            coarse_logits, fine_logits = self.model(input_ids, attention_mask)
            
            # Multi-task loss
            coarse_loss = self.coarse_criterion(coarse_logits, coarse_labels)
            fine_loss = self.fine_criterion(fine_logits, fine_labels)
            total_loss_batch = self.coarse_weight * coarse_loss + self.fine_weight * fine_loss
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += total_loss_batch.item()
            
            # Accuracy
            _, coarse_pred = torch.max(coarse_logits.data, 1)
            _, fine_pred = torch.max(fine_logits.data, 1)
            
            total += coarse_labels.size(0)
            coarse_correct += (coarse_pred == coarse_labels).sum().item()
            fine_correct += (fine_pred == fine_labels).sum().item()
        
        coarse_acc = 100 * coarse_correct / total
        fine_acc = 100 * fine_correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, coarse_acc, fine_acc
    
    def evaluate(self, test_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        coarse_correct = 0
        fine_correct = 0
        total = 0
        
        all_coarse_pred = []
        all_fine_pred = []
        all_coarse_true = []
        all_fine_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                coarse_labels = batch['coarse_label'].to(self.device)
                fine_labels = batch['fine_label'].to(self.device)
                
                coarse_logits, fine_logits = self.model(input_ids, attention_mask)
                
                coarse_loss = self.coarse_criterion(coarse_logits, coarse_labels)
                fine_loss = self.fine_criterion(fine_logits, fine_labels)
                total_loss += (self.coarse_weight * coarse_loss + self.fine_weight * fine_loss).item()
                
                _, coarse_pred = torch.max(coarse_logits.data, 1)
                _, fine_pred = torch.max(fine_logits.data, 1)
                
                total += coarse_labels.size(0)
                coarse_correct += (coarse_pred == coarse_labels).sum().item()
                fine_correct += (fine_pred == fine_labels).sum().item()
                
                all_coarse_pred.extend(coarse_pred.cpu().numpy())
                all_fine_pred.extend(fine_pred.cpu().numpy())
                all_coarse_true.extend(coarse_labels.cpu().numpy())
                all_fine_true.extend(fine_labels.cpu().numpy())
        
        coarse_acc = 100 * coarse_correct / total
        fine_acc = 100 * fine_correct / total
        avg_loss = total_loss / len(test_loader)
        
        return (avg_loss, coarse_acc, fine_acc, 
                np.array(all_coarse_pred), np.array(all_fine_pred),
                np.array(all_coarse_true), np.array(all_fine_true))

class HierarchicalClassificationPipeline:
    """Pipeline hoàn chỉnh cho hierarchical classification"""
    
    def __init__(self, model_name: str = 'vinai/phobert-base', max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        
        print(f"📥 Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.label_encoder = HierarchicalLabelEncoder()
        self.model = None
        self.trainer = None
    
    def prepare_data(self, texts: List[str], labels: List[str], test_size: float = 0.2):
        """Chuẩn bị dữ liệu hierarchical"""
        print("📊 Chuẩn bị dữ liệu hierarchical...")
        
        # Create hierarchical labels
        coarse_labels, fine_labels = self.label_encoder.create_hierarchy_from_data(texts, labels)
        coarse_encoded, fine_encoded = self.label_encoder.encode_hierarchical(texts, labels)
        
        self.num_coarse_classes = len(self.label_encoder.coarse_encoder.classes_)
        self.num_fine_classes = max(len(encoder.classes_) for encoder in self.label_encoder.fine_encoders.values())
        
        print(f"🎭 Coarse classes: {self.num_coarse_classes}")
        print(f"🎭 Fine classes: {self.num_fine_classes}")
        
        # Split data
        X_train, X_test, y_coarse_train, y_coarse_test, y_fine_train, y_fine_test = train_test_split(
            texts, coarse_encoded, fine_encoded, test_size=test_size, random_state=42, stratify=coarse_encoded
        )
        
        # Validation split
        X_train, X_val, y_coarse_train, y_coarse_val, y_fine_train, y_fine_val = train_test_split(
            X_train, y_coarse_train, y_fine_train, test_size=0.125, random_state=42, stratify=y_coarse_train
        )
        
        print(f"📏 Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        return (X_train, X_val, X_test, 
                y_coarse_train, y_coarse_val, y_coarse_test,
                y_fine_train, y_fine_val, y_fine_test)
    
    def create_data_loaders(self, X_train, X_val, X_test, 
                          y_coarse_train, y_coarse_val, y_coarse_test,
                          y_fine_train, y_fine_val, y_fine_test, batch_size: int = 16):
        """Tạo data loaders"""
        train_dataset = HierarchicalDataset(X_train, y_coarse_train, y_fine_train, 
                                          self.tokenizer, self.max_length)
        val_dataset = HierarchicalDataset(X_val, y_coarse_val, y_fine_val, 
                                        self.tokenizer, self.max_length)
        test_dataset = HierarchicalDataset(X_test, y_coarse_test, y_fine_test, 
                                         self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def build_model(self, device='cpu'):
        """Build hierarchical model"""
        print(f"🏗️ Xây dựng Hierarchical PhoBERT...")
        
        self.model = HierarchicalPhoBERT(
            model_name=self.model_name,
            num_coarse_classes=self.num_coarse_classes,
            num_fine_classes=self.num_fine_classes
        )
        
        self.trainer = HierarchicalTrainer(self.model, device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model built with {total_params:,} parameters")
        
        return self.model
    
    def train(self, train_loader, val_loader, num_epochs: int = 3, learning_rate: float = 2e-5):
        """Training"""
        if self.trainer is None:
            raise ValueError("Model chưa được build")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        print(f"🚀 Bắt đầu training hierarchical model...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
        
        best_val_acc = 0
        history = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_coarse_acc, train_fine_acc = self.trainer.train_epoch(train_loader, optimizer)
            
            # Validation
            val_loss, val_coarse_acc, val_fine_acc, _, _, _, _ = self.trainer.evaluate(val_loader)
            
            print(f"Train - Loss: {train_loss:.4f}, Coarse Acc: {train_coarse_acc:.2f}%, Fine Acc: {train_fine_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Coarse Acc: {val_coarse_acc:.2f}%, Fine Acc: {val_fine_acc:.2f}%")
            
            # Save best model
            val_combined_acc = (val_coarse_acc + val_fine_acc) / 2
            if val_combined_acc > best_val_acc:
                best_val_acc = val_combined_acc
                torch.save(self.model.state_dict(), 'best_hierarchical_model.pth')
                print(f"✅ Saved best model (Combined Val Acc: {val_combined_acc:.2f}%)")
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_coarse_acc': train_coarse_acc,
                'train_fine_acc': train_fine_acc,
                'val_loss': val_loss,
                'val_coarse_acc': val_coarse_acc,
                'val_fine_acc': val_fine_acc
            })
        
        print(f"\n🎉 Training hoàn thành!")
        print(f"🏆 Best combined validation accuracy: {best_val_acc:.2f}%")
        
        return history
    
    def evaluate(self, test_loader):
        """Evaluate hierarchical model"""
        if self.trainer is None:
            raise ValueError("Model chưa được train")
        
        return self.trainer.evaluate(test_loader)
    
    def predict_hierarchical(self, texts: List[str], batch_size: int = 16) -> List[Tuple[str, str]]:
        """Predict with hierarchical output"""
        if self.model is None:
            raise ValueError("Model chưa được train")
        
        # Create dataset
        dummy_coarse = np.zeros(len(texts))
        dummy_fine = np.zeros(len(texts))
        dataset = HierarchicalDataset(texts, dummy_coarse, dummy_fine, 
                                    self.tokenizer, self.max_length, use_chunks=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Predict
        self.model.eval()
        coarse_predictions = []
        fine_predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.trainer.device)
                attention_mask = batch['attention_mask'].to(self.trainer.device)
                
                coarse_logits, fine_logits = self.model(input_ids, attention_mask)
                
                _, coarse_pred = torch.max(coarse_logits.data, 1)
                _, fine_pred = torch.max(fine_logits.data, 1)
                
                coarse_predictions.extend(coarse_pred.cpu().numpy())
                fine_predictions.extend(fine_pred.cpu().numpy())
        
        # Decode predictions
        results = self.label_encoder.decode_hierarchical(
            np.array(coarse_predictions), np.array(fine_predictions)
        )
        
        return results

def test_hierarchical_pipeline():
    """Test hierarchical classification pipeline"""
    print("🧪 TESTING HIERARCHICAL CLASSIFICATION")
    print("="*50)
    
    # Sample data với hierarchical structure
    sample_texts = [
        "Luật Dân sự quy định về quyền và nghĩa vụ của cá nhân",
        "Nghị định hướng dẫn thi hành Luật Doanh nghiệp",
        "Thông tư hướng dẫn quy trình đăng ký kinh doanh",
        "Hợp đồng mua bán nhà đất giữa hai bên",
        "Bản án hình sự về tội trộm cắp tài sản",
        "Báo cáo tài chính của công ty"
    ] * 10  # Duplicate để có đủ data
    
    sample_labels = [
        "Luật", "Nghị định", "Thông tư", "Hợp đồng", "Bản án", "Báo cáo"
    ] * 10
    
    # Initialize pipeline
    pipeline = HierarchicalClassificationPipeline(max_length=128)
    
    # Prepare data
    data_splits = pipeline.prepare_data(sample_texts, sample_labels)
    X_train, X_val, X_test = data_splits[:3]
    y_coarse_train, y_coarse_val, y_coarse_test = data_splits[3:6]
    y_fine_train, y_fine_val, y_fine_test = data_splits[6:9]
    
    # Create data loaders
    train_loader, val_loader, test_loader = pipeline.create_data_loaders(
        X_train, X_val, X_test,
        y_coarse_train, y_coarse_val, y_coarse_test,
        y_fine_train, y_fine_val, y_fine_test,
        batch_size=4
    )
    
    # Build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline.build_model(device)
    
    # Training (1 epoch để test)
    history = pipeline.train(train_loader, val_loader, num_epochs=1, learning_rate=5e-5)
    
    # Evaluation
    test_results = pipeline.evaluate(test_loader)
    test_loss, test_coarse_acc, test_fine_acc = test_results[:3]
    
    print(f"\n✅ Test completed!")
    print(f"   Test Coarse Accuracy: {test_coarse_acc:.2f}%")
    print(f"   Test Fine Accuracy: {test_fine_acc:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Test prediction
    test_texts = ["Luật Hình sự về tội giết người", "Nghị định về thuế thu nhập"]
    predictions = pipeline.predict_hierarchical(test_texts)
    
    print(f"\n🔍 Sample predictions:")
    for text, (coarse, fine) in zip(test_texts, predictions):
        print(f"   Text: {text}")
        print(f"   Predicted: {coarse} -> {fine}")
    
    return True

if __name__ == "__main__":
    test_hierarchical_pipeline() 