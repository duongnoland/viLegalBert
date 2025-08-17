"""
Data loader và tiền xử lý cho phân loại văn bản pháp lý
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class LegalTextDataset(Dataset):
    """Dataset cho văn bản pháp lý"""
    def __init__(self, texts, labels, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            # Cho PhoBERT
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
        
        return {'text': text, 'label': label}

class DataLoader:
    """Tải và tiền xử lý dữ liệu văn bản pháp lý"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
    
    def clean_text(self, text):
        """Làm sạch văn bản"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Loại bỏ ký tự đặc biệt
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_data(self, text_column='text', label_column='label', sample_size=None):
        """
        Tải dữ liệu từ file CSV
        
        Args:
            text_column: Tên cột chứa văn bản
            label_column: Tên cột chứa nhãn
            sample_size: Số lượng mẫu (None = toàn bộ)
        """
        print(f"Đang tải dữ liệu từ {self.data_path}...")
        
        # Đọc dữ liệu
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.json'):
            df = pd.read_json(self.data_path)
        else:
            raise ValueError("Chỉ hỗ trợ file .csv và .json")
        
        # Lấy mẫu nếu cần
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Làm sạch dữ liệu
        df[text_column] = df[text_column].apply(self.clean_text)
        
        # Loại bỏ các dòng trống
        df = df[df[text_column] != ""]
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(df[label_column])
        
        print(f"Đã tải {len(df)} mẫu với {len(self.label_encoder.classes_)} lớp")
        print(f"Các lớp: {self.label_encoder.classes_}")
        
        return df[text_column].values, labels_encoded, self.label_encoder.classes_
    
    def split_data(self, texts, labels, test_size=0.2, val_size=0.1, random_state=42):
        """Chia dữ liệu train/val/test"""
        
        # Chia train và temp (test + val)
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=(test_size + val_size), 
            random_state=random_state, stratify=labels
        )
        
        # Chia temp thành val và test
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio), 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                          tokenizer=None, batch_size=32, max_length=512):
        """Tạo DataLoader cho deep learning models"""
        
        train_dataset = LegalTextDataset(X_train, y_train, tokenizer, max_length)
        val_dataset = LegalTextDataset(X_val, y_val, tokenizer, max_length)
        test_dataset = LegalTextDataset(X_test, y_test, tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader 