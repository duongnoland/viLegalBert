#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏋️ PhoBERT Trainer cho Google Colab (Dataset Có Sẵn)
Phân loại văn bản pháp luật Việt Nam với PhoBERT
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
        import transformers
        print("✅ transformers đã sẵn sàng")
    except:
        os.system("pip install transformers")
        print("📦 Đã cài đặt transformers")
    
    try:
        import torch
        print("✅ PyTorch đã sẵn sàng")
    except:
        os.system("pip install torch")
        print("📦 Đã cài đặt PyTorch")
    
    try:
        import datasets
        print("✅ datasets đã sẵn sàng")
    except:
        os.system("pip install datasets")
        print("📦 Đã cài đặt datasets")

# Import sau khi cài đặt
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

class PhoBERTTrainer:
    """Trainer cho mô hình PhoBERT"""
    
    def __init__(self):
        """Khởi tạo trainer"""
        self.model_name = "vinai/phobert-base"
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Sử dụng device: {self.device}")
        
        # Cấu hình training
        self.config = {
            'max_length': 512,
            'batch_size': 8,
            'learning_rate': 2e-5,
            'num_epochs': 3,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'logging_steps': 100,
            'eval_steps': 500,
            'save_steps': 1000
        }
    
    def load_tokenizer_and_model(self, num_labels):
        """Load tokenizer và model"""
        print(f"📥 Loading PhoBERT tokenizer và model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        
        # Chuyển model lên device
        self.model.to(self.device)
        
        print(f"✅ Đã load PhoBERT với {num_labels} labels")
    
    def prepare_dataset(self, texts, labels, max_length=512):
        """Chuẩn bị dataset cho PhoBERT"""
        print("🔄 Chuẩn bị dataset...")
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Tạo dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        
        return dataset
    
    def train_level1(self, data_path: str):
        """Training cho Level 1 (Loại văn bản)"""
        print("🏷️ Training Level 1 (Loại văn bản) với PhoBERT...")
        
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        texts = df['text'].fillna('').tolist()
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df['type_level1'])
        num_labels = len(label_encoder.classes_)
        
        print(f"📊 Số labels: {num_labels}")
        print(f"📊 Classes: {label_encoder.classes_}")
        
        # Load model
        self.load_tokenizer_and_model(num_labels)
        
        # Chia data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Chuẩn bị datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels, self.config['max_length'])
        val_dataset = self.prepare_dataset(val_labels, val_labels, self.config['max_length'])
        
        # Cấu hình training
        training_args = TrainingArguments(
            output_dir="./phobert_level1_results",
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir="./phobert_level1_logs",
            logging_steps=self.config['logging_steps'],
            evaluation_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_steps=self.config['save_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            report_to=None
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Training
        print("🏋️ Bắt đầu training...")
        trainer.train()
        
        # Evaluation
        print("📊 Evaluation...")
        eval_results = trainer.evaluate()
        
        # Lưu model
        model_path = "models/saved_models/level1_classifier/phobert_level1/phobert_level1_model"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Lưu label encoder
        with open(f"{model_path}/label_encoder.pkl", 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"💾 Model đã được lưu: {model_path}")
        
        return {
            'model_path': model_path,
            'eval_results': eval_results,
            'label_encoder': label_encoder
        }
    
    def train_level2(self, data_path: str):
        """Training cho Level 2 (Domain pháp lý)"""
        print("🏷️ Training Level 2 (Domain pháp lý) với PhoBERT...")
        
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        texts = df['text'].fillna('').tolist()
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df['domain_level2'])
        num_labels = len(label_encoder.classes_)
        
        print(f"📊 Số labels: {num_labels}")
        print(f"📊 Classes: {label_encoder.classes_}")
        
        # Load model
        self.load_tokenizer_and_model(num_labels)
        
        # Chia data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Chuẩn bị datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels, self.config['max_length'])
        val_dataset = self.prepare_dataset(val_texts, val_labels, self.config['max_length'])
        
        # Cấu hình training
        training_args = TrainingArguments(
            output_dir="./phobert_level2_results",
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir="./phobert_level2_logs",
            logging_steps=self.config['logging_steps'],
            evaluation_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_steps=self.config['save_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            report_to=None
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Training
        print("🏋️ Bắt đầu training...")
        trainer.train()
        
        # Evaluation
        print("📊 Evaluation...")
        eval_results = trainer.evaluate()
        
        # Lưu model
        model_path = "models/saved_models/level2_classifier/phobert_level2/phobert_level2_model"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Lưu label encoder
        with open(f"{model_path}/label_encoder.pkl", 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"💾 Model đã được lưu: {model_path}")
        
        return {
            'model_path': model_path,
            'eval_results': eval_results,
            'label_encoder': label_encoder
        }

def main():
    """Hàm chính"""
    print("🏋️ PHOBERT TRAINER CHO GOOGLE COLAB!")
    print("📊 SỬ DỤNG DATASET CÓ SẴN")
    print("=" * 50)
    
    # Cài đặt dependencies
    install_deps()
    
    # Tạo cấu trúc thư mục
    from pathlib import Path
    Path("models/saved_models/level1_classifier/phobert_level1").mkdir(parents=True, exist_ok=True)
    Path("models/saved_models/level2_classifier/phobert_level2").mkdir(parents=True, exist_ok=True)
    
    # Kiểm tra dataset có sẵn
    dataset_path = "data/processed/hierarchical_legal_dataset.csv"
    if not Path(dataset_path).exists():
        print(f"❌ Không tìm thấy dataset: {dataset_path}")
        print("🔍 Tìm kiếm dataset trong các thư mục...")
        
        possible_paths = [
            "hierarchical_legal_dataset.csv",
            "data/hierarchical_legal_dataset.csv",
            "dataset.csv",
            "legal_dataset.csv"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                dataset_path = path
                print(f"✅ Tìm thấy dataset: {dataset_path}")
                break
        else:
            print("❌ Không tìm thấy dataset nào. Vui lòng upload dataset vào Colab")
            return
    
    # Khởi tạo trainer
    trainer = PhoBERTTrainer()
    
    # Training Level 1
    print("\n🏷️ TRAINING LEVEL 1...")
    results_level1 = trainer.train_level1(dataset_path)
    
    # Training Level 2
    print("\n🏷️ TRAINING LEVEL 2...")
    results_level2 = trainer.train_level2(dataset_path)
    
    print("\n🎉 PHOBERT TRAINING HOÀN THÀNH!")
    print(f"📊 Level 1 model: {results_level1['model_path']}")
    print(f"📊 Level 2 model: {results_level2['model_path']}")

if __name__ == "__main__":
    main() 