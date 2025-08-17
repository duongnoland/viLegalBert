#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏋️ PhoBERT Trainer cho Google Colab (GPU Optimized)
Phân loại văn bản pháp luật Việt Nam với PhoBERT
"""

import os
import pickle
import pandas as pd
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
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "pandas",
        "numpy"
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
# 🏋️ PHOBERT TRAINER
# ============================================================================

# Import sau khi cài đặt
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

class PhoBERTTrainer:
    """Trainer cho mô hình PhoBERT với GPU optimization"""
    
    def __init__(self):
        self.model_name = "vinai/phobert-base"
        self.tokenizer = None
        self.model = None
        
        # Kiểm tra GPU
        self.use_gpu = setup_gpu()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"🚀 Sử dụng device: {self.device}")
        
        # Cấu hình training tối ưu cho GPU/CPU
        if self.use_gpu:
            self.config = {
                'max_length': 512,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'num_epochs': 3,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'gradient_accumulation_steps': 2,
                'fp16': True,
                'dataloader_num_workers': 4
            }
        else:
            self.config = {
                'max_length': 512,
                'batch_size': 8,
                'learning_rate': 2e-5,
                'num_epochs': 3,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'gradient_accumulation_steps': 1,
                'fp16': False,
                'dataloader_num_workers': 2
            }
        
        print(f"🚀 PhoBERTTrainer - GPU: {'✅' if self.use_gpu else '❌'}")
    
    def load_model(self, num_labels):
        """Load tokenizer và model"""
        print(f"📥 Loading PhoBERT với {num_labels} labels...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels, problem_type="single_label_classification"
        )
        
        # Chuyển model lên device
        self.model.to(self.device)
        
        # GPU optimization
        if self.use_gpu:
            if self.config['fp16']:
                self.model = self.model.half()
            torch.cuda.empty_cache()
            print(f"🚀 Model đã được tối ưu cho GPU")
        
        print(f"✅ Đã load PhoBERT")
        print(f"🚀 Device: {self.device}")
    
    def prepare_dataset(self, texts, labels, max_length=512):
        """Chuẩn bị dataset cho PhoBERT"""
        print("🔄 Chuẩn bị dataset...")
        
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
        
        return dataset
    
    def train_level1(self, data_path):
        """Training cho Level 1"""
        print("🏷️ Training Level 1...")
        
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
        self.load_model(num_labels)
        
        # Chia data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Chuẩn bị datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels, self.config['max_length'])
        val_dataset = self.prepare_dataset(val_texts, val_labels, self.config['max_length'])
        
        # Cấu hình training
        training_args = TrainingArguments(
            output_dir="./phobert_level1_results",
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir="./phobert_level1_logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            fp16=self.config['fp16'],
            dataloader_num_workers=self.config['dataloader_num_workers'],
            report_to=None,
            dataloader_pin_memory=True if self.use_gpu else False,
            remove_unused_columns=False
        )
        
        # Data collator và trainer
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, tokenizer=self.tokenizer, data_collator=data_collator
        )
        
        # Training
        print("🏋️ Bắt đầu training...")
        if self.use_gpu:
            print(f"🚀 GPU: Batch size {self.config['batch_size']}, Mixed precision {'✅' if self.config['fp16'] else '❌'}")
        
        trainer.train()
        
        # Evaluation
        print("📊 Evaluation...")
        eval_results = trainer.evaluate()
        
        # Lưu model
        base_dir = "/content/viLegalBert"
        model_path = f"{base_dir}/models/saved_models/level1_classifier/phobert_level1/phobert_level1_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Lưu label encoder và config
        with open(f"{model_path}/label_encoder.pkl", 'wb') as f:
            pickle.dump(label_encoder, f)
        
        with open(f"{model_path}/training_config.pkl", 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"💾 Model đã lưu: {model_path}")
        return {
            'model_path': model_path,
            'eval_results': eval_results,
            'label_encoder': label_encoder,
            'gpu_optimized': self.use_gpu
        }
    
    def train_level2(self, data_path):
        """Training cho Level 2"""
        print("🏷️ Training Level 2...")
        
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
        self.load_model(num_labels)
        
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
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            fp16=self.config['fp16'],
            dataloader_num_workers=self.config['dataloader_num_workers'],
            report_to=None,
            dataloader_pin_memory=True if self.use_gpu else False,
            remove_unused_columns=False
        )
        
        # Data collator và trainer
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, tokenizer=self.tokenizer, data_collator=data_collator
        )
        
        # Training
        print("🏋️ Bắt đầu training...")
        if self.use_gpu:
            print(f"🚀 GPU: Batch size {self.config['batch_size']}, Mixed precision {'✅' if self.config['fp16'] else '❌'}")
        
        trainer.train()
        
        # Evaluation
        print("📊 Evaluation...")
        eval_results = trainer.evaluate()
        
        # Lưu model
        base_dir = "/content/viLegalBert"
        model_path = f"{base_dir}/models/saved_models/level2_classifier/phobert_level2/phobert_level2_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Lưu label encoder và config
        with open(f"{model_path}/label_encoder.pkl", 'wb') as f:
            pickle.dump(label_encoder, f)
        
        with open(f"{model_path}/training_config.pkl", 'wb') as f:
            pickle.dump(self.config, f)
        
        print(f"💾 Model đã lưu: {model_path}")
        return {
            'model_path': model_path,
            'eval_results': eval_results,
            'label_encoder': label_encoder,
            'gpu_optimized': self.use_gpu
        }

def main():
    """Hàm chính"""
    print("🏋️ PHOBERT TRAINER - GPU OPTIMIZED")
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
    os.makedirs(f"{base_dir}/models/saved_models/level1_classifier/phobert_level1", exist_ok=True)
    os.makedirs(f"{base_dir}/models/saved_models/level2_classifier/phobert_level2", exist_ok=True)
    
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
    trainer = PhoBERTTrainer()
    
    # Bước 6: Training Level 1
    print("\n🏷️ TRAINING LEVEL 1...")
    train_path = f"{base_dir}/data/processed/dataset_splits/train.csv"
    results_level1 = trainer.train_level1(train_path)  # Chỉ training trên train set
    
    # Bước 7: Training Level 2
    print("\n🏷️ TRAINING LEVEL 2...")
    results_level2 = trainer.train_level2(train_path)  # Chỉ training trên train set
    
    # Tóm tắt kết quả
    print("\n🎉 PHOBERT TRAINING HOÀN THÀNH!")
    print("=" * 80)
    print(f"📊 Level 1 model: {results_level1['model_path']}")
    print(f"📊 Level 2 model: {results_level2['model_path']}")
    print(f"🚀 GPU Status: {'✅ Available' if gpu_available else '❌ Not Available'}")

if __name__ == "__main__":
    main() 