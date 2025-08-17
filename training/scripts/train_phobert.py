#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Script Training cho mô hình PhoBERT - viLegalBert
Phân loại văn bản pháp luật Việt Nam sử dụng PhoBERT
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Thêm src vào path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from transformers import (
    AutoTokenizer, AutoModel, 
    TrainingArguments, Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_phobert.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PhoBERTTrainer:
    """Trainer cho mô hình PhoBERT"""
    
    def __init__(self, config_path: str = "config/model_configs/phobert_config.yaml"):
        """Khởi tạo trainer"""
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Setup device
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
    
    def _setup_tokenizer(self) -> None:
        """Thiết lập tokenizer"""
        try:
            model_name = self.config['tokenizer']['model_name']
            logger.info(f"🔧 Thiết lập tokenizer: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Thêm padding token nếu cần
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"✅ Tokenizer đã sẵn sàng")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi thiết lập tokenizer: {e}")
            raise
    
    def _setup_model(self, num_classes: int, level: str) -> None:
        """Thiết lập mô hình PhoBERT"""
        try:
            model_name = self.config['model']['model_name']
            logger.info(f"🔧 Thiết lập mô hình: {model_name}")
            
            # Load pretrained model
            base_model = AutoModel.from_pretrained(model_name)
            
            # Tạo classifier head
            from transformers import AutoModelForSequenceClassification
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                problem_type="single_label_classification"
            )
            
            # Move to device
            self.model.to(self.device)
            
            logger.info(f"✅ Mô hình đã sẵn sàng cho {level}")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi thiết lập mô hình: {e}")
            raise
    
    def _tokenize_data(self, texts: pd.Series, max_length: int = None) -> Dataset:
        """Tokenize dữ liệu"""
        try:
            if max_length is None:
                max_length = self.config['tokenizer']['max_length']
            
            logger.info(f"🔧 Tokenizing {len(texts)} texts với max_length={max_length}")
            
            # Tokenize texts
            tokenized = self.tokenizer(
                texts.tolist(),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Convert to Dataset
            dataset = Dataset.from_dict({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': None  # Will be set later
            })
            
            logger.info(f"✅ Tokenization hoàn thành")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tokenize dữ liệu: {e}")
            raise
    
    def _prepare_dataset(self, texts: pd.Series, labels: pd.Series, level: str) -> Dataset:
        """Chuẩn bị dataset cho training"""
        try:
            logger.info(f"🔧 Chuẩn bị dataset cho {level}")
            
            # Tokenize texts
            dataset = self._tokenize_data(texts)
            
            # Add labels
            dataset = dataset.add_column('labels', labels.tolist())
            
            # Split train/validation
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            logger.info(f"✅ Dataset đã sẵn sàng: {len(train_dataset)} train, {len(val_dataset)} val")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi chuẩn bị dataset: {e}")
            raise
    
    def _setup_training_args(self, level: str) -> TrainingArguments:
        """Thiết lập training arguments"""
        try:
            training_config = self.config['training']
            
            # Tạo output directory
            output_dir = f"models/checkpoints/level{level[-1]}_checkpoints/phobert_{level}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=training_config['num_epochs'],
                per_device_train_batch_size=training_config['batch_size'],
                per_device_eval_batch_size=training_config['batch_size'],
                warmup_steps=training_config['warmup_steps'],
                weight_decay=training_config['weight_decay'],
                logging_dir=f"logs/training_logs/phobert_{level}",
                logging_steps=training_config['logging_steps'],
                evaluation_strategy="steps",
                eval_steps=training_config['eval_steps'],
                save_steps=training_config['save_steps'],
                save_total_limit=training_config['save_total_limit'],
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                report_to=None,  # Disable wandb for now
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                fp16=torch.cuda.is_available(),  # Use mixed precision if available
            )
            
            logger.info(f"✅ Training arguments đã sẵn sàng")
            return training_args
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi thiết lập training arguments: {e}")
            raise
    
    def _setup_trainer(self, train_dataset: Dataset, val_dataset: Dataset, level: str) -> Trainer:
        """Thiết lập trainer"""
        try:
            logger.info(f"🔧 Thiết lập trainer cho {level}")
            
            # Training arguments
            training_args = self._setup_training_args(level)
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # Metrics
            metric = evaluate.load("f1")
            
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return metric.compute(predictions=predictions, references=labels, average="weighted")
            
            # Trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            logger.info(f"✅ Trainer đã sẵn sàng")
            return self.trainer
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi thiết lập trainer: {e}")
            raise
    
    def _train_model(self, level: str) -> Dict[str, Any]:
        """Train mô hình"""
        try:
            logger.info(f"🏋️ Bắt đầu training mô hình {level}")
            
            # Train
            train_result = self.trainer.train()
            
            # Evaluate
            eval_result = self.trainer.evaluate()
            
            # Save model
            self.trainer.save_model()
            
            logger.info(f"✅ Training hoàn thành cho {level}")
            logger.info(f"📊 Train loss: {train_result.training_loss:.4f}")
            logger.info(f"📊 Eval loss: {eval_result['eval_loss']:.4f}")
            logger.info(f"📊 Eval F1: {eval_result['eval_f1']:.4f}")
            
            return {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'eval_f1': eval_result['eval_f1'],
                'eval_accuracy': eval_result.get('eval_accuracy', 0.0),
                'eval_precision': eval_result.get('eval_precision', 0.0),
                'eval_recall': eval_result.get('eval_recall', 0.0),
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training mô hình: {e}")
            raise
    
    def _save_model(self, level: str, results: Dict[str, Any]) -> None:
        """Lưu mô hình và kết quả"""
        try:
            # Tạo thư mục lưu
            save_dir = Path(f"models/saved_models/level{level[-1]}_classifier/phobert_level{level[-1]}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu mô hình
            model_path = save_dir / "phobert_model"
            self.trainer.save_model(str(model_path))
            
            # Lưu tokenizer
            tokenizer_path = save_dir / "tokenizer"
            self.tokenizer.save_pretrained(str(tokenizer_path))
            
            # Lưu kết quả
            results_path = save_dir / "evaluation_results.yaml"
            with open(results_path, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
            
            # Lưu metadata
            metadata = {
                'model_type': 'PhoBERT',
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
            logger.info(f"✅ Lưu tokenizer vào {tokenizer_path}")
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
            
            # Setup tokenizer
            self._setup_tokenizer()
            
            # Setup model
            num_classes = y_level1.nunique()
            self._setup_model(num_classes, "level1")
            
            # Prepare dataset
            train_dataset, val_dataset = self._prepare_dataset(X, y_level1, "level1")
            
            # Setup trainer
            self._setup_trainer(train_dataset, val_dataset, "level1")
            
            # Train model
            results = self._train_model("level1")
            
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
            
            # Setup tokenizer (nếu chưa có)
            if self.tokenizer is None:
                self._setup_tokenizer()
            
            # Setup model
            num_classes = y_level2.nunique()
            self._setup_model(num_classes, "level2")
            
            # Prepare dataset
            train_dataset, val_dataset = self._prepare_dataset(X, y_level2, "level2")
            
            # Setup trainer
            self._setup_trainer(train_dataset, val_dataset, "level2")
            
            # Train model
            results = self._train_model("level2")
            
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
        trainer = PhoBERTTrainer()
        
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
        logger.info("📊 TÓM TẮT KẾT QUẢ TRAINING PHOBERT")
        logger.info("=" * 60)
        logger.info(f"🎯 Level 1 - F1: {results_level1['eval_f1']:.4f}, Loss: {results_level1['eval_loss']:.4f}")
        logger.info(f"🎯 Level 2 - F1: {results_level2['eval_f1']:.4f}, Loss: {results_level2['eval_loss']:.4f}")
        logger.info("🎉 Training PhoBERT hoàn thành thành công!")
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 