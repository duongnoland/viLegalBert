#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Main entry point cho viLegalBert
Phân loại văn bản pháp luật Việt Nam với kiến trúc phân cấp 2 tầng
"""

import argparse
import sys
from pathlib import Path
import logging

# Thêm thư mục hiện tại vào path
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """Thiết lập logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/main.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(
        description="viLegalBert - Phân loại văn bản pháp luật Việt Nam"
    )
    
    parser.add_argument(
        '--mode',
        choices=[
            'create_dataset', 
            'train_svm', 
            'train_phobert', 
            'train_bilstm', 
            'train_ensemble',
            'evaluate_all',
            'predict'
        ],
        default='create_dataset',
        help='Chế độ hoạt động'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/raw/vbpl_crawl.json',
        help='Đường dẫn đến file dữ liệu gốc'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/processed/',
        help='Đường dẫn output'
    )
    
    parser.add_argument(
        '--model_type',
        choices=['svm', 'phobert', 'bilstm', 'ensemble'],
        default='svm',
        help='Loại mô hình để train'
    )
    
    parser.add_argument(
        '--level',
        choices=['level1', 'level2', 'both'],
        default='both',
        help='Tầng để train (level1: loại văn bản, level2: domain pháp lý)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info("🚀 Khởi động viLegalBert")
    logger.info(f"📋 Mode: {args.mode}")
    logger.info(f"📁 Data path: {args.data_path}")
    logger.info(f"📁 Output path: {args.output_path}")
    
    try:
        if args.mode == 'create_dataset':
            logger.info("📊 Tạo dataset phân cấp 2 tầng...")
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from create_hierarchical_dataset import create_hierarchical_dataset, create_training_splits
            
            # Tạo dataset chính
            output_csv = Path(args.output_path) / "hierarchical_legal_dataset.csv"
            df = create_hierarchical_dataset(args.data_path, str(output_csv), target_size=10000)
            
            # Tạo các tập train/val/test
            splits_dir = Path(args.output_path) / "dataset_splits"
            create_training_splits(str(output_csv), str(splits_dir))
            
            logger.info("✅ Tạo dataset hoàn thành!")
            
        elif args.mode == 'train_svm':
            logger.info("🏋️ Training mô hình SVM...")
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from training.scripts.train_svm import SVMTrainer
            
            trainer = SVMTrainer()
            data_path = Path(args.output_path) / "hierarchical_legal_dataset.csv"
            
            if args.level in ['level1', 'both']:
                logger.info("🏷️ Training Level 1 (Loại văn bản)...")
                results_level1 = trainer.train_level1(str(data_path))
                
            if args.level in ['level2', 'both']:
                logger.info("🏷️ Training Level 2 (Domain pháp lý)...")
                results_level2 = trainer.train_level2(str(data_path))
                
            logger.info("✅ Training SVM hoàn thành!")
            
        elif args.mode == 'train_phobert':
            logger.info("🏋️ Training mô hình PhoBERT...")
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from training.scripts.train_phobert import PhoBERTTrainer
            
            trainer = PhoBERTTrainer()
            data_path = Path(args.output_path) / "hierarchical_legal_dataset.csv"
            
            if args.level in ['level1', 'both']:
                logger.info("🏷️ Training Level 1 (Loại văn bản)...")
                results_level1 = trainer.train_level1(str(data_path))
                
            if args.level in ['level2', 'both']:
                logger.info("🏷️ Training Level 2 (Domain pháp lý)...")
                results_level2 = trainer.train_level2(str(data_path))
                
            logger.info("✅ Training PhoBERT hoàn thành!")
            
        elif args.mode == 'train_bilstm':
            logger.info("🏋️ Training mô hình BiLSTM...")
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from training.scripts.train_bilstm import BiLSTMTrainer
            
            trainer = BiLSTMTrainer()
            data_path = Path(args.output_path) / "hierarchical_legal_dataset.csv"
            
            if args.level in ['level1', 'both']:
                logger.info("🏷️ Training Level 1 (Loại văn bản)...")
                results_level1 = trainer.train_level1(str(data_path))
                
            if args.level in ['level2', 'both']:
                logger.info("🏷️ Training Level 2 (Domain pháp lý)...")
                results_level2 = trainer.train_level2(str(data_path))
                
            logger.info("✅ Training BiLSTM hoàn thành!")
            
        elif args.mode == 'train_ensemble':
            logger.info("🏋️ Training mô hình Ensemble...")
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from training.scripts.train_ensemble import EnsembleTrainer
            
            trainer = EnsembleTrainer()
            data_path = Path(args.output_path) / "hierarchical_legal_dataset.csv"
            
            if args.level in ['level1', 'both']:
                logger.info("🏷️ Training Level 1 (Loại văn bản)...")
                results_level1 = trainer.train_level1(str(data_path))
                
            if args.level in ['level2', 'both']:
                logger.info("🏷️ Training Level 2 (Domain pháp lý)...")
                results_level2 = trainer.train_level2(str(data_path))
                
            logger.info("✅ Training Ensemble hoàn thành!")
            
        elif args.mode == 'evaluate_all':
            logger.info("📊 Đánh giá tất cả các mô hình...")
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from training.scripts.evaluate_all_models import ComprehensiveEvaluator
            
            evaluator = ComprehensiveEvaluator()
            test_data_path = Path(args.output_path) / "dataset_splits/test.csv"
            
            if not test_data_path.exists():
                logger.error(f"❌ Không tìm thấy file test data: {test_data_path}")
                logger.info("💡 Hãy chạy create_dataset trước")
                return
            
            results = evaluator.evaluate_all(str(test_data_path))
            logger.info("✅ Evaluation tổng hợp hoàn thành!")
            
        elif args.mode == 'predict':
            logger.info("🔮 Thực hiện dự đoán...")
            # TODO: Implement prediction interface
            logger.warning("⚠️ Prediction interface chưa được implement")
            
        else:
            logger.error(f"❌ Mode không hợp lệ: {args.mode}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình thực thi: {e}")
        return 1
    
    logger.info("🎉 Hoàn thành thành công!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 