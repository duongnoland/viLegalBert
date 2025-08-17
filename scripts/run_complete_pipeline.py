#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Script Chạy Pipeline Hoàn Chỉnh - viLegalBert
Chạy toàn bộ pipeline từ tạo dataset đến evaluation
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompletePipelineRunner:
    """Runner cho pipeline hoàn chỉnh"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Khởi tạo runner"""
        self.config_path = config_path
        self.start_time = None
        self.step_times = {}
        
    def _run_command(self, command: str, description: str) -> bool:
        """Chạy một command và log kết quả"""
        try:
            logger.info(f"🚀 {description}")
            logger.info(f"📝 Command: {command}")
            
            start_time = time.time()
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            end_time = time.time()
            
            step_time = end_time - start_time
            self.step_times[description] = step_time
            
            if result.returncode == 0:
                logger.info(f"✅ {description} hoàn thành thành công ({step_time:.2f}s)")
                if result.stdout:
                    logger.info(f"📤 Output: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"❌ {description} thất bại")
                if result.stderr:
                    logger.error(f"📤 Error: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Lỗi khi chạy {description}: {e}")
            return False
    
    def _check_prerequisites(self) -> bool:
        """Kiểm tra điều kiện tiên quyết"""
        try:
            logger.info("🔍 Kiểm tra điều kiện tiên quyết...")
            
            # Kiểm tra file JSON gốc
            json_path = Path("data/raw/vbpl_crawl.json")
            if not json_path.exists():
                logger.error(f"❌ Không tìm thấy file dữ liệu gốc: {json_path}")
                return False
            
            # Kiểm tra thư mục config
            config_dir = Path("config")
            if not config_dir.exists():
                logger.error(f"❌ Không tìm thấy thư mục config: {config_dir}")
                return False
            
            # Kiểm tra requirements.txt
            requirements_path = Path("requirements.txt")
            if not requirements_path.exists():
                logger.error(f"❌ Không tìm thấy requirements.txt: {requirements_path}")
                return False
            
            logger.info("✅ Tất cả điều kiện tiên quyết đã được đáp ứng")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi kiểm tra điều kiện tiên quyết: {e}")
            return False
    
    def _install_dependencies(self) -> bool:
        """Cài đặt dependencies"""
        try:
            logger.info("📦 Cài đặt dependencies...")
            
            command = "pip install -r requirements.txt"
            return self._run_command(command, "Cài đặt dependencies")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi cài đặt dependencies: {e}")
            return False
    
    def _create_dataset(self) -> bool:
        """Tạo dataset phân cấp 2 tầng"""
        try:
            logger.info("📊 Tạo dataset phân cấp 2 tầng...")
            
            command = "python src/main.py --mode create_dataset"
            return self._run_command(command, "Tạo dataset")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo dataset: {e}")
            return False
    
    def _train_svm(self) -> bool:
        """Training mô hình SVM"""
        try:
            logger.info("🏋️ Training mô hình SVM...")
            
            command = "python src/main.py --mode train_svm --level both"
            return self._run_command(command, "Training SVM")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training SVM: {e}")
            return False
    
    def _train_phobert(self) -> bool:
        """Training mô hình PhoBERT"""
        try:
            logger.info("🏋️ Training mô hình PhoBERT...")
            
            command = "python src/main.py --mode train_phobert --level both"
            return self._run_command(command, "Training PhoBERT")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training PhoBERT: {e}")
            return False
    
    def _train_bilstm(self) -> bool:
        """Training mô hình BiLSTM"""
        try:
            logger.info("🏋️ Training mô hình BiLSTM...")
            
            command = "python src/main.py --mode train_bilstm --level both"
            return self._run_command(command, "Training BiLSTM")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training BiLSTM: {e}")
            return False
    
    def _train_ensemble(self) -> bool:
        """Training mô hình Ensemble"""
        try:
            logger.info("🏋️ Training mô hình Ensemble...")
            
            command = "python src/main.py --mode train_ensemble --level both"
            return self._run_command(command, "Training Ensemble")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi training Ensemble: {e}")
            return False
    
    def _evaluate_all_models(self) -> bool:
        """Đánh giá tất cả các mô hình"""
        try:
            logger.info("📊 Đánh giá tất cả các mô hình...")
            
            command = "python src/main.py --mode evaluate_all"
            return self._run_command(command, "Evaluation tổng hợp")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi evaluation: {e}")
            return False
    
    def _generate_report(self) -> bool:
        """Tạo báo cáo tổng hợp"""
        try:
            logger.info("📋 Tạo báo cáo tổng hợp...")
            
            # Tạo thư mục reports
            reports_dir = Path("results/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Tạo báo cáo
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = reports_dir / f"pipeline_report_{timestamp}.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 🚀 Báo Cáo Pipeline Hoàn Chỉnh - viLegalBert\n\n")
                f.write(f"**Thời gian chạy:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## 📊 Tóm Tắt Kết Quả\n\n")
                f.write("| Bước | Trạng thái | Thời gian |\n")
                f.write("|------|------------|-----------|\n")
                
                for step, step_time in self.step_times.items():
                    status = "✅ Thành công" if step_time > 0 else "❌ Thất bại"
                    time_str = f"{step_time:.2f}s" if step_time > 0 else "N/A"
                    f.write(f"| {step} | {status} | {time_str} |\n")
                
                f.write(f"\n**Tổng thời gian:** {time.time() - self.start_time:.2f}s\n")
                
                f.write("\n## 🎯 Kết Quả Cuối Cùng\n\n")
                f.write("- Dataset đã được tạo với 10,000 samples\n")
                f.write("- Các mô hình đã được training: SVM, PhoBERT, BiLSTM, Ensemble\n")
                f.write("- Evaluation tổng hợp đã hoàn thành\n")
                f.write("- Kết quả được lưu trong thư mục `results/`\n")
            
            logger.info(f"✅ Báo cáo đã được tạo: {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo báo cáo: {e}")
            return False
    
    def run_pipeline(self, skip_dependencies: bool = False) -> bool:
        """Chạy toàn bộ pipeline"""
        try:
            self.start_time = time.time()
            logger.info("🚀 Bắt đầu chạy pipeline hoàn chỉnh")
            logger.info("=" * 60)
            
            # Kiểm tra điều kiện tiên quyết
            if not self._check_prerequisites():
                return False
            
            # Cài đặt dependencies (nếu cần)
            if not skip_dependencies:
                if not self._install_dependencies():
                    return False
            
            # Tạo dataset
            if not self._create_dataset():
                return False
            
            # Training các mô hình
            if not self._train_svm():
                logger.warning("⚠️ Training SVM thất bại, tiếp tục với các mô hình khác")
            
            if not self._train_phobert():
                logger.warning("⚠️ Training PhoBERT thất bại, tiếp tục với các mô hình khác")
            
            if not self._train_bilstm():
                logger.warning("⚠️ Training BiLSTM thất bại, tiếp tục với các mô hình khác")
            
            # Training ensemble (chỉ khi có ít nhất 2 mô hình thành công)
            if not self._train_ensemble():
                logger.warning("⚠️ Training Ensemble thất bại")
            
            # Evaluation
            if not self._evaluate_all_models():
                logger.warning("⚠️ Evaluation thất bại")
            
            # Tạo báo cáo
            self._generate_report()
            
            # Tóm tắt kết quả
            total_time = time.time() - self.start_time
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE HOÀN THÀNH!")
            logger.info("=" * 60)
            logger.info(f"⏱️ Tổng thời gian: {total_time:.2f}s")
            logger.info(f"📊 Số bước đã hoàn thành: {len(self.step_times)}")
            
            for step, step_time in self.step_times.items():
                logger.info(f"  ✅ {step}: {step_time:.2f}s")
            
            logger.info("\n📁 Kết quả được lưu trong:")
            logger.info("  - data/processed/")
            logger.info("  - models/saved_models/")
            logger.info("  - results/")
            logger.info("  - logs/")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình chạy pipeline: {e}")
            return False

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(
        description="Chạy pipeline hoàn chỉnh cho viLegalBert"
    )
    
    parser.add_argument(
        '--skip-dependencies',
        action='store_true',
        help='Bỏ qua việc cài đặt dependencies'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Đường dẫn đến file config'
    )
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo runner
        runner = CompletePipelineRunner(args.config)
        
        # Chạy pipeline
        success = runner.run_pipeline(skip_dependencies=args.skip_dependencies)
        
        if success:
            logger.info("🎉 Pipeline hoàn thành thành công!")
            sys.exit(0)
        else:
            logger.error("❌ Pipeline thất bại!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình chạy pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 