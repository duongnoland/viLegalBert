#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sao lưu models và kết quả (results) lên Google Drive.

Mặc định (luôn thực hiện cả 2):
- Models nguồn:  /content/viLegalBert/models/saved_models → Drive: /content/drive/MyDrive/viLegalBert/models/saved_models
- Results nguồn: /content/viLegalBert/results            → Drive: /content/drive/MyDrive/viLegalBert/results

Có thể thay đổi qua tham số dòng lệnh hoặc biến môi trường (xem --help):
- GOOGLE_DRIVE_DIR: override thư mục gốc của Google Drive (vd: /content/drive/MyDrive)
"""

import os
import sys
import shutil
import argparse


def try_mount_google_drive(default_mount_point: str = "/content/drive") -> None:
    """Mount Google Drive trong môi trường Colab nếu có sẵn module google.colab."""
    try:
        if not os.path.exists(default_mount_point):
            from google.colab import drive  # type: ignore
            print("🔌 Mounting Google Drive...")
            drive.mount(default_mount_point)
            print("✅ Google Drive mounted")
    except Exception as exc:
        # Không phải môi trường Colab hoặc không thể mount; bỏ qua
        print(f"ℹ️ Bỏ qua mount Google Drive: {exc}")


def copy_directory_tree(source_dir: str, target_dir: str, overwrite: bool = True) -> None:
    """Sao chép đệ quy toàn bộ thư mục từ source sang target.

    - Nếu overwrite=True, xóa target_dir trước khi copy.
    - Giữ nguyên cấu trúc thư mục; dùng copy2 để giữ metadata cơ bản.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Nguồn không tồn tại: {source_dir}")

    if overwrite and os.path.exists(target_dir):
        print(f"🧹 Xóa thư mục đích cũ: {target_dir}")
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    for root, dir_names, file_names in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        dest_root = target_dir if rel_path == "." else os.path.join(target_dir, rel_path)
        os.makedirs(dest_root, exist_ok=True)

        for directory_name in dir_names:
            os.makedirs(os.path.join(dest_root, directory_name), exist_ok=True)

        for file_name in file_names:
            src_file = os.path.join(root, file_name)
            dst_file = os.path.join(dest_root, file_name)
            shutil.copy2(src_file, dst_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sao lưu models/saved_models lên Google Drive")
    parser.add_argument(
        "--source",
        default=os.getenv("VILEGALBERT_MODELS_DIR", "/content/viLegalBert/models/saved_models"),
        help="Thư mục nguồn chứa models đã train (mặc định: /content/viLegalBert/models/saved_models)",
    )
    parser.add_argument(
        "--drive-dir",
        default=os.getenv("GOOGLE_DRIVE_DIR", "/content/drive/MyDrive"),
        help="Thư mục gốc của Google Drive (mặc định: /content/drive/MyDrive)",
    )
    parser.add_argument(
        "--target-subdir",
        default="viLegalBert/models/saved_models",
        help="Đường dẫn con trong Drive để lưu (mặc định: viLegalBert/models/saved_models)",
    )
    parser.add_argument(
        "--results-source",
        default=os.getenv("VILEGALBERT_RESULTS_DIR", "/content/viLegalBert/results"),
        help="Thư mục nguồn chứa kết quả (mặc định: /content/viLegalBert/results)",
    )
    parser.add_argument(
        "--results-target-subdir",
        default="viLegalBert/results",
        help="Đường dẫn con trong Drive để lưu kết quả (mặc định: viLegalBert/results)",
    )
    parser.add_argument(
        "--skip-results",
        action="store_true",
        help="Bỏ qua sao lưu thư mục results (mặc định là sao lưu)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Không xóa trước nếu thư mục đích đã tồn tại",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Mount Drive nếu hợp lệ
    try_mount_google_drive()

    source_dir = os.path.abspath(args.source)
    drive_root = os.path.abspath(args.drive_dir)
    target_dir = os.path.join(drive_root, args.target_subdir)

    print("📁 Nguồn:", source_dir)
    print("☁️  Drive:", drive_root)
    print("🎯 Đích:", target_dir)

    if not os.path.exists(drive_root):
        print(
            "❌ Không tìm thấy Google Drive. Hãy mount Drive vào '/content/drive' hoặc đặt biến môi trường 'GOOGLE_DRIVE_DIR'."
        )
        sys.exit(1)

    try:
        # Sao lưu models
        copy_directory_tree(source_dir, target_dir, overwrite=(not args.no_overwrite))
        print("✅ Sao lưu models hoàn tất!")

        # Sao lưu results (mặc định luôn chạy, có thể bỏ qua bằng --skip-results)
        if not args.skip_results:
            results_source = os.path.abspath(args.results_source)
            results_target = os.path.join(drive_root, args.results_target_subdir)
            print("📁 Nguồn (results):", results_source)
            print("🎯 Đích (results):", results_target)
            copy_directory_tree(results_source, results_target, overwrite=(not args.no_overwrite))
            print("✅ Sao lưu results hoàn tất!")

        print("✅ Tất cả sao lưu đã hoàn tất!")
    except Exception as exc:
        print(f"❌ Sao lưu thất bại: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()


