#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📦 Setup script cho viLegalBert
Phân loại văn bản pháp luật Việt Nam với kiến trúc phân cấp 2 tầng
"""

from setuptools import setup, find_packages
from pathlib import Path

# Đọc README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Đọc requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="viLegalBert",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Mô hình phân loại văn bản pháp luật Việt Nam với kiến trúc phân cấp 2 tầng",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/viLegalBert",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vilegalbert=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    keywords=[
        "vietnamese",
        "legal",
        "text-classification",
        "nlp",
        "machine-learning",
        "deep-learning",
        "phobert",
        "bilstm",
        "svm",
        "hierarchical-classification"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/viLegalBert/issues",
        "Source": "https://github.com/yourusername/viLegalBert",
        "Documentation": "https://github.com/yourusername/viLegalBert/docs",
    },
) 