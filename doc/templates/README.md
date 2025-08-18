# Templates cho Chương Thực nghiệm

Thư mục này cung cấp khung LaTeX và Markdown để biên soạn chương Thực nghiệm.

## 1) LaTeX (experiment_chapter.tex)
- Biên dịch đề xuất bằng XeLaTeX (hỗ trợ tiếng Việt tốt):

```
xelatex experiment_chapter.tex
```

- Thay thế các chỗ “Điền từ ...csv” bằng dữ liệu sinh ra từ script:
  - results/report/stats_type_level1.csv
  - results/report/stats_domain_level2.csv
  - results/report/main_results.csv
  - results/report/ablation_phobert_level1.csv

- Đảm bảo các hình đã được tạo:
  - results/report/length_distribution_Train.png
  - results/report/cm_*.png
  - results/report/learning_curves_*.png

## 2) Markdown (experiment_chapter.md)
- Có thể mở trong VS Code hoặc chuyển sang Word/PDF bằng pandoc:

```
pandoc experiment_chapter.md -o experiment_chapter.docx
pandoc experiment_chapter.md -o experiment_chapter.pdf
```

## 3) Sinh số liệu/hình từ code
- Chạy báo cáo trên Colab:

```
python scripts/colab_compare_and_report.py --show_ui --run_ablation
```

- Kết quả sẽ nằm trong results/report/ và có thể chèn trực tiếp vào LaTeX/Markdown.


