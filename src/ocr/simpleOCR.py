import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import glob
from pathlib import Path


class OCRWorkflow:
    def __init__(self, input_folder="../../data/raw/pdf", output_folder="../../data/raw/ocr_output"):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Tạo thư mục output nếu chưa có
        os.makedirs(output_folder, exist_ok=True)

    def find_unprocessed_pdfs(self):
        """Tìm file PDF chưa có _done"""
        # Debug: Kiểm tra đường dẫn
        print(f"🔍 Đang tìm PDF trong: {os.path.abspath(self.input_folder)}")
        print(f"📁 Thư mục có tồn tại: {os.path.exists(self.input_folder)}")

        # Liệt kê tất cả file trong thư mục
        if os.path.exists(self.input_folder):
            all_files = os.listdir(self.input_folder)
            print(f"📋 Tất cả file: {all_files}")

        # Tìm file PDF
        search_pattern = os.path.join(self.input_folder, "*.pdf")
        print(f"🔍 Pattern tìm kiếm: {search_pattern}")

        pdf_files = glob.glob(search_pattern)
        print(f"📄 Tìm thấy {len(pdf_files)} file PDF: {[os.path.basename(f) for f in pdf_files]}")

        unprocessed = []

        for pdf_file in pdf_files:
            if not pdf_file.endswith("_done.pdf"):
                unprocessed.append(pdf_file)

        print(f"✅ File chưa xử lý: {[os.path.basename(f) for f in unprocessed]}")
        return unprocessed

    def process_page(self, args):
        """Xử lý một trang PDF"""
        i, pil_image = args

        # Core OCR logic - đơn giản và hiệu quả
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(binary_image, lang='vie')

        return i, text

    def ocr_pdf(self, pdf_path):
        """Thực hiện OCR trên file PDF"""
        print(f"\n📄 Đang xử lý: {os.path.basename(pdf_path)}")

        try:
            # Chuyển PDF sang hình ảnh
            print("   🔄 Chuyển đổi PDF sang hình ảnh...")
            images = convert_from_path(pdf_path, dpi=200, poppler_path=r'C:\poppler-24.08.0\Library\bin')
            print(f"   ✅ Chuyển đổi thành công {len(images)} trang")

            # Xử lý song song để tăng tốc
            print("   🔍 Đang thực hiện OCR...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                page_args = [(i, img) for i, img in enumerate(images)]
                results = list(executor.map(self.process_page, page_args))

            # Sắp xếp lại theo thứ tự trang
            results.sort(key=lambda x: x[0])

            # Ghép kết quả
            full_text = ""
            for i, text in results:
                full_text += f"--- TRANG {i + 1} ---\n"
                full_text += text + "\n\n"

            print(f"   ✅ OCR hoàn thành - {len(results)} trang")
            return full_text

        except Exception as e:
            print(f"   ❌ Lỗi OCR: {e}")
            return None

    def export_result(self, pdf_path, ocr_text):
        """Export kết quả OCR ra file text"""
        try:
            # Tạo tên file output
            pdf_name = Path(pdf_path).stem  # Lấy tên không có extension
            output_filename = f"{pdf_name}_ocr.txt"
            output_path = os.path.join(self.output_folder, output_filename)

            # Ghi file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"=== KẾT QUẢ OCR ===\n")
                f.write(f"File gốc: {os.path.basename(pdf_path)}\n")
                f.write(f"Ngày xử lý: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(ocr_text)

            print(f"   💾 Đã lưu: {output_filename}")
            return output_path

        except Exception as e:
            print(f"   ❌ Lỗi export: {e}")
            return None

    def mark_as_done(self, pdf_path):
        """Đổi tên file PDF thành _done"""
        try:
            # Tạo tên file mới với _done
            file_dir = os.path.dirname(pdf_path)
            file_name = Path(pdf_path).stem
            new_name = f"{file_name}_done.pdf"
            new_path = os.path.join(file_dir, new_name)

            # Đổi tên file
            os.rename(pdf_path, new_path)
            print(f"   ✅ Đã đổi tên: {os.path.basename(new_path)}")
            return new_path

        except Exception as e:
            print(f"   ❌ Lỗi đổi tên: {e}")
            return None

    def process_single_file(self, pdf_path):
        """Xử lý một file PDF hoàn chỉnh"""
        # 1. Thực hiện OCR
        ocr_text = self.ocr_pdf(pdf_path)
        if not ocr_text:
            return False

        # 2. Export kết quả
        output_path = self.export_result(pdf_path, ocr_text)
        if not output_path:
            return False

        # 3. Đánh dấu file đã xong
        done_path = self.mark_as_done(pdf_path)
        if not done_path:
            return False

        return True

    def process_all_files(self):
        """Xử lý tất cả file PDF chưa xong"""
        # Tìm file chưa xử lý OCR
        unprocessed_files = self.find_unprocessed_pdfs()

        if not unprocessed_files:
            print("🎉 Không có file PDF nào cần xử lý!")
            return

        print(f"📋 Tìm thấy {len(unprocessed_files)} file cần xử lý:")
        for file in unprocessed_files:
            print(f"   - {os.path.basename(file)}")

        print("\n🚀 Bắt đầu xử lý...")

        success_count = 0
        failed_files = []

        for pdf_file in unprocessed_files:
            success = self.process_single_file(pdf_file)
            if success:
                success_count += 1
            else:
                failed_files.append(pdf_file)

        # Báo cáo kết quả
        print(f"\n📊 KẾT QUẢ:")
        print(f"   ✅ Thành công: {success_count}/{len(unprocessed_files)}")

        if failed_files:
            print(f"   ❌ Thất bại: {len(failed_files)}")
            for failed in failed_files:
                print(f"      - {os.path.basename(failed)}")

        print(f"\n📁 Kết quả OCR được lưu trong thư mục: {self.output_folder}")


# Bắt đầu chạy
if __name__ == "__main__":
    from datetime import datetime

    # Khởi tạo class workflow
    ocr_workflow = OCRWorkflow(
        input_folder="../../data/raw/pdf",  # Thư mục chứa PDF cần xử lý
        output_folder="../../data/raw/ocr_output"  # Thư mục lưu kết quả
    )

    print("🔍 BẮT ĐẦU QUY TRÌNH OCR")
    print("=" * 40)

    # Xử lý tất cả file
    ocr_workflow.process_all_files()

    print("\n✨ HOÀN THÀNH!")


# Hoặc xử lý file đơn lẻ
def process_single_pdf(pdf_path):
    """Hàm tiện ích để xử lý 1 file"""
    from datetime import datetime

    workflow = OCRWorkflow()

    print(f"🔍 Xử lý file: {pdf_path}")

    # Kiểm tra file có tồn tại không
    if not os.path.exists(pdf_path):
        print(f"❌ File không tồn tại: {pdf_path}")
        return False

    # Kiểm tra file đã xử lý chưa
    if pdf_path.endswith("_done.pdf"):
        print(f"⚠️  File đã được xử lý rồi: {pdf_path}")
        return False

    # Xử lý
    success = workflow.process_single_file(pdf_path)

    if success:
        print(f"✅ Xử lý thành công!")
    else:
        print(f"❌ Xử lý thất bại!")

    return success

# Ví dụ sử dụng cho file đơn lẻ
# process_single_pdf("document.pdf")