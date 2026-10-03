"""
capture.py
Template pengambilan citra untuk dataset RET503.

Gunakan kamera proyek sendiri dan simpan citra ke dataset_raw/<kelas>/.
Metadata pengambilan dicatat di dataset_raw/metadata.csv.

Sesuaikan bagian capture dengan kamera yang digunakan pada proyek.
"""

from pathlib import Path
import cv2


def main():
    output_dir = Path("dataset_raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Template capture.py")
    print("Tambahkan konfigurasi kamera dan kelas proyek Anda.")
    print(f"Folder output: {output_dir.resolve()}")

    # Contoh pembuka kamera USB:
    # cap = cv2.VideoCapture(0)
    # ... ambil frame, simpan PNG/JPG, dan catat metadata ...


if __name__ == "__main__":
    main()
