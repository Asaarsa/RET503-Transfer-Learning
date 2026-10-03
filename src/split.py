"""
split.py
Membagi dataset menjadi train/val.

Untuk eksperimen yang lebih valid, pembagian sebaiknya mempertimbangkan
session_id/kondisi pengambilan agar frame yang sangat mirip tidak bocor
antara train dan validation.
"""

from pathlib import Path
import shutil


RAW = Path("dataset_raw")
OUT = Path("dataset")


def main():
    print("Template split.py")
    print(f"Dataset raw: {RAW.resolve()}")
    print(f"Output: {OUT.resolve()}")
    print("Lengkapi strategi split setelah metadata dataset tersedia.")


if __name__ == "__main__":
    main()
