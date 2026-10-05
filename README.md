# Klasifikasi Baut vs Mur Menggunakan Transfer Learning ResNet-18

Proyek **RET503 Computer Vision and Deep Learning – Pertemuan 3** untuk klasifikasi citra dua kelas: **baut** dan **mur**.

## Eksperimen
1. Feature Extraction
2. Partial Fine-Tuning
3. Training from Scratch

Model utama: **ResNet-18**.

## Dataset
Target awal: minimal **50 citra per kelas**. Foto dikumpulkan sendiri dengan variasi jarak, posisi, orientasi, pencahayaan, latar, occlusion, tumpukan, dan motion blur.

Metadata dicatat di `dataset_raw/metadata.csv`.

## Struktur
````
RET503-Transfer-Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset_raw/
│   ├── metadata.csv
│   ├── baut/
│   └── paku/
├── dataset/
│   ├── train/
│   │   ├── baut/
│   │   └── paku/
│   └── val/
│       ├── baut/
│       └── paku/
├── src/
│   ├── capture.py
│   ├── split.py
│   ├── train.py
│   └── latency.py
├── models/
│   ├── feature_extraction/
│   ├── partial_finetuning/
│   └── scratch/
├── results/
└── docs/
    └── desain_pipeline_persepsi.md
````

## Konfigurasi Awal
| Mode | Bobot awal | Parameter dilatih | Learning rate |
|---|---|---|---|
| Feature Extraction | ImageNet | FC | 1e-3 |
| Partial Fine-Tuning | ImageNet | layer4 + FC | layer4 1e-4, FC 1e-3 |
| Scratch | Acak | Semua layer | 1e-3 |

10 epoch, augmentasi training, dan Cosine Annealing digunakan sebagai konfigurasi awal.

## Menjalankan
1. Install dependency:
`bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
`
2. Masukkan foto ke `dataset_raw/baut/` dan `dataset_raw/mur/`.
3. Lengkapi `dataset_raw/metadata.csv`.
4. Split:
`bash
python src/split.py
`
5. Training:
`bash
python src/train.py --mode feature
python src/train.py --mode partial
python src/train.py --mode scratch
`
6. Latency:
`bash
python src/latency.py --model models/feature_extraction/best.pt
`

## Metrik
- training accuracy
- validation accuracy
- best validation accuracy dan epoch terbaik
- waktu training
- latency inference dan FPS

## Pencegahan Data Leakage
Jika tersedia, pembagian menggunakan `session_id` agar foto dari sesi pengambilan yang sama tidak tersebar secara sembarangan antara train dan validation.

Kesimpulan akhir akan ditulis setelah eksperimen benar-benar dijalankan.
