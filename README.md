# Klasifikasi Baut vs Paku Menggunakan Transfer Learning ResNet-18

Proyek **RET503 Computer Vision and Deep Learning – Pertemuan 3** untuk klasifikasi citra dua kelas: **baut** dan **paku**.

## Status Dataset
Dataset awal dikumpulkan menggunakan webcam dengan latar putih dari kertas:
- baut: 15 citra
- paku: 15 citra
- total: 30 citra

Dataset ini digunakan sebagai **uji awal pipeline**. Target eksperimen final tetap minimal **50 citra per kelas** dengan variasi jarak, posisi, orientasi, pencahayaan, latar, occlusion, tumpukan, dan motion blur.

## Eksperimen
1. Feature Extraction
2. Partial Fine-Tuning
3. Training from Scratch

Model utama: **ResNet-18**.

## Struktur
```
RET503-Transfer-Learning/
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
│   ├── feature/
│   ├── partial/
│   └── scratch/
├── results/
└── docs/
    └── desain_pipeline_persepsi.md
```

## Konfigurasi Awal
| Mode | Bobot awal | Parameter dilatih | Learning rate |
|---|---|---|---|
| Feature Extraction | ImageNet | FC | 1e-3 |
| Partial Fine-Tuning | ImageNet | layer4 + FC | layer4 1e-4, FC 1e-3 |
| Scratch | Acak | Semua layer | 1e-3 |

Konfigurasi awal: 10 epoch, batch size 16, input 224×224, augmentasi training, dan Cosine Annealing.

## Menjalankan
Aktifkan virtual environment:
```bash
source .venv/bin/activate
```

Split dataset:
```bash
python src/split.py
```

Training:
```bash
python src/train.py --mode feature
python src/train.py --mode partial
python src/train.py --mode scratch
```

Latency:
```bash
python src/latency.py --model models/feature/best.pt
```

## Metrik
- training accuracy
- validation accuracy
- best validation accuracy dan epoch terbaik
- waktu training
- inference latency dan FPS

## Pencegahan Data Leakage
Jika tersedia, pembagian menggunakan `session_id` agar foto dari sesi pengambilan yang sama tidak tersebar sembarangan antara train dan validation.

Kesimpulan akhir ditulis setelah eksperimen dengan dataset final selesai.
