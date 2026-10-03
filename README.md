# RET503 - Transfer Learning

Project for **RET503 Computer Vision and Deep Learning – Pertemuan 3**.

## Tujuan

Menerapkan **transfer learning** pada tugas klasifikasi citra menggunakan model pretrained, kemudian membandingkan:

1. Feature Extraction
2. Partial Fine-Tuning
3. Training from Scratch

Materi praktikum menggunakan **ResNet-18** sebagai model utama dan mengarahkan pengukuran akurasi validasi, waktu pelatihan, epoch pencapaian target akurasi, serta latency model.

## Struktur Repository

```
RET503-Transfer-Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset_raw/
│   ├── metadata.csv
│   ├── baut/
│   └── mur/
├── dataset/
│   ├── train/
│   └── val/
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
│   ├── accuracy.csv
│   ├── training_time.csv
│   ├── latency.csv
│   └── graphs/
└── docs/
    └── desain_pipeline_persepsi.md
```

## Rencana Eksperimen

| Mode | Bobot awal | Parameter dilatih | Learning rate |
|---|---|---|---|
| Feature Extraction | ImageNet | FC saja | 1e-3 |
| Partial Fine-Tuning | ImageNet | layer4 + FC | 1e-4 / 1e-3 |
| Scratch | Acak | Semua layer | 1e-3 |

Eksperimen awal dirancang untuk **10 epoch** dengan augmentasi pada data training dan scheduler Cosine Annealing.

## Dataset

Dataset akan dikumpulkan dari kamera robot/lingkungan operasi proyek. Kondisi pengambilan perlu mencakup variasi:

- jarak dekat, sedang, dan jauh
- posisi objek di tengah dan tepi citra
- orientasi tegak, miring, dan terbalik
- kondisi cahaya terang/redup, jendela, dan bayangan
- latar dan objek pengganggu
- kondisi sulit seperti occlusion, tumpukan, dan motion blur

Target awal: **minimal 50 citra per kelas**.

Metadata dicatat di `dataset_raw/metadata.csv`.

> Jangan memasukkan data pribadi atau data yang tidak memiliki izin penggunaan ke repository publik.

## Pencegahan Data Leakage

Frame berurutan dari sesi pengambilan yang sama tidak boleh tersebar secara sembarangan ke train dan validation. Pembagian data sebaiknya mempertimbangkan **sesi atau kondisi pengambilan**.

## Cara Menjalankan

### 1. Install dependency

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Siapkan dataset

Letakkan citra asli di `dataset_raw/` dan lengkapi `metadata.csv`.

### 3. Split dataset

```bash
python src/split.py
```

### 4. Training

```bash
python src/train.py --mode feature
python src/train.py --mode partial
python src/train.py --mode scratch
```

### 5. Ukur latency

```bash
python src/latency.py
```

## Hasil

Hasil eksperimen akan dicatat pada folder `results/`, termasuk:

- akurasi validasi
- waktu training
- epoch pencapaian target
- latency/inference time
- grafik akurasi per epoch

## Catatan

Repository ini merupakan kerangka awal proyek. Dataset, hasil eksperimen, grafik, dan kesimpulan akan ditambahkan setelah praktikum dan pengambilan data dilakukan.
