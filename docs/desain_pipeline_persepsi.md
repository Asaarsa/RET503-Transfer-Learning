# Desain Pipeline Persepsi — Klasifikasi Baut vs Mur

## 1. Misi Proyek
Sistem visi digunakan untuk mengenali apakah objek yang terlihat kamera termasuk **baut** atau **mur**. Hasil klasifikasi dapat menjadi komponen persepsi untuk sistem robot/otomasi yang membutuhkan identifikasi objek.

## 2. Kelas Objek
| Kelas | Deskripsi | Target awal |
|---|---|---:|
| baut | Komponen baut | ≥50 citra |
| mur | Komponen mur | ≥50 citra |

## 3. Rencana Pengambilan Data
- jarak dekat, sedang, dan jauh
- objek di tengah dan tepi frame
- orientasi tegak, miring, dan terbalik
- cahaya terang dan redup
- jendela/bayangan
- latar berbeda dan objek pengganggu
- sebagian objek tertutup
- tumpukan
- motion blur

Setiap foto dicatat di `dataset_raw/metadata.csv`.

## 4. Pencegahan Data Leakage
Foto dari sesi pengambilan yang sama sebaiknya dikelompokkan melalui `session_id`. Script split akan mencoba menggunakan kelompok sesi terlebih dahulu. Jika metadata sesi belum tersedia, digunakan random split sederhana berdasarkan kelas.

## 5. Model ResNet-18
### Feature Extraction
- pretrained ImageNet
- backbone dibekukan
- hanya FC dilatih

### Partial Fine-Tuning
- pretrained ImageNet
- layer4 dan FC dilatih
- learning rate layer4 = 1e-4
- learning rate FC = 1e-3

### Training from Scratch
- bobot awal acak
- semua layer dilatih
- learning rate = 1e-3

## 6. Training
- epoch: 10
- batch size: 16
- input: 224 × 224
- augmentasi: horizontal flip, rotation, brightness/contrast
- scheduler: Cosine Annealing

## 7. Metrik
- training accuracy
- validation accuracy
- best validation accuracy
- epoch terbaik
- total waktu training
- inference latency
- FPS

## 8. Risiko dan Mitigasi
| Risiko | Dampak | Mitigasi |
|---|---|---|
| Data leakage | Validation terlalu optimistis | Split berdasarkan sesi |
| Dataset terlalu sedikit | Overfitting | Augmentasi dan transfer learning |
| Kondisi cahaya berubah | Akurasi dapat turun | Tambahkan variasi cahaya |
| Background terlalu seragam | Model belajar background | Variasikan latar |
| Occlusion/tumpukan | Klasifikasi lebih sulit | Tambahkan contoh occlusion |
| Motion blur | Fitur kurang jelas | Tambahkan contoh blur |