# Desain Awal Pipeline Persepsi

## 1. Misi Proyek

Tuliskan peran persepsi/vision dalam proyek robot dalam 2–3 kalimat.

## 2. Kelas Objek

| Kelas | Contoh | Jumlah citra |
|---|---|---:|
| Kelas 1 | - | - |
| Kelas 2 | - | - |

Target awal sesuai praktikum: minimal 50 citra per kelas.

## 3. Kamera dan Dudukan

- Resolusi:
- Tinggi kamera:
- Sudut kamera:
- Jarak kerja:
- Metode kalibrasi/undistort:

## 4. Unit Komputasi

- Perangkat:
- CPU/GPU:
- RAM:
- Mode daya:

## 5. Target Kinerja

- Target akurasi:
- Target FPS:
- Target latency:
- Target latency ROS2 (jika digunakan):

Materi memberikan contoh anggaran 15 FPS atau sekitar 67 ms per frame untuk seluruh pipeline.

## 6. Kandidat Model

### ResNet-18
Alasan pemilihan:

### MobileNetV3-Small
Alasan pemilihan:

## 7. Strategi Transfer Learning

Strategi awal:

- [ ] Feature Extraction
- [ ] Partial Fine-Tuning
- [ ] Fine-Tuning penuh

Alasan:

## 8. Rencana Data

Jelaskan variasi:

- jarak
- posisi objek
- orientasi
- pencahayaan
- latar
- occlusion
- motion blur

## 9. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Data leakage | Validasi terlalu tinggi | Split berdasarkan sesi/kondisi |
| Dataset terlalu sedikit | Overfitting | Augmentasi + transfer learning |
| Domain terlalu berbeda | Negative transfer | Fine-tuning lebih banyak layer |
