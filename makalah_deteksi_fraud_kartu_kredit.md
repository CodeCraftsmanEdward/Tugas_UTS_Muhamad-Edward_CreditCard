# Deteksi Penipuan Transaksi Kartu Kredit Menggunakan Logistic Regression dan Isolation Forest

**Nama:** [Muhamad Edward]  
**NIM:** [241011400875]  
**Program Studi:** [Tenik Informatika]  
**Mata Kuliah:** Machine Learning  
**Institusi:** [Universitas Pamulang]  
**Tahun:** 2026

## Abstrak

Penipuan transaksi kartu kredit merupakan masalah klasifikasi yang menantang karena jumlah transaksi fraud jauh lebih sedikit daripada transaksi normal. Makalah ini membahas analisis dataset transaksi kartu kredit dengan 284.807 baris dan 31 kolom, termasuk label `Class` sebagai penanda transaksi normal atau fraud. Hanya 492 transaksi (0,173%) yang berlabel fraud. Program membandingkan Logistic Regression dengan bobot kelas seimbang sebagai pendekatan klasifikasi terawasi dan Isolation Forest sebagai pendekatan deteksi anomali. Data dibagi menjadi data latih dan data uji dengan rasio 80:20 serta stratifikasi kelas. Pada data uji yang berisi 98 transaksi fraud, Logistic Regression memperoleh recall fraud 0,9184, precision 0,0610, F1-score 0,1144, ROC-AUC 0,9721, dan PR-AUC 0,7190. Isolation Forest memperoleh precision 0,1923, recall 0,2551, dan F1-score 0,2193. Hasil menunjukkan adanya kompromi antara kemampuan menemukan fraud dan jumlah alarm palsu. Oleh karena itu, evaluasi tidak cukup hanya menggunakan akurasi; pemilihan model dan ambang klasifikasi perlu mempertimbangkan biaya fraud yang terlewat dan biaya pemeriksaan transaksi yang salah ditandai.

**Kata kunci:** deteksi fraud, transaksi kartu kredit, imbalanced data, Logistic Regression, Isolation Forest, precision-recall

## 1. Pendahuluan

### 1.1 Latar Belakang

Transaksi kartu kredit dalam jumlah besar perlu dipantau untuk menemukan aktivitas yang berpotensi sebagai penipuan. Sistem deteksi harus mengidentifikasi transaksi fraud tanpa terlalu sering menandai transaksi normal. Tantangannya, pola fraud tidak selalu sama dengan pola transaksi normal dan jumlah kasus fraud biasanya sangat kecil dibandingkan seluruh transaksi.

Ketimpangan kelas dapat membuat akurasi memberikan gambaran yang keliru. Jika hampir semua transaksi normal, model yang selalu memprediksi normal dapat memperoleh akurasi tinggi tetapi sama sekali tidak menemukan fraud. Karena itu, evaluasi perlu memperhatikan precision, recall, F1-score, serta kurva Precision-Recall.

### 1.2 Rumusan Masalah

1. Bagaimana mempersiapkan dataset transaksi kartu kredit yang memiliki ketimpangan kelas?
2. Bagaimana kinerja Logistic Regression dan Isolation Forest dalam mendeteksi transaksi fraud?
3. Mengapa metrik selain akurasi perlu digunakan pada kasus ini?

### 1.3 Tujuan

1. Menganalisis distribusi kelas pada dataset transaksi kartu kredit.
2. Membangun model klasifikasi dan deteksi anomali untuk mengidentifikasi transaksi fraud.
3. Membandingkan model menggunakan metrik yang relevan untuk kelas minoritas.

## 2. Tinjauan Pustaka

### 2.1 Imbalanced Data

Imbalanced data adalah kondisi ketika jumlah observasi antar kelas berbeda secara signifikan. Dalam penelitian ini, kelas normal merupakan kelas mayoritas sedangkan fraud merupakan kelas minoritas. Model dapat tampak memiliki akurasi tinggi dengan mengutamakan kelas mayoritas, tetapi tetap gagal mendeteksi fraud.

### 2.2 Logistic Regression

Logistic Regression merupakan model klasifikasi terawasi yang memperkirakan probabilitas suatu observasi termasuk dalam kelas tertentu. Implementasi menggunakan `class_weight="balanced"` untuk memberi bobot lebih besar pada kesalahan kelas yang jumlahnya sedikit. StandardScaler ditempatkan dalam pipeline bersama model agar fitur distandardisasi ketika proses pelatihan dan prediksi berlangsung.

### 2.3 Isolation Forest

Isolation Forest adalah algoritma deteksi anomali yang memisahkan observasi melalui pemilihan fitur dan titik pemisah secara acak. Anomali cenderung terisolasi dengan lebih sedikit pemisahan. Pada program ini, model dilatih menggunakan sampel transaksi normal dari data latih; prediksi anomali kemudian dipetakan sebagai kandidat fraud.

### 2.4 Metrik Evaluasi

- **Precision** mengukur proporsi prediksi fraud yang benar-benar fraud: $TP / (TP + FP)$.
- **Recall** mengukur proporsi seluruh fraud yang berhasil ditemukan: $TP / (TP + FN)$.
- **F1-score** adalah rata-rata harmonik precision dan recall: $2PR / (P + R)$.
- **ROC-AUC** mengukur kemampuan model memeringkat kelas pada berbagai ambang menggunakan kurva ROC.
- **PR-AUC / Average Precision** merangkum hubungan precision dan recall; metrik ini informatif saat kelas positif sangat langka.
- **Confusion matrix** menampilkan jumlah prediksi benar dan salah untuk setiap kelas.

Keterangan: TP adalah fraud yang terdeteksi benar, FP adalah transaksi normal yang salah ditandai sebagai fraud, dan FN adalah fraud yang terlewat.

## 3. Metodologi

### 3.1 Dataset

Dataset dibaca dari berkas lokal `creditcard.csv`. Dataset memiliki 284.807 transaksi dan 31 kolom: 30 fitur numerik (`Time`, `V1` sampai `V28`, dan `Amount`) serta kolom target `Class`. Nilai `Class = 0` menunjukkan transaksi normal, sedangkan `Class = 1` menunjukkan transaksi fraud. Pemeriksaan pada program menemukan tidak ada nilai kosong pada kolom-kolom dataset.

Distribusi kelas yang diamati:

| Kelas | Jumlah transaksi | Persentase |
|---|---:|---:|
| Normal | 284.315 | 99,827% |
| Fraud | 492 | 0,173% |
| **Total** | **284.807** | **100%** |

### 3.2 Tahapan Pengerjaan Program

1. **Mengimpor pustaka dan memuat data.** Program menggunakan pandas untuk membaca CSV, seaborn dan matplotlib untuk visualisasi, serta scikit-learn untuk pembagian data, pelatihan model, dan evaluasi.
2. **Memeriksa ukuran dan kolom.** Program memeriksa jumlah baris, jumlah fitur, dan nama kolom untuk memastikan berkas sesuai dengan analisis.
3. **Memeriksa struktur dan statistik.** `data.info()` menampilkan tipe data dan nilai terisi. `data.describe()` merangkum distribusi numerik.
4. **Memeriksa nilai kosong.** Program menghitung nilai null. Tidak ditemukan nilai kosong pada dataset yang diuji, sehingga imputasi tidak diperlukan.
5. **Mengukur ketimpangan kelas.** Jumlah dan persentase `Class` dihitung. Fraud hanya 0,173% dari transaksi.
6. **Memvisualisasikan kelas.** Diagram batang menampilkan jumlah transaksi normal dibandingkan fraud.
7. **Membagi fitur dan target.** Kolom `Class` dipisahkan menjadi target `y`, sementara kolom lain menjadi fitur `X`.
8. **Membagi data latih dan data uji.** Data dibagi 80:20 dengan `stratify=y`, sehingga proporsi kelas tetap terjaga. Random state ditetapkan agar pembagian dapat direproduksi.
9. **Membuat baseline.** `DummyClassifier` selalu memilih kelas mayoritas. Model ini digunakan untuk menunjukkan bahwa akurasi saja tidak cukup.
10. **Melatih Logistic Regression.** Pipeline menerapkan StandardScaler dan Logistic Regression dengan bobot kelas seimbang. Model dilatih hanya menggunakan data latih.
11. **Mengevaluasi Logistic Regression.** Program menghitung classification report, ROC-AUC, dan PR-AUC dari data uji.
12. **Melihat confusion matrix.** Matriks digunakan untuk melihat fraud yang ditemukan, fraud yang terlewat, dan transaksi normal yang salah ditandai.
13. **Membuat kurva Precision-Recall.** Kurva memperlihatkan kompromi precision dan recall untuk berbagai ambang probabilitas.
14. **Melatih Isolation Forest.** Model dilatih dengan sampel hingga 50.000 transaksi normal dari data latih. Data uji tidak digunakan dalam pelatihan.
15. **Mengevaluasi Isolation Forest.** Label anomali `-1` dikonversi menjadi prediksi fraud, lalu dibandingkan dengan target data uji.
16. **Membandingkan model.** Precision, recall, F1-score untuk fraud dan PR-AUC Logistic Regression dirangkum dalam tabel.

### 3.3 Pembagian Data

Dengan rasio uji 20% dan stratifikasi kelas, data latih berisi 227.845 transaksi dan data uji berisi 56.962 transaksi. Data uji memiliki 98 transaksi fraud. Pemisahan dilakukan sebelum pelatihan model sehingga evaluasi dilakukan pada data yang tidak dipakai untuk melatih model.

## 4. Hasil dan Pembahasan

### 4.1 Hasil Baseline

Baseline yang selalu memprediksi transaksi normal menghasilkan akurasi yang dibulatkan menjadi 1,00, tetapi precision, recall, dan F1-score untuk fraud bernilai 0. Hasil ini membuktikan bahwa akurasi tinggi tidak berarti sistem mampu mendeteksi fraud.

### 4.2 Hasil Model

| Model | Precision fraud | Recall fraud | F1-score fraud | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,0610 | 0,9184 | 0,1144 | 0,9721 | 0,7190 |
| Isolation Forest | 0,1923 | 0,2551 | 0,2193 | Tidak dihitung | Tidak dihitung |

Isolation Forest pada implementasi ini menghasilkan label anomali, bukan skor probabilitas, sehingga program tidak menghitung ROC-AUC atau PR-AUC untuk model tersebut.

### 4.3 Pembahasan

Logistic Regression menemukan sekitar 91,8% transaksi fraud pada data uji. Namun, precision 0,0610 berarti hanya sekitar 6 dari setiap 100 transaksi yang ditandai sebagai fraud benar-benar fraud. Model ini cocok bila prioritas utama adalah mengurangi fraud yang terlewat dan proses pemeriksaan lanjutan mampu menangani banyak alarm palsu.

Isolation Forest menghasilkan precision 0,1923 dan recall 0,2551. Proporsi alarm yang benar lebih besar daripada Logistic Regression, tetapi model ini melewatkan sebagian besar transaksi fraud. Pada konfigurasi dan pembagian data yang diuji, F1-score Isolation Forest lebih tinggi, sedangkan Logistic Regression jauh lebih tinggi pada recall dan PR-AUC. Pilihan model bergantung pada biaya false positive dan false negative; hasil ini tidak menetapkan satu model sebagai terbaik untuk semua kebutuhan.

Penggunaan bobot kelas membantu Logistic Regression memperhatikan fraud, tetapi ambang default model tidak harus menjadi ambang terbaik untuk operasional. Ambang dapat dipilih dari kurva Precision-Recall berdasarkan kapasitas pemeriksaan dan toleransi terhadap fraud yang lolos. Evaluasi lanjutan sebaiknya menetapkan ambang pada data validasi, bukan memilihnya berdasarkan data uji.

Hasil Isolation Forest juga perlu ditafsirkan sesuai rancangan eksperimen: model dipasang pada sampel normal, tetapi nilai `contamination` ditentukan dari proporsi fraud di data latih. Dalam penggunaan nyata, proporsi fraud mungkin tidak diketahui atau berubah, sehingga parameter ini perlu ditentukan melalui validasi dan kebutuhan operasional.

## 5. Kesimpulan dan Saran

### 5.1 Kesimpulan

1. Dataset memiliki ketimpangan kelas yang kuat: hanya 492 dari 284.807 transaksi berlabel fraud.
2. Baseline kelas mayoritas menunjukkan akurasi tinggi tetapi recall fraud nol, sehingga akurasi tidak memadai sebagai ukuran tunggal.
3. Pada pengujian ini, Logistic Regression memiliki recall fraud 0,9184 dan PR-AUC 0,7190, tetapi precision fraud hanya 0,0610.
4. Isolation Forest memiliki precision fraud 0,1923 dan recall 0,2551 pada konfigurasi yang digunakan.
5. Pemilihan model perlu mempertimbangkan keseimbangan deteksi fraud dan jumlah alarm palsu serta diuji kembali pada data validasi.

### 5.2 Saran Pengembangan

1. Menentukan ambang prediksi Logistic Regression menggunakan data validasi dan biaya operasional false positive/false negative.
2. Mencoba model tambahan seperti Random Forest atau Gradient Boosting dengan pembagian data dan metrik evaluasi yang sama.
3. Menguji resampling hanya pada data latih, bukan sebelum pemisahan train/test, untuk menghindari kebocoran data.
4. Menggunakan validasi silang stratified dan pengujian pada periode waktu berbeda untuk mengukur kestabilan model.
5. Menambahkan analisis biaya, kalibrasi probabilitas, serta pemantauan perubahan pola transaksi jika model akan digunakan secara operasional.

## Daftar Pustaka

Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., dan Bontempi, G. (2018). Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy. *IEEE Transactions on Neural Networks and Learning Systems*, 29(8), 3784-3797.

Liu, F. T., Ting, K. M., dan Zhou, Z.-H. (2008). Isolation Forest. *Proceedings of the 2008 Eighth IEEE International Conference on Data Mining*, 413-422.

Pedregosa, F., dkk. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

## Catatan Penyuntingan

Lengkapi identitas penulis dan institusi pada halaman awal. Sesuaikan format judul, sitasi, margin, serta tata letak dengan pedoman makalah dari dosen atau institusi. Angka hasil dalam draf ini berasal dari eksekusi notebook saat disusun dan dapat berubah jika random state, data, atau konfigurasi model diubah.
