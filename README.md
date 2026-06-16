# Analisis Sentimen Ulasan Spotify Indonesia Menggunakan Machine Learning

## Project Overview

Perkembangan layanan streaming musik digital telah membuat Spotify menjadi salah satu aplikasi yang paling banyak digunakan oleh masyarakat Indonesia. Pengguna dapat memberikan penilaian dan ulasan terhadap aplikasi melalui Google Play Store. Ulasan tersebut mengandung berbagai opini mengenai kualitas layanan, fitur, maupun pengalaman pengguna selama menggunakan aplikasi Spotify.

Jumlah ulasan yang sangat banyak menyebabkan proses analisis secara manual menjadi tidak efektif. Oleh karena itu, diperlukan teknik analisis sentimen untuk mengelompokkan ulasan pengguna ke dalam kategori sentimen positif, negatif, dan netral secara otomatis.

Pada proyek ini dilakukan pembangunan model klasifikasi sentimen menggunakan algoritma Logistic Regression dan Support Vector Machine (SVM) dengan memanfaatkan metode TF-IDF sebagai ekstraksi fitur. Model terbaik kemudian diimplementasikan dalam bentuk aplikasi berbasis Gradio dan di-deploy menggunakan Hugging Face Spaces.

Dataset yang digunakan adalah **Spotify Reviews - Indonesia (Google Play Store)** yang diperoleh dari Kaggle.

---

# Business Understanding

## Problem Statements

1. Bagaimana mengklasifikasikan ulasan pengguna Spotify Indonesia ke dalam kategori sentimen positif, negatif, dan netral secara otomatis?

2. Seberapa efektif algoritma Logistic Regression dibandingkan dengan Support Vector Machine (SVM) dalam melakukan klasifikasi sentimen pada ulasan Spotify?

3. Bagaimana membangun model yang mampu membantu mengidentifikasi tingkat kepuasan pengguna berdasarkan ulasan yang diberikan pada Google Play Store?

## Goals

1. Mengembangkan model analisis sentimen yang dapat mengelompokkan ulasan pengguna Spotify ke dalam kategori positif, negatif, dan netral.

2. Membandingkan performa algoritma Logistic Regression dan Support Vector Machine (SVM) untuk mengetahui metode yang memberikan hasil terbaik dalam klasifikasi sentimen.

3. Mengembangkan sistem yang dapat digunakan untuk memprediksi sentimen ulasan pengguna secara otomatis melalui proses deployment.

## Solution Approach

### Text Preprocessing

Melakukan tahapan pembersihan teks yang meliputi cleaning text, case folding, tokenization, dan stopword removal untuk menghasilkan data yang siap digunakan dalam proses pemodelan.

### TF-IDF Vectorization

Mengubah data teks menjadi representasi numerik menggunakan Term Frequency-Inverse Document Frequency (TF-IDF) sehingga dapat diproses oleh algoritma machine learning.

### Logistic Regression

Menggunakan algoritma Logistic Regression sebagai model baseline untuk melakukan klasifikasi sentimen ulasan pengguna Spotify.

### Support Vector Machine (SVM)

Menggunakan algoritma Support Vector Machine (SVM) dengan kernel linear untuk membandingkan performanya dengan Logistic Regression dalam tugas klasifikasi sentimen.

### Model Evaluation

Mengevaluasi performa kedua model menggunakan metrik accuracy, precision, recall, F1-score, serta confusion matrix untuk menentukan model terbaik.

---

# Data Understanding

Dataset yang digunakan pada proyek ini diperoleh dari Kaggle dengan nama **Spotify Reviews - Indonesia (Google Play Store)**. Dataset berisi ulasan pengguna aplikasi Spotify pada Google Play Store yang dapat digunakan untuk melakukan analisis sentimen.

Dataset terdiri dari satu tabel utama dengan total 100.000 data ulasan dan 6 fitur.

## Dataset Components

| Dataset         | Jumlah Data | Fitur                                                |
| --------------- | ----------- | ---------------------------------------------------- |
| Spotify Reviews | 100.000     | Nama User, Ulasan, Rating, Tanggal, Likes, Versi App |

## Uraian Fitur

* Nama User : Nama pengguna yang memberikan ulasan.
* Ulasan : Isi ulasan atau komentar pengguna terhadap aplikasi Spotify.
* Rating : Penilaian pengguna terhadap aplikasi dengan rentang 1 sampai 5.
* Tanggal : Waktu pengguna memberikan ulasan.
* Likes : Jumlah pengguna lain yang menyukai ulasan tersebut.
* Versi App : Versi aplikasi Spotify yang digunakan saat ulasan diberikan.

## Kondisi Data

### Missing Values

Ditemukan nilai yang hilang pada kolom Versi App sebanyak 27.702 data, sedangkan kolom lainnya tidak memiliki missing values.

### Duplikat Data

Hasil pemeriksaan menggunakan fungsi `duplicated()` menunjukkan bahwa tidak ditemukan data duplikat.

### Outlier

Kolom Rating memiliki rentang nilai 1–5 yang masih sesuai dengan skala penilaian Google Play Store. Sementara itu, kolom Likes memiliki beberapa nilai yang sangat tinggi dibandingkan sebagian besar data lainnya. Namun, karena penelitian berfokus pada analisis sentimen berbasis teks, nilai ekstrem pada kolom Likes tidak dilakukan penanganan lebih lanjut.

---

# Exploratory Data Analysis (EDA)

Exploratory Data Analysis dilakukan untuk memahami karakteristik dan pola data yang terdapat pada dataset ulasan Spotify Indonesia sebelum dilakukan proses pemodelan.

Analisis dilakukan terhadap:

* Distribusi rating pengguna.
* Distribusi sentimen positif, negatif, dan netral.
* Karakteristik panjang ulasan pengguna.
* Hubungan antar variabel pada dataset.

Hasil visualisasi menunjukkan bahwa mayoritas pengguna memberikan rating tinggi terhadap aplikasi Spotify, terutama pada rating 5. Hal ini menunjukkan bahwa sebagian besar pengguna memiliki pengalaman yang positif terhadap aplikasi Spotify.

---

# Data Preparation

Tahap data preparation dilakukan untuk menghasilkan data yang bersih dan siap digunakan pada proses pemodelan.

Tahapan yang dilakukan meliputi:

* Pemilihan fitur yang relevan.
* Penanganan missing values.
* Cleaning text dan case folding.
* Tokenization.
* Stopword removal.
* Pembentukan teks hasil preprocessing.
* Pembagian data menjadi data latih dan data uji.
* Ekstraksi fitur menggunakan TF-IDF.

Tahapan tersebut memastikan bahwa model machine learning memperoleh data yang bersih, konsisten, dan terstruktur dengan baik sehingga menghasilkan performa yang lebih optimal.

---

# Modeling and Results

## Logistic Regression

Logistic Regression digunakan sebagai model baseline karena memiliki performa yang baik pada permasalahan klasifikasi teks.

### Hasil Logistic Regression

Accuracy : **87.79%**

| Sentimen | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Negatif  | 0.75      | 0.76   | 0.75     |
| Netral   | 0.22      | 0.01   | 0.02     |
| Positif  | 0.91      | 0.96   | 0.94     |

### Keunggulan

* Proses training relatif cepat.
* Cocok digunakan sebagai model baseline.
* Mampu memberikan performa yang baik pada klasifikasi teks.

### Kelemahan

* Kurang mampu mengenali kelas minoritas (netral).
* Sensitif terhadap ketidakseimbangan data.

---

## Support Vector Machine (SVM)

### Hasil SVM

Accuracy : **87.34%**

| Sentimen | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Negatif  | 0.73      | 0.76   | 0.74     |
| Netral   | 0.20      | 0.02   | 0.04     |
| Positif  | 0.91      | 0.96   | 0.93     |

### Keunggulan

* Mampu bekerja dengan baik pada data berdimensi tinggi.
* Cocok digunakan pada klasifikasi teks berbasis TF-IDF.

### Kelemahan

* Waktu training lebih lama dibandingkan Logistic Regression.
* Performa sedikit lebih rendah pada dataset yang digunakan.

---

# Evaluation Model

Evaluasi dilakukan menggunakan Accuracy, Precision, Recall, dan F1-score.

## Perbandingan Model

| Model                        | Accuracy |
| ---------------------------- | -------- |
| Logistic Regression          | 87.79%   |
| Support Vector Machine (SVM) | 87.34%   |

Berdasarkan hasil evaluasi, model Logistic Regression memberikan performa terbaik sehingga dipilih sebagai model yang digunakan pada tahap deployment.

---

# Deployment

Model terbaik yaitu Logistic Regression diimplementasikan ke dalam aplikasi berbasis Gradio dan di-deploy menggunakan Hugging Face Spaces.

File deployment terdiri atas:

* `app.py`
* `requirements.txt`
* `model_logreg.pkl`
* `tfidf_vectorizer.pkl`

Aplikasi yang dibangun memungkinkan pengguna memasukkan ulasan Spotify dan memperoleh hasil prediksi sentimen secara otomatis.

---

# Conclusion

* Proses text preprocessing dan ekstraksi fitur menggunakan TF-IDF berhasil mengubah data teks menjadi representasi numerik yang dapat diproses oleh algoritma machine learning.

* Logistic Regression dan Support Vector Machine mampu melakukan klasifikasi sentimen dengan performa yang cukup baik.

* Logistic Regression menghasilkan accuracy sebesar 87.79%, sedikit lebih tinggi dibandingkan Support Vector Machine yang memperoleh accuracy sebesar 87.34%.

* Logistic Regression dipilih sebagai model terbaik pada penelitian ini dan digunakan pada tahap deployment.

* Model masih dapat dikembangkan lebih lanjut dengan teknik penyeimbangan data, hyperparameter tuning, maupun model berbasis transformer seperti IndoBERT untuk meningkatkan performa klasifikasi, khususnya pada kelas sentimen netral.
