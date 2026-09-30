# 🎮 Intelligent Gaming Monetization Engine
> **Dual-Stage LTV Prediction & Dynamic Content Recommendation System**

Aplikasi web *End-to-End* berbasis kecerdasan buatan (*Data Science & Machine Learning*) yang dirancang untuk mengatasi **Revenue Leakage**, **Offer Fatigue**, dan **Cold-Start Problem** di industri game seluler dan PC.

---

## 📌 Latar Belakang & Masalah Bisnis
Dalam industri game modern, pendekatan pemasaran yang disamaratakan (*one-size-fits-all*) seringkali tidak efektif:
1. **Revenue Leakage:** Perusahaan membuang anggaran promosi karena gagal mendeteksi kelompok pemain elit (*Whales*) sejak dini.
2. **Offer Fatigue:** Pemain kasual atau *Free-to-Play* (F2P) merasa terganggu dengan pop-up promosi barang mahal yang berujung pada *churn*.
3. **Cold-Start Problem:** Publisher kesulitan merekomendasikan game baru secara tepat kepada pemain yang sesuai dengan selera mereka.

---

## ⚙️ Arsitektur Solusi (Multi-Stage Pipeline)
Proyek ini menggunakan dua tahapan model utama:
* **Stage 1 (Deep Learning - LTV Prediction):** Menggunakan *Feed-Forward Neural Network (TensorFlow/Keras)* untuk memprediksi nilai *Lifetime Value (180 Hari)* berdasarkan data aktivitas awal, guna membedakan pemain *Whales*, *Ad-Watchers*, dan *F2P Murni*.
* **Stage 2 (NLP Recommender System):** Memanfaatkan ekstraksi teks (*TF-IDF & Cosine Similarity*) dari katalog game Steam untuk memberikan rekomendasi konten yang hiper-terpersonalisasi.

---

## 🛠️ Tech Stack
* **Bahasa Pemrograman:** Python
* **Data Science & ML:** Pandas, NumPy, Scikit-Learn, TensorFlow / Keras, Joblib
* **Visualisasi & Dashboard:** Streamlit, Matplotlib, Seaborn
* **Deployment & Version Control:** Git, GitHub, Streamlit Cloud

---

## 📂 Struktur Direktori Proyek
```text
gaming-monetization-engine/
│
├── models/
│   ├── ltv_dl_model.keras        # Model utama Deep Learning LTV
│   ├── ltv_label_encoders.pkl    # Enkoder kategori data seluler
│   ├── ltv_scaler.pkl            # StandardScaler untuk normalisasi data
│   ├── steam_cosine_sim.pkl      # Matriks Cosine Similarity NLP Steam
│   └── steam_games_clean.pkl     # Dataset katalog game bersih
│
├── app.py                        # Script utama antarmuka Streamlit
├── requirements.txt              # Daftar pustaka (dependencies)
├── .gitignore                    # File/folder yang diabaikan Git
└── README.md                     # Dokumentasi proyek

Cara Menjalankan Secara Lokal (Local Installation)

1. Clone repositori ini:

Bash
git clone [https://github.com/Cayocayo/gaming-monetization-engine.git](https://github.com/Cayocayo/gaming-monetization-engine.git)
cd gaming-monetization-engine

2. Buat dan aktifkan Virtual Environment:

Bash
python3 -m venv env
source env/bin/activate

3. Instal seluruh pustaka yang dibutuhkan:

Bash
pip install --upgrade pip
pip install -r requirements.txt

4. Jalankan aplikasi Streamlit:

Bash
streamlit run app.py

👨‍💻 Author
Cahyo Widyonarko

Data Science & AI Bootcamp Student at dibimbing.id