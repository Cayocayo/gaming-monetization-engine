# 🎮 Intelligent Gaming Monetization Engine

**Final Project Data Science & AI — dibimbing.id**  
**Oleh:** Cahyo Widyonarko

Sebuah sistem kecerdasan buatan terintegrasi (*Dual-Stage Pipeline*) yang dirancang untuk mengatasi *Revenue Leakage* dan *Offer Fatigue* di industri *gaming* melalui prediksi nilai pemain (LTV) dan rekomendasi konten yang dipersonalisasi.

## 📌 Problem Statement
Distribusi pemain *game* sangat timpang (*highly right-skewed*), di mana >84% pemain adalah *Free-to-Play* ($0 revenue) dan hanya kelompok kecil elit (*Whales*) yang menyumbang mayoritas pendapatan. Keterlambatan mendeteksi *Whales* dan pemberian promosi yang tidak relevan sering kali memicu *churn*.

## 🏗️ Arsitektur AI (Dual-Stage Pipeline)
Proyek ini memadukan dua pendekatan *Machine Learning*:

1. **Stage 1: Player LTV Prediction (Deep Learning)**
   - **Fungsi:** Memprediksi *Lifetime Value* (LTV) 180 hari ke depan berdasarkan perilaku 7 hari pertama (platform, durasi sesi, frekuensi main, dll).
   - **Model:** *Sequential Deep Neural Network (DNN)* menggunakan TensorFlow/Keras.
   - **Evaluasi:** Baseline MAE $12.40. Setelah *hyperparameter tuning* (Dropout 0.2, Adam LR 0.001), model mencapai **MAE: $5.21** (Telah melewati *business threshold* < $10.00).

2. **Stage 2: Dynamic Content Recommender (NLP)**
   - **Fungsi:** Mengatasi *Cold-Start Problem* dengan mencocokkan profil pemain dengan 15.000+ katalog game di Steam. Fitur *multi-select* menyatukan vektor *Cosine Similarity* dari beberapa game sekaligus (agregasi *Mean Vector*).
   - **Model:** *Content-Based Filtering* berbasis TF-IDF (*Genres & Tags*).
   - **Evaluasi:** Mampu memfilter *self-match* (mengecualikan game yang sudah dimainkan) dan memberikan Top 5 rekomendasi dengan skor kemiripan profil rata-rata **> 80%**.

## 🛠️ Tech Stack & Deployment
- **Data Science & ML:** Python, Pandas, NumPy, Scikit-Learn, TensorFlow/Keras.
- **Web App (Frontend):** Streamlit
- **Backend API:** FastAPI (Inferensi asli menggunakan `ltv_dl_model.keras`)
- **MLOps & CI/CD:** GitHub Actions (Flake8 Linter & Pytest)

## 📁 Struktur Direktori Utama
```bash
gaming-monetization-engine/
├── assets/                  # Penyimpanan gambar (grafik EDA & QR Code)
├── models/                  # Artifacts model produksi (.keras & .pkl)
├── api.py                   # Endpoint FastAPI untuk LTV Prediction
├── app.py                   # UI Streamlit Terintegrasi (Dashboard & AI Engine)
├── test_api.py              # Unit testing untuk endpoint FastAPI
├── test_app.py              # Unit testing struktural
├── requirements.txt         # Dependensi library Python
└── README.md                # Dokumentasi Proyek

🚀 Instalasi & Cara Penggunaan
1. Clone Repository

Bash
git clone [https://github.com/Cayocayo/gaming-monetization-engine.git](https://github.com/Cayocayo/gaming-monetization-engine.git)
cd gaming-monetization-engine
2. Instalasi Dependensi

Bash
pip install -r requirements.txt
3. Menjalankan Dashboard Streamlit (UI)

Bash
streamlit run app.py
Akses melalui browser di http://localhost:8501.

4. Menjalankan Backend API (FastAPI)

Bash
uvicorn api:app --reload
Akses Swagger UI Documentation di http://localhost:8000/docs.

5. Menjalankan Unit Testing (CI/CD Local)
Untuk memastikan integritas fungsi API dan Dataframe sebelum push:

Bash
pytest
🌐 Live Access
Aplikasi telah di-deploy dan dapat diakses secara langsung melalui Streamlit Cloud:Buka Live Dashboard Engine


Setelah memperbarui *file* `README.md` ini, jangan lupa untuk melakukan *commit* (`git add README.md`, `git commit -m "docs: update README with real inference and testing details"`, `git push origin main`). 