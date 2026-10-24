# 🎮 Intelligent Gaming Monetization Engine

[![CI Pipeline](https://github.com/Cayocayo/gaming-monetization-engine/actions/workflows/main.yml/badge.svg)](https://github.com/Cayocayo/gaming-monetization-engine/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

End-to-End Artificial Intelligence Pipeline untuk optimalisasi monetisasi *mobile game* dan PC game. Sistem ini dirancang untuk menyelesaikan masalah ketimpangan pendapatan (*Highly Imbalanced Revenue*) di mana 97% pemain adalah pengguna gratis (F2P/Ad-Watchers) dan hanya 3% yang merupakan pembeli besar (*Whales*).

---

## 🏗️ Arsitektur Dual-Stage AI Pipeline

Proyek ini menggunakan dua tahap pemodelan *Machine Learning* yang berjalan secara paralel:

1. **Stage 1: LTV Prediction Engine (Deep Learning)**
   * **Tujuan:** Memprediksi *Lifetime Value* (LTV) pemain selama 180 hari ke depan hanya berdasarkan aktivitas 7 hari pertama.
   * **Model:** Deep Neural Network (TensorFlow/Keras) dengan arsitektur `Dense(64) -> Dropout(0.2) -> Dense(32) -> Dense(16)`.
   * **Performa:** MAE $5.21 (Sangat Presisi).

2. **Stage 2: Dynamic Content Recommender (NLP)**
   * **Tujuan:** Memberikan rekomendasi game terpersonalisasi untuk mengatasi *Cold-Start Problem* dan *Offer Fatigue*.
   * **Model:** Content-Based Filtering menggunakan ekstraksi teks `TF-IDF Vectorizer` dan pengukuran `Cosine Similarity`.
   * **Kinerja:** Akurasi pencocokan *niche* genre >80%.

---

## 🚀 Live Demo & Akses Aplikasi

Aplikasi dasbor interaktif telah di-deploy menggunakan Streamlit Cloud dan dapat diakses publik:
* 🌐 **Live Web App:** ` https://gaming-monetization-engine-pjnvgrlwwt7nkwkqy2fo67.streamlit.app/ `
* ⚙️ **API Docs (Swagger UI):** `[Link Streamlit Cloud]/docs` (Berjalan via FastAPI)

---

## 📖 Dokumentasi REST API

Sistem ini menyediakan layanan *backend* berbasis FastAPI (`api.py`) yang terintegrasi dengan pencatatan *database* SQLite (`database.py`). 

### 1. Endpoint: Prediksi LTV
* **URL:** `/api/v1/predict-ltv`
* **Method:** `POST`
* **Request Body (JSON):**
  ```json
  {
    "platform": "iOS",
    "session_duration": 45.5,
    "event_hour": 19,
    "revenue_usd": 2.99
  }

  Success Response (JSON):
  {
  "status": "success",
  "predicted_ltv_usd": 68.45,
  "player_tier": "Whale",
  "message": "Prediction logged to database successfully"
}

Cara Menjalankan Secara Lokal
Jika Anda ingin menjalankan proyek ini di mesin lokal (komputer sendiri), ikuti langkah-langkah berikut:

1. Clone Repository

Bash
git clone [https://github.com/Cayocayo/gaming-monetization-engine.git](https://github.com/Cayocayo/gaming-monetization-engine.git)
cd gaming-monetization-engine
2. Instalasi Dependensi
Disarankan menggunakan virtual environment.

Bash
pip install -r requirements.txt
3. Menjalankan Dashboard Streamlit (UI)

Bash
streamlit run app.py
Aplikasi web akan terbuka otomatis di browser pada http://localhost:8501.

4. Menjalankan Server FastAPI (Backend)
Buka terminal baru dan jalankan:

Bash
uvicorn api:app --reload
API Docs interaktif dapat diakses di http://localhost:8000/docs.

© 2026 Cahyo Widyonarko — Final Project Data Science & AI (dibimbing.id)

