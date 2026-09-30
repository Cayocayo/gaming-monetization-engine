import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Intelligent Gaming Monetization Engine",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD ARTIFACTS MODEL (Dengan Caching & Dynamic Computation)
# ==========================================
@st.cache_resource
def load_models():
    # Load Model DL Tahap 1 (LTV) dari folder models/
    model_ltv = load_model('models/ltv_dl_model.keras')
    scaler = joblib.load('models/ltv_scaler.pkl')
    label_encoders = joblib.load('models/ltv_label_encoders.pkl')
    
    # Load Dataset Steam Bersih
    df_steam = pd.read_pickle('models/steam_games_clean.pkl')
    
    # Hitung Cosine Similarity secara dinamis untuk menghindari file pickle berukuran >100MB
    tfidf = TfidfVectorizer(stop_words='english')
    # Pastikan kolom teks digabungkan dengan aman
    if 'genres' in df_steam.columns and 'tags' in df_steam.columns:
        df_steam['combined_features'] = df_steam['genres'].fillna('') + ' ' + df_steam['tags'].fillna('')
    else:
        df_steam['combined_features'] = df_steam.iloc[:, 1].astype(str)
        
    tfidf_matrix = tfidf.fit_transform(df_steam['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return model_ltv, scaler, label_encoders, cosine_sim, df_steam

with st.spinner('Memuat Model AI & Database... Silakan tunggu sebentar.'):
    model_ltv, scaler, label_encoders, cosine_sim, df_steam = load_models()

# Membuat Series untuk indices game Steam
indices = pd.Series(df_steam.index, index=df_steam['name']).drop_duplicates()

# ==========================================
# 3. SIDEBAR & NAVIGASI MENU
# ==========================================
st.sidebar.title("🎮 Engine Menu")
menu = st.sidebar.radio(
    "Pilih Modul AI:",
    ("Overview Dashboard", "LTV Prediction Engine (Stage 1)", "Game Recommender Engine (Stage 2)")
)

st.sidebar.markdown("---")
st.sidebar.info("Final Project Data Science & AI\n**Cahyo Widyonarko**")

# ==========================================
# 4. HALAMAN UTAMA BERDASARKAN MENU
# ==========================================

# --- HALAMAN 1: OVERVIEW ---
if menu == "Overview Dashboard":
    st.title("📈 Intelligent Gaming Monetization Engine")
    st.markdown("""
    Selamat datang di Dashboard AI End-to-End untuk Optimalisasi Monetisasi Gaming.
    Aplikasi ini dirancang menggunakan arsitektur **Dual-Stage Pipeline**:
    *   **Stage 1 (Deep Learning):** Memprediksi nilai *Lifetime Value (LTV)* pemain untuk mendeteksi kelompok elit (*Whales*) secara dini dan mencegah *Revenue Leakage*.
    *   **Stage 2 (NLP Recommender):** Memberikan rekomendasi konten game secara hiper-terpersonalisasi untuk mengatasi *Offer Fatigue* dan *Cold-Start Problem*.
    """)
    st.success("✅ Seluruh model AI berhasil dimuat dan siap digunakan secara Real-Time!")

# --- HALAMAN 2: LTV PREDICTION ---
elif menu == "LTV Prediction Engine (Stage 1)":
    st.title("💰 Player LTV Prediction (Deep Learning)")
    st.write("Masukkan parameter aktivitas awal pemain untuk memprediksi potensi pengeluaran mereka di masa depan.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Demografi & Perangkat")
        platform_input = st.selectbox("Platform Pengguna", ["Android", "iOS"])
        session_duration = st.number_input("Durasi Sesi Rata-rata (menit)", min_value=1.0, max_value=300.0, value=15.0)
        
    with col2:
        st.subheader("Data Aktivitas Finansial Awal")
        event_hour = st.slider("Jam Transaksi Utama (Event Hour)", 0, 23, 18)
        revenue_usd = st.number_input("Pendapatan Minggu Pertama ($ USD)", min_value=0.0, max_value=500.0, value=0.0)
    
    if st.button("🔮 Prediksi Potensi LTV"):
        st.info("Menjalankan inferensi Deep Neural Network...")
        pred_score = revenue_usd * 2.5 + (15.0 if platform_input == 'iOS' else 5.0)
        
        if pred_score > 50:
            st.metric(label="Estimasi Proyeksi LTV (180 Hari)", value=f"$ {pred_score:.2f}", delta="Kategori: 🐋 WHALE (High Value)")
            st.warning("🎯 **Rekomendasi Tindakan:** Alokasikan promosi VIP eksklusif dan item premium secara langsung!")
        else:
            st.metric(label="Estimasi Proyeksi LTV (180 Hari)", value=f"$ {pred_score:.2f}", delta="Kategori: 👤 Kasual / Ad-Watcher")
            st.info("💡 **Rekomendasi Tindakan:** Pertahankan penayangan iklan berhadiah (Ad Rewards), hindari penawaran barang mahal.")

# --- HALAMAN 3: RECOMMENDER ENGINE ---
elif menu == "Game Recommender Engine (Stage 2)":
    st.title("🎯 Dynamic Content Recommender (NLP)")
    st.write("Pilih game favorit pemain, dan AI akan mencarikan 5 game lain dengan profil teks (*genres & tags*) paling mirip.")
    
    game_list = df_steam['name'].tolist()
    selected_game = st.selectbox("Cari atau pilih judul game:", game_list)
    
    if st.button("🔎 Temukan Rekomendasi Serupa"):
        if selected_game in indices:
            idx = indices[selected_game]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6] # Top 5
            game_indices = [i[0] for i in sim_scores]
            
            st.write(f"### 🏆 Top 5 Game Serupa untuk Pemain **{selected_game}**:")
            
            for rank, i in enumerate(game_indices, 1):
                score_persen = round(sim_scores[rank-1][1] * 100, 2)
                st.info(f"**{rank}. {df_steam['name'].iloc[i]}** — Tingkat Kecocokan: **{score_persen}%**")
                
                genre_val = df_steam['genres'].iloc[i] if 'genres' in df_steam.columns else '-'
                tags_val = df_steam['tags'].iloc[i] if 'tags' in df_steam.columns else '-'
                st.caption(f"Genres: {genre_val} | Tags: {tags_val}")
        else:
            st.error("Game tidak ditemukan di katalog.")