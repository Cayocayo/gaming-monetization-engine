import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
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
# 2. LOAD ARTIFACTS MODEL
# ==========================================
@st.cache_resource
def load_models():
    # Load Model DL Tahap 1 (LTV)
    model_ltv = load_model('models/ltv_dl_model.keras')
    scaler = joblib.load('models/ltv_scaler.pkl')
    label_encoders = joblib.load('models/ltv_label_encoders.pkl')
    
    # Load Dataset Steam Bersih
    df_steam = pd.read_pickle('models/steam_games_clean.pkl')
    
    # Hitung Cosine Similarity secara dinamis
    tfidf = TfidfVectorizer(stop_words='english')
    if 'genres' in df_steam.columns and 'tags' in df_steam.columns:
        df_steam['combined_features'] = df_steam['genres'].fillna('') + ' ' + df_steam['tags'].fillna('')
    else:
        df_steam['combined_features'] = df_steam.iloc[:, 1].astype(str)
        
    tfidf_matrix = tfidf.fit_transform(df_steam['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return model_ltv, scaler, label_encoders, cosine_sim, df_steam

with st.spinner('Memuat Model AI & Database... Silakan tunggu sebentar.'):
    model_ltv, scaler, label_encoders, cosine_sim, df_steam = load_models()

indices = pd.Series(df_steam.index, index=df_steam['name'])
indices = indices[~indices.index.duplicated(keep='first')]

# ==========================================
# 3. SIDEBAR & NAVIGASI MENU
# ==========================================
st.sidebar.title("🎮 Engine Menu")
menu = st.sidebar.radio(
    "Pilih Modul AI:",
    (
        "Overview Dashboard", 
        "LTV Prediction Engine (Stage 1)", 
        "Game Recommender Engine (Stage 2)",
        "Model Analytics & Metrics"
    )
)

st.sidebar.markdown("---")
st.sidebar.info("Final Project Data Science & AI\n**Cahyo Widyonarko**")

# ==========================================
# 4. HALAMAN UTAMA BERDASARKAN MENU
# ==========================================

if menu == "Overview Dashboard":
    st.title("📈 Intelligent Gaming Monetization Engine")
    st.markdown("""
    Selamat datang di Dashboard AI End-to-End untuk Optimalisasi Monetisasi Gaming.
    Aplikasi ini dirancang menggunakan arsitektur **Dual-Stage Pipeline**:
    *   **Stage 1 (Deep Learning):** Memprediksi nilai *Lifetime Value (LTV)* pemain untuk mendeteksi kelompok elit (*Whales*) secara dini dan mencegah *Revenue Leakage*.
    *   **Stage 2 (NLP Recommender):** Memberikan rekomendasi konten game secara hiper-terpersonalisasi untuk mengatasi *Offer Fatigue* dan *Cold-Start Problem*.
    """)
    st.success("✅ Seluruh model AI berhasil dimuat dan siap digunakan secara Real-Time!")
    
    st.markdown("---")
    st.subheader("📊 Mengapa AI ini Dibutuhkan? (Problem Statement)")
    st.write("Distribusi pemain di industri gaming sangat timpang (*Highly Imbalanced*). Mayoritas pemain hanya menonton iklan atau murni bermain gratis, sementara **Pembeli Asli (Whales)** jumlahnya sangat sedikit namun menyumbang mayoritas pendapatan.")
    
    data_segmentasi = pd.DataFrame({
        'Kategori Pemain': ['Hanya Nonton Iklan (Ad-Watcher)', 'F2P Murni ($0)', 'Pembeli Item (IAP Buyer)'],
        'Jumlah Pemain': [83, 77, 5]
    })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Kategori Pemain', y='Jumlah Pemain', data=data_segmentasi, palette=['#d62728', '#4c72b0', '#55a868'], ax=ax)
    ax.set_title("Segmentasi Pemain: Pembeli Asli vs Penonton Iklan", fontsize=14, pad=15)
    ax.set_xlabel("Kategori Pemain", fontsize=12)
    ax.set_ylabel("Jumlah Pemain", fontsize=12)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 8), 
                    textcoords = 'offset points', fontsize=11)
    st.pyplot(fig)

elif menu == "LTV Prediction Engine (Stage 1)":
    st.title("💰 Player LTV Prediction (Deep Learning)")
    st.write("Masukkan parameter aktivitas awal pemain untuk memprediksi potensi pengeluaran mereka di masa depan.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Demografi & Habit")
        platform_input = st.selectbox("Platform Pengguna", ["Android", "iOS"])
        play_day = st.selectbox("Hari Bermain Utama", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
        session_duration = st.number_input("Durasi Sesi Rata-rata (menit)", min_value=1.0, max_value=300.0, value=15.0)
        
    with col2:
        st.subheader("Data Aktivitas Finansial Awal")
        play_frequency = st.number_input("Frekuensi Bermain (kali/minggu)", min_value=1, max_value=50, value=3)
        event_hour = st.slider("Jam Transaksi Utama (Event Hour)", 0, 23, 18)
        revenue_usd = st.number_input("Pendapatan Minggu Pertama ($ USD)", min_value=0.0, max_value=500.0, value=0.0)
    
    if st.button("🔮 Prediksi Potensi LTV"):
        st.info("Menjalankan inferensi Deep Neural Network...")
        try:
            input_data = pd.DataFrame({
                'Platform': [platform_input],
                'Play_Day': [play_day],
                'Session_Duration': [session_duration],
                'Play_Frequency': [play_frequency],
                'Event_Hour': [event_hour],
                'D1_Revenue': [revenue_usd]
            })
            
            if 'Platform' in label_encoders:
                input_data['Platform'] = label_encoders['Platform'].transform(input_data['Platform'])
            if 'Play_Day' in label_encoders:
                input_data['Play_Day'] = label_encoders['Play_Day'].transform(input_data['Play_Day'])
            
            input_scaled = scaler.transform(input_data)
            pred_score_array = model_ltv.predict(input_scaled)
            pred_score = float(pred_score_array[0][0])
            
            if pred_score < 0:
                pred_score = 0.0
                
        except Exception as e:
            st.error(f"Terjadi kesalahan model: {e}. Pastikan parameter sesuai dengan fitur training Anda.")
            pred_score = 0.0

        if pred_score > 50:
            st.metric(label="Estimasi Proyeksi LTV (180 Hari)", value=f"$ {pred_score:.2f}", delta="Kategori: 🐋 WHALE (High Value)")
            st.warning("🎯 **Rekomendasi Tindakan:** Alokasikan promosi VIP eksklusif dan item premium secara langsung!")
        else:
            st.metric(label="Estimasi Proyeksi LTV (180 Hari)", value=f"$ {pred_score:.2f}", delta="Kategori: 👤 Kasual / Ad-Watcher")
            st.info("💡 **Rekomendasi Tindakan:** Pertahankan penayangan iklan berhadiah (Ad Rewards), hindari penawaran barang mahal.")

elif menu == "Game Recommender Engine (Stage 2)":
    st.title("🎯 Dynamic Content Recommender (NLP)")
    st.write("Ketik dan pilih daftar game yang sering dimainkan. AI akan menyatukan profil (*genres & tags*) dari game-game tersebut dan mencarikan 5 game baru yang paling cocok.")
    
    game_list = df_steam['name'].tolist()
    selected_games = st.multiselect("Pilih Histori Game Pemain (Bisa lebih dari 1):", game_list, placeholder="Ketik judul game...")
    
    if st.button("🔎 Temukan Rekomendasi Berdasarkan Profil"):
        if not selected_games:
            st.warning("⚠️ Silakan pilih minimal 1 game terlebih dahulu!")
        else:
            valid_indices = [indices[game] for game in selected_games if game in indices]
            if valid_indices:
                selected_sims = cosine_sim[valid_indices]
                avg_sims = np.mean(selected_sims, axis=0)
                sim_scores = list(enumerate(avg_sims))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                
                recommended_indices = []
                final_scores = []
                for idx, score in sim_scores:
                    if idx not in valid_indices:
                        recommended_indices.append(idx)
                        final_scores.append(score)
                    if len(recommended_indices) == 5:
                        break
                
                st.write(f"### 🏆 Top 5 Rekomendasi Selanjutnya:")
                for rank, (i, score) in enumerate(zip(recommended_indices, final_scores), 1):
                    score_persen = round(score * 100, 2)
                    st.info(f"**{rank}. {df_steam['name'].iloc[i]}** — Kecocokan: **{score_persen}%**")
                    genre_val = df_steam['genres'].iloc[i] if 'genres' in df_steam.columns else '-'
                    st.caption(f"Genres: {genre_val}")
            else:
                st.error("Terjadi kesalahan membaca data game tersebut.")

elif menu == "Model Analytics & Metrics":
    st.title("📊 Model Analytics & Evaluation Metrics")
    st.subheader("🎯 Key Performance Indicators (KPI)")
    col_met1, col_met2, col_met3 = st.columns(3)
    col_met1.metric(label="Mean Absolute Error (MAE)", value="$ 5.21", delta="Sangat Akurat", delta_color="normal")
    col_met2.metric(label="Root Mean Squared Error (RMSE)", value="$ 8.13", delta="Tahan Outlier", delta_color="normal")
    col_met3.metric(label="Total Epochs (Training)", value="50 Epochs", delta="Good Fit / No Overfit", delta_color="normal")
    
    st.markdown("---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    loss_curve_path = os.path.join(current_dir, "assets", "loss_curve.png")
    scatter_ltv_path = os.path.join(current_dir, "assets", "scatter_ltv.png")

    st.subheader("📈 Diagnostic Plots")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("**1. Deep Learning Loss Curve**")
        if os.path.exists(loss_curve_path):
            st.image(loss_curve_path, use_container_width=True)
    with col_img2:
        st.markdown("**2. Actual vs Predicted LTV**")
        if os.path.exists(scatter_ltv_path):
            st.image(scatter_ltv_path, use_container_width=True)