from fastapi.testclient import TestClient
from api import app, classify_player_tier

client = TestClient(app)


def test_root_and_health_endpoints():
    """Menguji endpoint root dan health check merespons status 200 OK."""
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert "endpoints" in root_res.json()

    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"


def test_predict_ltv_casual_player_and_db_logging():
    """Menguji prediksi LTV untuk pemain kasual (F2P/Minnow) dan memvalidasi pencatatan SQLite."""
    payload = {
        "Platform": "Android",
        "Play_Day": "Minggu",
        "Session_Duration": 12.0,
        "Play_Frequency": 2,
        "Event_Hour": 14,
        "D1_Revenue": 0.0
    }

    response = client.post("/predict-ltv", json=payload)
    assert response.status_code == 200, f"API Error: {response.text}"

    data = response.json()
    assert "predicted_ltv_180_days" in data
    assert data["predicted_ltv_180_days"] >= 0.0
    assert "player_segment" in data
    assert "recommended_crm_action" in data
    assert data.get("db_logged") is True


def test_predict_ltv_whale_player():
    """Menguji skenario pemain berpotensi Whale dengan transaksi awal dan durasi tinggi."""
    payload = {
        "Platform": "iOS",
        "Play_Day": "Sabtu",
        "Session_Duration": 120.0,
        "Play_Frequency": 14,
        "Event_Hour": 20,
        "D1_Revenue": 49.99
    }

    response = client.post("/predict-ltv", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_ltv_180_days"] > 0.0
    assert data["status"] == "success"


def test_predict_ltv_invalid_payload_validation():
    """Menguji validasi Pydantic agar menolak input negatif atau jam di luar 0-23 (HTTP 422)."""
    invalid_payload = {
        "Platform": "Android",
        "Play_Day": "Senin",
        "Session_Duration": -10.0,  # Tidak boleh negatif
        "Play_Frequency": 5,
        "Event_Hour": 28,           # Jam tidak valid (>23)
        "D1_Revenue": 5.0
    }
    response = client.post("/predict-ltv", json=invalid_payload)
    assert response.status_code == 422


def test_recommend_games_endpoint_no_self_match():
    """Menguji endpoint rekomendasi NLP dan memastikan tidak ada self-match dengan input."""
    payload = {
        "favorite_games": ["Hades", "The Art Theft by Jay Doherty"],
        "top_n": 5
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) <= 5

    # Pastikan "Hades" tidak muncul kembali di hasil rekomendasi (No Self-Match)
    recommended_titles = [item["title"].lower() for item in data["recommendations"]]
    assert "hades" not in recommended_titles


def test_predictions_history_and_stats():
    """Menguji pengambilan riwayat database SQLite dan ringkasan statistik."""
    hist_response = client.get("/predictions-history?limit=5")
    assert hist_response.status_code == 200
    hist_data = hist_response.json()
    assert hist_data["count"] >= 1
    assert "summary_stats" in hist_data
    assert hist_data["summary_stats"]["total_predictions_logged"] >= 1


def test_tier_classification_logic():
    """Menguji fungsi bisnis pengelompokan tier pemain secara langsung."""
    whale = classify_player_tier(150.0)
    assert whale["tier_code"] == "TIER_1_WHALE"

    f2p = classify_player_tier(0.0)
    assert f2p["tier_code"] == "TIER_4_F2P"