from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_ltv_endpoint():
    # Simulasi data payload yang dikirim ke API
    payload = {
        "Platform": "Android",
        "Play_Day": "Minggu",
        "Session_Duration": 25.5,
        "Play_Frequency": 5,
        "Event_Hour": 19,
        "D1_Revenue": 5.99
    }
    
    # Kirim request POST ke endpoint
    response = client.post("/predict-ltv", json=payload)
    
    # Pastikan API merespons dengan status 200 OK, cetak error jika gagal
    assert response.status_code == 200, f"API Error Output: {response.text}"
    
    # Pastikan response memiliki key yang diharapkan
    data = response.json()
    assert "predicted_ltv_180_days" in data
    assert "player_segment" in data