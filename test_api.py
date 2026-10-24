from fastapi.testclient import TestClient
from api import app  # pastikan nama file api Anda adalah api.py

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
    
    # Pastikan API merespons dengan status 200 OK
    assert response.status_code == 200
    
    # Pastikan response memiliki key "predicted_ltv_180_days"
    data = response.json()
    assert "predicted_ltv_180_days" in data
    assert "player_segment" in data