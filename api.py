from fastapi import FastAPI
from pydantic import BaseModel
from database import init_db, log_prediction

app = FastAPI(title="Gaming Monetization AI API")

# Skema Input Request
class PlayerData(BaseModel):
    platform: str
    session_duration: float
    event_hour: int
    revenue_usd: float

# Inisialisasi DB saat API menyala
@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/api/v1/predict-ltv")
def predict_ltv(data: PlayerData):
    # Simulasi inferensi (sesuaikan dengan logic model Anda)
    pred_score = data.revenue_usd * 2.5 + (15.0 if data.platform == 'iOS' else 5.0)
    
    # Simpan hasil ke Database SQLite
    log_prediction(data.platform, data.session_duration, data.event_hour, data.revenue_usd, pred_score)
    
    tier = "Whale" if pred_score > 50 else "Casual"
    
    return {
        "status": "success",
        "predicted_ltv_usd": round(pred_score, 2),
        "player_tier": tier,
        "message": "Prediction logged to database successfully"
    }