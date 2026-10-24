from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(title="Gaming Monetization API", version="1.0")

# Load model dan artifacts saat server berjalan
try:
    model_ltv = load_model('models/ltv_dl_model.keras')
    scaler = joblib.load('models/ltv_scaler.pkl')
    label_encoders = joblib.load('models/ltv_label_encoders.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

# Definisikan format input dari user (JSON)
class PlayerData(BaseModel):
    Platform: str
    Play_Day: str
    Session_Duration: float
    Play_Frequency: int
    Event_Hour: int
    D1_Revenue: float

@app.post("/predict-ltv")
def predict_ltv(data: PlayerData):
    try:
        # 1. Ubah input JSON menjadi DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # 2. Encode categorical data
        if 'Platform' in label_encoders:
            input_data['Platform'] = label_encoders['Platform'].transform(input_data['Platform'])
        if 'Play_Day' in label_encoders:
            input_data['Play_Day'] = label_encoders['Play_Day'].transform(input_data['Play_Day'])
            
        # 3. Scaling
        input_scaled = scaler.transform(input_data)
        
        # 4. Prediksi
        pred_score_array = model_ltv.predict(input_scaled)
        pred_score = float(pred_score_array[0][0])
        
        if pred_score < 0:
            pred_score = 0.0
            
        # 5. Tentukan Kategori
        kategori = "Whale (High Value)" if pred_score > 50 else "Kasual / Ad-Watcher"
        
        return {
            "predicted_ltv_180_days": round(pred_score, 2),
            "player_segment": kategori,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan inferensi: {str(e)}")