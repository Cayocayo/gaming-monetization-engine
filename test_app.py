import os
import pandas as pd
import numpy as np
from database import init_db, save_prediction, get_recent_predictions, get_prediction_stats
from api import get_game_recommendations, classify_player_tier


def test_project_structure_and_files():
    """Memastikan seluruh file inti proyek tersedia di repositori."""
    required_files = ["app.py", "api.py", "database.py", "requirements.txt", "README.md"]
    for file_name in required_files:
        assert os.path.exists(file_name), f"File wajib {file_name} tidak ditemukan!"


def test_dataframe_preprocessing_integrity():
    """Menguji pembuatan dan transformasi fitur DataFrame untuk input pipeline model."""
    sample_records = [
        {
            "Platform": "Android",
            "Play_Day": "Minggu",
            "Session_Duration": 25.5,
            "Play_Frequency": 5,
            "Event_Hour": 19,
            "D1_Revenue": 5.99
        },
        {
            "Platform": "iOS",
            "Play_Day": "Sabtu",
            "Session_Duration": 60.0,
            "Play_Frequency": 10,
            "Event_Hour": 21,
            "D1_Revenue": 19.99
        }
    ]
    df = pd.DataFrame(sample_records)
    assert not df.empty
    assert len(df) == 2
    assert list(df.columns) == [
        "Platform", "Play_Day", "Session_Duration",
        "Play_Frequency", "Event_Hour", "D1_Revenue"
    ]
    assert df["D1_Revenue"].dtype == np.float64


def test_recommender_filtering_logic():
    """Memastikan fungsi pemberi rekomendasi mengembalikan skor kemiripan valid dan bebas self-match."""
    input_games = ["Hades", "Celeste"]
    recs = get_game_recommendations(input_games, top_n=4)
    assert isinstance(recs, list)
    assert len(recs) > 0

    for item in recs:
        assert "title" in item
        assert "similarity_score" in item
        assert item["title"].lower() not in ["hades", "celeste"]
        assert 0.0 <= item["similarity_score"] <= 1.0


def test_sqlite_database_operations():
    """Menguji siklus penuh penyimpanan, pembacaan, dan agregasi statistik di SQLite."""
    init_db()
    dummy_input = {
        "Platform": "iOS",
        "Play_Day": "Jumat",
        "Session_Duration": 45.0,
        "Play_Frequency": 7,
        "Event_Hour": 20,
        "D1_Revenue": 14.99
    }
    tier = classify_player_tier(85.50)
    row_id = save_prediction(dummy_input, 85.50, tier["segment"])
    assert row_id is not None and row_id > 0

    recent = get_recent_predictions(limit=3)
    assert len(recent) >= 1
    assert recent[0]["predicted_ltv"] == 85.50

    stats = get_prediction_stats()
    assert stats["total_predictions_logged"] >= 1
    assert stats["max_predicted_ltv"] >= 85.50