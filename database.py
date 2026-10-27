import sqlite3
from datetime import datetime
from typing import Dict, Any, List

DB_NAME = "predictions.db"


def init_db() -> None:
    """Membuat tabel ltv_predictions jika belum tersedia di SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ltv_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            platform TEXT,
            play_day TEXT,
            session_duration REAL,
            play_frequency INTEGER,
            event_hour INTEGER,
            d1_revenue REAL,
            predicted_ltv REAL,
            player_segment TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(data: Dict[str, Any], predicted_ltv: float, segment: str) -> int:
    """Menyimpan satu baris hasil inferensi prediksi LTV ke dalam database SQLite."""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ltv_predictions (
            timestamp, platform, play_day, session_duration,
            play_frequency, event_hour, d1_revenue, predicted_ltv, player_segment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(data.get("Platform", "Unknown")),
        str(data.get("Play_Day", "Unknown")),
        float(data.get("Session_Duration", 0.0)),
        int(data.get("Play_Frequency", 0)),
        int(data.get("Event_Hour", 0)),
        float(data.get("D1_Revenue", 0.0)),
        float(predicted_ltv),
        str(segment)
    ))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_recent_predictions(limit: int = 10) -> List[Dict[str, Any]]:
    """Mengambil daftar riwayat prediksi terbaru dalam format dictionary yang rapi."""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM ltv_predictions ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_prediction_stats() -> Dict[str, Any]:
    """Menghitung metrik agregat dari seluruh prediksi yang tersimpan di database."""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*), AVG(predicted_ltv), MAX(predicted_ltv), MIN(predicted_ltv)
        FROM ltv_predictions
    """)
    row = cursor.fetchone()
    conn.close()

    total = row[0] if row and row[0] else 0
    return {
        "total_predictions_logged": total,
        "average_predicted_ltv": round(float(row[1]), 2) if row and row[1] is not None else 0.0,
        "max_predicted_ltv": round(float(row[2]), 2) if row and row[2] is not None else 0.0,
        "min_predicted_ltv": round(float(row[3]), 2) if row and row[3] is not None else 0.0
    }