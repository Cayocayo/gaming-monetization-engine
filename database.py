import sqlite3
import datetime

def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    # Membuat tabel untuk menyimpan histori LTV
    c.execute('''CREATE TABLE IF NOT EXISTS ltv_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  platform TEXT,
                  session_duration REAL,
                  event_hour INTEGER,
                  revenue_usd REAL,
                  predicted_ltv REAL)''')
    conn.commit()
    conn.close()

def log_prediction(platform, session_duration, event_hour, revenue_usd, predicted_ltv):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute("""INSERT INTO ltv_logs 
                 (timestamp, platform, session_duration, event_hour, revenue_usd, predicted_ltv) 
                 VALUES (?, ?, ?, ?, ?, ?)""",
              (timestamp, platform, session_duration, event_hour, revenue_usd, predicted_ltv))
    conn.commit()
    conn.close()