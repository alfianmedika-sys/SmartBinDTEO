"""
db_setup.py
Jalankan SEKALI untuk membuat file database SmartBin.
Usage: python3 db_setup.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "smartbin.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # -------------------------------------------------
    # Tabel 1: Pembacaan ultrasonik -> volume/level tong
    # ESP32 kirim data ini via HTTP POST
    # -------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   DATETIME DEFAULT (datetime('now','localtime')),
            bin_id      TEXT    NOT NULL,   -- 'organik' | 'anorganik'
            distance_cm REAL    NOT NULL,   -- jarak ultrasonik (cm)
            bin_height_cm REAL  NOT NULL,   -- tinggi total tong (cm)
            level_pct   REAL    GENERATED ALWAYS AS
                            (ROUND((1.0 - distance_cm / bin_height_cm) * 100.0, 1))
                        VIRTUAL                -- % penuh, dihitung otomatis
        )
    """)

    # -------------------------------------------------
    # Tabel 2: Pembacaan load cell -> berat sampah
    # ESP32 kirim data ini via HTTP POST
    # -------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weight_readings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   DATETIME DEFAULT (datetime('now','localtime')),
            bin_id      TEXT    NOT NULL,   -- 'organik' | 'anorganik'
            weight_kg   REAL    NOT NULL    -- berat dalam kg
        )
    """)

    # -------------------------------------------------
    # Tabel 3: Hasil deteksi YOLO dari Raspberry Pi
    # WasteSorter.py tulis langsung ke DB ini
    # -------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   DATETIME DEFAULT (datetime('now','localtime')),
            class_name  TEXT    NOT NULL,   -- 'organik' | 'anorganik' | 'anorganik_styrofoam'
            confidence  REAL    NOT NULL,   -- 0.0 - 1.0
            routed_to   TEXT                -- 'organik' | 'anorganik' (tong tujuan servo)
        )
    """)

    # Index agar query dashboard cepat
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sensor_ts   ON sensor_readings (timestamp DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_weight_ts   ON weight_readings  (timestamp DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_detect_ts   ON detections       (timestamp DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sensor_bin  ON sensor_readings  (bin_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_weight_bin  ON weight_readings  (bin_id)")

    conn.commit()
    conn.close()
    print(f"[DB] Database siap: {DB_PATH}")
    print("[DB] Tabel yang dibuat:")
    print("     - sensor_readings  (ultrasonik -> level tong)")
    print("     - weight_readings  (load cell  -> berat)")
    print("     - detections       (YOLO       -> hasil klasifikasi)")

if __name__ == "__main__":
    init_db()