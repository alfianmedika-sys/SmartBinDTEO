"""
db_helper.py
Fungsi bantu tulis dan baca database — dipakai oleh:
  - WasteSorter.py (RPi, tulis deteksi)
  - app.py / Flask (baca untuk dashboard)
  - Endpoint penerima data ESP32 (tulis sensor + berat)
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "smartbin.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # hasil query bisa diakses kayak dict
    return conn


# =====================================================================
# WRITE — dipanggil dari WasteSorter.py dan endpoint Flask penerima ESP32
# =====================================================================

def insert_sensor(bin_id: str, distance_cm: float, bin_height_cm: float):
    """Simpan satu pembacaan ultrasonik."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sensor_readings (bin_id, distance_cm, bin_height_cm) VALUES (?,?,?)",
            (bin_id, distance_cm, bin_height_cm)
        )

def insert_weight(bin_id: str, weight_kg: float):
    """Simpan satu pembacaan berat load cell."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO weight_readings (bin_id, weight_kg) VALUES (?,?)",
            (bin_id, weight_kg)
        )

def insert_detection(class_name: str, confidence: float, routed_to: str):
    """Simpan satu hasil deteksi YOLO."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO detections (class_name, confidence, routed_to) VALUES (?,?,?)",
            (class_name, confidence, routed_to)
        )


# =====================================================================
# READ — dipanggil dari app.py (Flask) untuk endpoint dashboard
# =====================================================================

def get_latest_sensor(bin_id: str) -> dict | None:
    """Ambil pembacaan ultrasonik terbaru untuk satu tong."""
    with get_conn() as conn:
        row = conn.execute("""
            SELECT bin_id, distance_cm, bin_height_cm, level_pct, timestamp
            FROM sensor_readings
            WHERE bin_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (bin_id,)).fetchone()
    return dict(row) if row else None

def get_latest_weight(bin_id: str) -> dict | None:
    """Ambil berat terbaru untuk satu tong."""
    with get_conn() as conn:
        row = conn.execute("""
            SELECT bin_id, weight_kg, timestamp
            FROM weight_readings
            WHERE bin_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (bin_id,)).fetchone()
    return dict(row) if row else None

def get_recent_detections(limit: int = 20) -> list[dict]:
    """Ambil N deteksi terbaru."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, class_name, confidence, routed_to, timestamp
            FROM detections
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]

def get_detection_summary() -> dict:
    """Hitung total deteksi per kelas sejak awal."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT class_name, COUNT(*) as total
            FROM detections
            GROUP BY class_name
        """).fetchall()
    return {r["class_name"]: r["total"] for r in rows}

def get_all_latest() -> dict:
    """
    Snapshot lengkap untuk endpoint /api/latest:
    level + berat kedua tong + deteksi terbaru.
    """
    return {
        "sensor": {
            "organik":    get_latest_sensor("organik"),
            "anorganik":  get_latest_sensor("anorganik"),
        },
        "weight": {
            "organik":    get_latest_weight("organik"),
            "anorganik":  get_latest_weight("anorganik"),
        },
        "detections":        get_recent_detections(limit=10),
        "detection_summary": get_detection_summary(),
    }


# =====================================================================
# READ HISTORY — tambahan untuk endpoint grafik
# =====================================================================

def get_sensor_history(bin_id: str, limit: int = 50) -> list[dict]:
    """Riwayat level tong untuk grafik (diurutkan ASC agar grafik runtut)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT timestamp, distance_cm, bin_height_cm, level_pct
            FROM sensor_readings
            WHERE bin_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (bin_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]   # balik ke ASC

def get_weight_history(bin_id: str, limit: int = 50) -> list[dict]:
    """Riwayat berat tong untuk grafik (diurutkan ASC)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT timestamp, weight_kg
            FROM weight_readings
            WHERE bin_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (bin_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]