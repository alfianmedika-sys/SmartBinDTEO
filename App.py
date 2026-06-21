"""
app.py — Flask Backend API SmartBin
Jalankan di Raspberry Pi: python3 App.py
"""

from flask import Flask, request, jsonify, send_from_directory
import os
from functools import wraps
import db_helper

app = Flask(__name__)

BIN_IDS = {"organik", "anorganik"}


# =====================================================================
# HELPER
# =====================================================================

def success(data):
    return jsonify({"status": "ok", "data": data})

def error(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code

def require_json(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return error("Content-Type harus application/json")
        return f(*args, **kwargs)
    return wrapper


# =====================================================================
# SERVE FRONTEND
# =====================================================================

@app.route("/")
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(os.path.join(base_dir, "static"), "index.html")


# =====================================================================
# GET — READ
# =====================================================================

@app.route("/api/latest")
def latest():
    return success(db_helper.get_all_latest())

@app.route("/api/sensor/<bin_id>")
def sensor_latest(bin_id):
    if bin_id not in BIN_IDS:
        return error(f"bin_id tidak valid. Pilihan: {BIN_IDS}")
    data = db_helper.get_latest_sensor(bin_id)
    if data is None:
        return error(f"Belum ada data untuk tong '{bin_id}'", 404)
    return success(data)

@app.route("/api/weight/<bin_id>")
def weight_latest(bin_id):
    if bin_id not in BIN_IDS:
        return error(f"bin_id tidak valid. Pilihan: {BIN_IDS}")
    data = db_helper.get_latest_weight(bin_id)
    if data is None:
        return error(f"Belum ada data untuk tong '{bin_id}'", 404)
    return success(data)

@app.route("/api/detections")
def detections():
    limit = request.args.get("limit", 20, type=int)
    limit = max(1, min(limit, 100))
    return success(db_helper.get_recent_detections(limit=limit))

@app.route("/api/detections/summary")
def detections_summary():
    return success(db_helper.get_detection_summary())

@app.route("/api/history/sensor")
def history_sensor():
    bin_id = request.args.get("bin_id")
    if bin_id not in BIN_IDS:
        return error(f"bin_id wajib diisi. Pilihan: {BIN_IDS}")
    limit = request.args.get("limit", 50, type=int)
    limit = max(1, min(limit, 500))
    return success(db_helper.get_sensor_history(bin_id, limit))

@app.route("/api/history/weight")
def history_weight():
    bin_id = request.args.get("bin_id")
    if bin_id not in BIN_IDS:
        return error(f"bin_id wajib diisi. Pilihan: {BIN_IDS}")
    limit = request.args.get("limit", 50, type=int)
    limit = max(1, min(limit, 500))
    return success(db_helper.get_weight_history(bin_id, limit))


# =====================================================================
# POST — WRITE
# =====================================================================

@app.route("/api/sensor", methods=["POST"])
@require_json
def sensor_post():
    body        = request.get_json()
    bin_id      = body.get("bin_id")
    distance_cm = body.get("distance_cm")
    bin_height  = body.get("bin_height_cm")

    if bin_id not in BIN_IDS:
        return error(f"bin_id tidak valid. Pilihan: {BIN_IDS}")
    if not isinstance(distance_cm, (int, float)) or distance_cm < 0:
        return error("distance_cm harus angka positif")
    if not isinstance(bin_height, (int, float)) or bin_height <= 0:
        return error("bin_height_cm harus angka positif")
    if distance_cm > bin_height:
        return error("distance_cm tidak boleh melebihi bin_height_cm")

    db_helper.insert_sensor(bin_id, float(distance_cm), float(bin_height))
    return success({"inserted": True, "bin_id": bin_id})

@app.route("/api/weight", methods=["POST"])
@require_json
def weight_post():
    body      = request.get_json()
    bin_id    = body.get("bin_id")
    weight_kg = body.get("weight_kg")

    if bin_id not in BIN_IDS:
        return error(f"bin_id tidak valid. Pilihan: {BIN_IDS}")
    if not isinstance(weight_kg, (int, float)) or weight_kg < 0:
        return error("weight_kg harus angka >= 0")

    db_helper.insert_weight(bin_id, float(weight_kg))
    return success({"inserted": True, "bin_id": bin_id})

@app.route("/api/detection", methods=["POST"])
@require_json
def detection_post():
    body       = request.get_json()
    class_name = body.get("class_name")
    confidence = body.get("confidence")
    routed_to  = body.get("routed_to")

    VALID_CLASSES = {"organik", "anorganik", "anorganik_styrofoam"}
    if class_name not in VALID_CLASSES:
        return error(f"class_name tidak valid. Pilihan: {VALID_CLASSES}")
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        return error("confidence harus angka antara 0.0 dan 1.0")
    if routed_to not in BIN_IDS:
        return error(f"routed_to tidak valid. Pilihan: {BIN_IDS}")

    db_helper.insert_detection(class_name, float(confidence), routed_to)
    return success({"inserted": True, "class_name": class_name})


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)