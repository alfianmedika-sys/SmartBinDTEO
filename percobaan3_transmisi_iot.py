"""
============================================================
 percobaan3_transmisi_iot.py
 Percobaan 3 — Transmisi Data IoT ke Dashboard
 SmartBin — Pengolahan Citra Digital

 Skenario yang diuji:
   1. Normal          — WiFi stabil, jarak < 5m  (3x ulangan)
   2. Jarak jauh      — 10m, 15m, 20m
   3. Sinyal lemah    — 1 bar (2x ulangan)
   4. Koneksi putus   — WiFi terputus & reconnect
   5. Beban data      — 10 dan 20 pengiriman berturut-turut

 Yang diukur:
   - Latency POST ke /api/detection (ms)
   - Keberhasilan data diterima dashboard (200/gagal)

 Cara pakai:
   - Jalankan kode ini SEMENTARA Flask (App.py) jalan di RPi
   - Tekan ENTER untuk kirim data sesuai skenario yang dipilih
   - Untuk skenario "koneksi putus": matikan WiFi saat kode berjalan

 Output:
   - CSV: hasil_percobaan3.csv
   - Log latency real-time di terminal
============================================================
"""

import os
import csv
import time
import requests
import statistics
from datetime import datetime

# ── KONFIGURASI ────────────────────────────────────────────
FLASK_URL     = "http://127.0.0.1:5000"    # ganti ke IP RPi jika dari device lain
FLASK_TIMEOUT = 10                          # timeout per request (detik)

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hasil_percobaan3.csv")
CSV_HEADER = ["No", "Timestamp", "Skenario_Uji", "Kondisi_Jaringan",
              "Data_Dikirim", "Latency_ms", "Data_Diterima_Dashboard",
              "HTTP_Status", "Keterangan"]

# Data sampel yang dikirim (simulasi deteksi)
SAMPLE_PAYLOADS = [
    {"class_name": "anorganik", "confidence": 0.92, "routed_to": "anorganik"},
    {"class_name": "organik",   "confidence": 0.88, "routed_to": "organik"},
    {"class_name": "anorganik_styrofoam", "confidence": 0.85, "routed_to": "anorganik"},
]

# Skenario sesuai tabel percobaan 3
SKENARIO = [
    {"no": 1,  "nama": "Normal",        "kondisi": "WiFi stabil, jarak < 5m",          "ulangan": 1},
    {"no": 2,  "nama": "Normal",        "kondisi": "WiFi stabil, jarak < 5m",          "ulangan": 2},
    {"no": 3,  "nama": "Normal",        "kondisi": "WiFi stabil, jarak < 5m",          "ulangan": 3},
    {"no": 4,  "nama": "Jarak jauh",    "kondisi": "WiFi stabil, jarak 10m",           "ulangan": 1},
    {"no": 5,  "nama": "Jarak jauh",    "kondisi": "WiFi stabil, jarak 15m",           "ulangan": 1},
    {"no": 6,  "nama": "Jarak jauh",    "kondisi": "WiFi stabil, jarak 20m",           "ulangan": 1},
    {"no": 7,  "nama": "Sinyal lemah",  "kondisi": "WiFi sinyal rendah (1 bar)",       "ulangan": 1},
    {"no": 8,  "nama": "Sinyal lemah",  "kondisi": "WiFi sinyal rendah (1 bar)",       "ulangan": 2},
    {"no": 9,  "nama": "Koneksi putus", "kondisi": "WiFi terputus saat kirim",         "ulangan": 1},
    {"no": 10, "nama": "Koneksi putus", "kondisi": "WiFi reconnect otomatis",          "ulangan": 1},
    {"no": 11, "nama": "Beban data",    "kondisi": "10 pengiriman berturut-turut",     "ulangan": 1, "burst": 10},
    {"no": 12, "nama": "Beban data",    "kondisi": "20 pengiriman berturut-turut",     "ulangan": 1, "burst": 20},
]


# ── CSV HELPER ─────────────────────────────────────────────
def init_csv():
    file_exists = os.path.exists(OUTPUT_CSV)
    f = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(CSV_HEADER)
        f.flush()
    return f, writer


# ── KIRIM SATU REQUEST ─────────────────────────────────────
def kirim_data(payload, timeout=FLASK_TIMEOUT):
    """
    Kirim satu POST ke /api/detection dan ukur latency.
    Return: (latency_ms, status_code, sukses)
    """
    t_mulai = time.time()
    try:
        r = requests.post(
            f"{FLASK_URL}/api/detection",
            json=payload,
            timeout=timeout
        )
        latency_ms = (time.time() - t_mulai) * 1000
        sukses = (r.status_code == 200)
        return round(latency_ms, 2), r.status_code, sukses
    except requests.exceptions.Timeout:
        latency_ms = (time.time() - t_mulai) * 1000
        return round(latency_ms, 2), "TIMEOUT", False
    except requests.exceptions.ConnectionError:
        latency_ms = (time.time() - t_mulai) * 1000
        return round(latency_ms, 2), "CONNECTION_ERROR", False
    except Exception as e:
        latency_ms = (time.time() - t_mulai) * 1000
        return round(latency_ms, 2), f"ERROR:{e}", False


# ── SKENARIO BURST ─────────────────────────────────────────
def jalankan_burst(n, writer, f_csv, skenario, payload):
    """Kirim n request berturut-turut, catat setiap hasil."""
    latencies = []
    print(f"\n[BURST] Mengirim {n} request berturut-turut...")
    for i in range(n):
        lat, status, sukses = kirim_data(payload)
        latencies.append(lat)
        label = "Ya" if sukses else "Tidak"
        ket = f"burst {i+1}/{n}"
        writer.writerow([skenario["no"],
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         skenario["nama"], skenario["kondisi"],
                         f"{payload['class_name']}+berat",
                         lat, label, status, ket])
        f_csv.flush()
        print(f"  [{i+1:2d}/{n}] Latency: {lat:.1f}ms | Status: {status}")
        time.sleep(0.05)   # jeda minimal antar request

    print(f"\n[BURST SUMMARY] n={n}")
    print(f"  Min    : {min(latencies):.1f} ms")
    print(f"  Max    : {max(latencies):.1f} ms")
    print(f"  Rata2  : {statistics.mean(latencies):.1f} ms")
    print(f"  Median : {statistics.median(latencies):.1f} ms")
    if len(latencies) > 1:
        print(f"  Stdev  : {statistics.stdev(latencies):.1f} ms")


# ── CEK KONEKSI FLASK ──────────────────────────────────────
def cek_flask():
    try:
        r = requests.get(f"{FLASK_URL}/api/latest", timeout=3)
        return r.status_code == 200
    except:
        return False


# ── MAIN ───────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  PERCOBAAN 3 — Transmisi Data IoT ke Dashboard")
    print("=" * 55)
    print(f"  Flask URL  : {FLASK_URL}")
    print(f"  Output CSV : {OUTPUT_CSV}")
    print()

    # Cek Flask
    print("[CEK] Menghubungi Flask API...")
    if cek_flask():
        print("[CEK] Flask OK - siap menerima data\n")
    else:
        print("[WARNING] Flask tidak merespons!")
        print(f"  Pastikan App.py sudah jalan di {FLASK_URL}")
        print("  Lanjutkan percobaan? (y/n): ", end="")
        if input().strip().lower() != 'y':
            return

    f_csv, writer = init_csv()

    print("Pilih mode:")
    print("  1 - Jalankan semua skenario secara berurutan (otomatis)")
    print("  2 - Pilih skenario manual satu per satu")
    mode = input("Pilihan (1/2): ").strip()

    if mode == "1":
        # ── Mode otomatis ──────────────────────────────────
        payload_idx = 0
        for skenario in SKENARIO:
            payload = SAMPLE_PAYLOADS[payload_idx % len(SAMPLE_PAYLOADS)]
            payload_idx += 1

            print(f"\n{'='*50}")
            print(f"[SKENARIO #{skenario['no']}] {skenario['nama']}")
            print(f"  Kondisi : {skenario['kondisi']}")

            if skenario["nama"] == "Koneksi putus":
                print(f"  [AKSI]  Atur kondisi jaringan sesuai kondisi di atas,")
                print(f"          lalu tekan ENTER untuk mulai pengiriman...")
                input()

            elif skenario.get("burst"):
                jalankan_burst(skenario["burst"], writer, f_csv, skenario, payload)
                continue

            else:
                print(f"  Tekan ENTER untuk kirim data...")
                input()

            lat, status, sukses = kirim_data(payload)
            label = "Ya" if sukses else "Tidak"
            ket = input("  Keterangan (kosong=OK): ").strip()

            writer.writerow([skenario["no"],
                             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             skenario["nama"], skenario["kondisi"],
                             f"{payload['class_name']}+berat",
                             lat, label, status, ket])
            f_csv.flush()
            print(f"  [HASIL] Latency={lat}ms | Status={status} | Diterima={label}")

    else:
        # ── Mode manual ────────────────────────────────────
        while True:
            print("\nDaftar skenario:")
            for s in SKENARIO:
                print(f"  {s['no']:2d}. [{s['nama']}] {s['kondisi']}")
            print("   0. Keluar")
            try:
                pilih = int(input("Pilih nomor skenario: "))
            except ValueError:
                continue

            if pilih == 0:
                break

            skenario = next((s for s in SKENARIO if s["no"] == pilih), None)
            if not skenario:
                print("[ERROR] Nomor tidak valid")
                continue

            payload = SAMPLE_PAYLOADS[pilih % len(SAMPLE_PAYLOADS)]

            if skenario.get("burst"):
                jalankan_burst(skenario["burst"], writer, f_csv, skenario, payload)
            else:
                print(f"\n[INFO] Atur kondisi: {skenario['kondisi']}")
                print("Tekan ENTER untuk kirim...")
                input()

                lat, status, sukses = kirim_data(payload)
                label = "Ya" if sukses else "Tidak"
                ket = input("Keterangan (kosong=OK): ").strip()

                writer.writerow([skenario["no"],
                                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 skenario["nama"], skenario["kondisi"],
                                 f"{payload['class_name']}+berat",
                                 lat, label, status, ket])
                f_csv.flush()
                print(f"[HASIL] Latency={lat}ms | Status={status} | Diterima={label}")

    f_csv.close()
    print(f"\n[SELESAI] Data tersimpan di: {OUTPUT_CSV}")

    # Tampilkan ringkasan
    print("\n[RINGKASAN] Baca CSV untuk analisis lengkap:")
    import pandas as pd
    try:
        df = pd.read_csv(OUTPUT_CSV)
        print(df[["No","Skenario_Uji","Latency_ms","Data_Diterima_Dashboard"]].to_string(index=False))
    except:
        print(f"  Buka file: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
