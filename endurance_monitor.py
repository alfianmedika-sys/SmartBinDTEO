"""
============================================================
 endurance_monitor.py
 Uji Endurance Raspberry Pi 4 — SmartBin
 Durasi: 7+ jam (bisa diatur di DURASI_JAM)

 Yang diukur setiap interval:
   - Waktu elapsed
   - Suhu CPU (derajat Celsius)
   - Penggunaan CPU (%)
   - Penggunaan RAM (%)
   - RAM terpakai (MB)
   - Voltase (under-voltage flag)
   - Jumlah deteksi kumulatif dari Flask API
   - Uptime sistem

 Output:
   - CSV: hasil_endurance.csv
   - Log ringkasan di terminal setiap interval
   - Laporan akhir otomatis saat selesai

 Cara jalankan (terminal terpisah, Flask + WasteSorter sudah jalan):
   python3 endurance_monitor.py

 Bisa dihentikan kapanpun dengan Ctrl+C — data yang sudah
 tercatat tetap tersimpan di CSV.
============================================================
"""

import os
import csv
import time
import subprocess
import threading
import requests
from datetime import datetime, timedelta

# ── KONFIGURASI ───────────────────────────────────────────
DURASI_JAM       = 7        # durasi total uji endurance (jam)
INTERVAL_DETIK   = 30       # catat data setiap N detik
FLASK_URL        = "http://127.0.0.1:5000"
FLASK_TIMEOUT    = 3

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = os.path.join(BASE_DIR, "hasil_endurance.csv")
LAPORAN_TXT = os.path.join(BASE_DIR, "laporan_endurance.txt")

CSV_HEADER = [
    "No",
    "Timestamp",
    "Elapsed_menit",
    "Suhu_CPU_C",
    "CPU_pct",
    "RAM_pct",
    "RAM_MB",
    "Under_Voltage",
    "Total_Deteksi",
    "Deteksi_Organik",
    "Deteksi_Anorganik",
    "Deteksi_Styrofoam",
    "Flask_Online",
    "Keterangan",
]

DURASI_DETIK = DURASI_JAM * 3600


# ── FUNGSI BACA HARDWARE ──────────────────────────────────

def baca_suhu_cpu():
    """Baca suhu CPU RPi dari vcgencmd."""
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        # Format: temp=45.6'C
        return float(out.replace("temp=", "").replace("'C", ""))
    except Exception:
        try:
            # Fallback: baca langsung dari sysfs
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            return -1.0


def baca_cpu_pct():
    """Baca penggunaan CPU dalam persen (rata-rata 1 detik)."""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        try:
            # Fallback tanpa psutil: baca /proc/stat
            def stat():
                with open("/proc/stat") as f:
                    line = f.readline().split()
                vals = [int(x) for x in line[1:]]
                idle  = vals[3]
                total = sum(vals)
                return idle, total

            i1, t1 = stat()
            time.sleep(1)
            i2, t2 = stat()
            dt = (t2 - t1)
            di = (i2 - i1)
            return round((1 - di / dt) * 100, 1) if dt > 0 else 0.0
        except Exception:
            return -1.0


def baca_ram():
    """Baca penggunaan RAM. Return (pct, used_mb)."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return round(mem.percent, 1), round(mem.used / 1024 / 1024, 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
            total   = info.get("MemTotal", 0)
            avail   = info.get("MemAvailable", 0)
            used    = total - avail
            pct     = round(used / total * 100, 1) if total > 0 else 0
            used_mb = round(used / 1024, 1)
            return pct, used_mb
        except Exception:
            return -1.0, -1.0


def baca_under_voltage():
    """Cek apakah RPi mengalami under-voltage (throttling)."""
    try:
        out = subprocess.check_output(
            ["vcgencmd", "get_throttled"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        # throttled=0x0 = normal, throttled=0x50005 = under-voltage
        val = int(out.split("=")[1], 16)
        # Bit 0: under-voltage detected, Bit 16: under-voltage occurred
        uv_now  = bool(val & 0x1)
        uv_ever = bool(val & 0x10000)
        if uv_now:
            return "YA_SEKARANG"
        elif uv_ever:
            return "PERNAH"
        return "TIDAK"
    except Exception:
        return "N/A"


def baca_flask_data():
    """Ambil data deteksi dari Flask API."""
    try:
        r = requests.get(f"{FLASK_URL}/api/detections/summary",
                         timeout=FLASK_TIMEOUT)
        if r.status_code == 200:
            summary = r.json().get("data", {})
            organik   = summary.get("organik", 0)
            anorganik = summary.get("anorganik", 0)
            styrofoam = summary.get("anorganik_styrofoam", 0)
            total     = organik + anorganik + styrofoam
            return True, total, organik, anorganik, styrofoam
    except Exception:
        pass
    return False, 0, 0, 0, 0


# ── LAPORAN AKHIR ─────────────────────────────────────────

def buat_laporan(rows, durasi_aktual_menit):
    """Buat ringkasan laporan dari semua data yang tercatat."""
    if not rows:
        return "Tidak ada data."

    suhu_list  = [r["Suhu_CPU_C"] for r in rows if r["Suhu_CPU_C"] > 0]
    cpu_list   = [r["CPU_pct"]    for r in rows if r["CPU_pct"]    > 0]
    ram_list   = [r["RAM_pct"]    for r in rows if r["RAM_pct"]    > 0]
    uv_list    = [r["Under_Voltage"] for r in rows]
    det_list   = [r["Total_Deteksi"] for r in rows]

    suhu_max  = max(suhu_list, default=0)
    suhu_min  = min(suhu_list, default=0)
    suhu_avg  = round(sum(suhu_list) / len(suhu_list), 1) if suhu_list else 0

    cpu_max   = max(cpu_list, default=0)
    cpu_avg   = round(sum(cpu_list) / len(cpu_list), 1) if cpu_list else 0

    ram_max   = max(ram_list, default=0)
    ram_avg   = round(sum(ram_list) / len(ram_list), 1) if ram_list else 0

    uv_count  = sum(1 for u in uv_list if u in ("YA_SEKARANG", "PERNAH"))
    det_akhir = det_list[-1] if det_list else 0
    det_awal  = det_list[0]  if det_list else 0

    laporan = f"""
============================================================
 LAPORAN ENDURANCE TEST — SmartBin Raspberry Pi 4
 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
============================================================

Durasi uji      : {durasi_aktual_menit:.1f} menit ({durasi_aktual_menit/60:.2f} jam)
Total sampel    : {len(rows)} titik data (interval {INTERVAL_DETIK} detik)

SUHU CPU
  Minimum       : {suhu_min} °C
  Maksimum      : {suhu_max} °C
  Rata-rata     : {suhu_avg} °C
  {"[PERINGATAN] Suhu melebihi 80°C!" if suhu_max > 80 else "[OK] Suhu dalam batas aman (<80°C)"}

PENGGUNAAN CPU
  Maksimum      : {cpu_max} %
  Rata-rata     : {cpu_avg} %

PENGGUNAAN RAM
  Maksimum      : {ram_max} %
  Rata-rata     : {ram_avg} %

TEGANGAN
  Kejadian under-voltage : {uv_count} kali
  {"[PERINGATAN] Under-voltage terdeteksi! Periksa adaptor." if uv_count > 0 else "[OK] Tidak ada under-voltage"}

DETEKSI YOLO
  Total deteksi : {det_akhir} objek
  Deteksi baru selama uji : {det_akhir - det_awal} objek

============================================================
"""
    return laporan.strip()


# ── MAIN ──────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ENDURANCE TEST — SmartBin Raspberry Pi 4")
    print(f"  Durasi: {DURASI_JAM} jam | Interval: {INTERVAL_DETIK} detik")
    print("=" * 55)
    print(f"[INFO] Data disimpan di: {OUTPUT_CSV}")
    print(f"[INFO] Tekan Ctrl+C untuk hentikan lebih awal\n")

    # Cek psutil
    try:
        import psutil
        print("[INFO] psutil tersedia - pembacaan CPU/RAM akurat")
    except ImportError:
        print("[INFO] psutil tidak ada, pakai fallback /proc")
        print("       (opsional: pip3 install psutil --break-system-packages)")

    # Cek Flask
    flask_online, _, _, _, _ = baca_flask_data()
    if flask_online:
        print("[INFO] Flask API terdeteksi - data deteksi akan tercatat")
    else:
        print("[WARN] Flask API tidak merespons - kolom deteksi akan kosong")
        print("       Pastikan python3 App.py sudah berjalan\n")

    # Init CSV
    file_baru = not os.path.exists(OUTPUT_CSV)
    csv_file  = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer    = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
    if file_baru:
        writer.writeheader()
        csv_file.flush()

    mulai      = time.time()
    no         = 1
    all_rows   = []

    try:
        while True:
            elapsed  = time.time() - mulai
            if elapsed > DURASI_DETIK:
                print(f"\n[SELESAI] Durasi {DURASI_JAM} jam tercapai.")
                break

            # Baca semua metrik
            ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed_menit = round(elapsed / 60, 2)
            suhu     = baca_suhu_cpu()
            cpu      = baca_cpu_pct()        # ini blocking ~1 detik
            ram_pct, ram_mb = baca_ram()
            uv       = baca_under_voltage()
            flask_ok, total_det, org, anorg, styro = baca_flask_data()

            row = {
                "No":                no,
                "Timestamp":         ts,
                "Elapsed_menit":     elapsed_menit,
                "Suhu_CPU_C":        suhu,
                "CPU_pct":           cpu,
                "RAM_pct":           ram_pct,
                "RAM_MB":            ram_mb,
                "Under_Voltage":     uv,
                "Total_Deteksi":     total_det,
                "Deteksi_Organik":   org,
                "Deteksi_Anorganik": anorg,
                "Deteksi_Styrofoam": styro,
                "Flask_Online":      "YA" if flask_ok else "TIDAK",
                "Keterangan":        "",
            }

            writer.writerow(row)
            csv_file.flush()
            all_rows.append(row)

            # Log terminal ringkas
            sisa_menit = max(0, (DURASI_DETIK - elapsed) / 60)
            print(f"[{ts}] #{no:4d} | "
                  f"Elapsed: {elapsed_menit:6.1f} min | "
                  f"Suhu: {suhu:5.1f}°C | "
                  f"CPU: {cpu:5.1f}% | "
                  f"RAM: {ram_pct:5.1f}% | "
                  f"UV: {uv:12s} | "
                  f"Det: {total_det:4d} | "
                  f"Sisa: {sisa_menit:.0f} min")

            # Peringatan suhu tinggi
            if suhu > 75:
                print(f"  [!!] SUHU TINGGI: {suhu}°C — pastikan ventilasi cukup!")

            no += 1

            # Tunggu interval berikutnya (dikurangi waktu proses ~1 detik untuk cpu_pct)
            waktu_proses = time.time() - mulai - (elapsed)
            tidur = max(0, INTERVAL_DETIK - waktu_proses - 1)
            time.sleep(tidur)

    except KeyboardInterrupt:
        elapsed_akhir = time.time() - mulai
        print(f"\n[DIHENTIKAN] Elapsed: {elapsed_akhir/60:.1f} menit")

    finally:
        csv_file.close()
        durasi_aktual = (time.time() - mulai) / 60

        # Buat laporan akhir
        laporan = buat_laporan(all_rows, durasi_aktual)
        print("\n" + laporan)

        with open(LAPORAN_TXT, "w") as f:
            f.write(laporan)

        print(f"\n[SELESAI] Data tersimpan:")
        print(f"  CSV    : {OUTPUT_CSV}")
        print(f"  Laporan: {LAPORAN_TXT}")


if __name__ == "__main__":
    main()