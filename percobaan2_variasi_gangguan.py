"""
============================================================
 percobaan2_variasi_gangguan.py  —  VERSI PERBAIKAN
 Percobaan 2 — Deteksi dengan Variasi Gangguan
 SmartBin — Pengolahan Citra Digital

 Skenario gangguan yang diuji:
   - Pencahayaan (redup, sangat terang, backlight)
   - Latar belakang (gelap, terang, bermotif)
   - Posisi objek (miring, terbalik, tertutup sebagian)
   - Multi-objek (2 objek berbeda kelas)

 Perubahan dari versi sebelumnya:

   FIX 1 — input() dihapus dari main loop.
           input() di terminal MEMBLOKIR thread utama, sehingga
           cv2.imshow() berhenti dipanggil ulang dan window kamera
           tampak hitam/freeze. Sekarang keterangan diketik LANGSUNG
           di overlay kamera (mode ketik), tanpa pernah keluar dari
           render loop.

   FIX 2 — Navigasi skenario sepenuhnya MANUAL.
           Tombol SPASI tidak lagi otomatis maju ke skenario
           berikutnya. Kamu bisa menekan SPASI berkali-kali untuk
           mengambil 3-5 sampel di skenario yang sama, baru tekan
           N untuk pindah skenario berikutnya secara sengaja.

 Prosedur:
   1. Atur kondisi lingkungan sesuai skenario yang tampil
   2. Tekan SPASI untuk ambil sampel (bisa berkali-kali)
   3. (Opsional) Tekan K untuk menambahkan keterangan pada sampel
      yang baru diambil
   4. Tekan N / P untuk pindah skenario berikutnya / sebelumnya
   5. Hasil dicatat ke CSV otomatis setiap SPASI ditekan

 Output:
   - CSV: hasil_percobaan2.csv
   - Screenshot setiap sampel

 Kontrol:
   SPASI    — ambil sampel deteksi (boleh berkali-kali per skenario)
   N        — skenario BERIKUTNYA (manual)
   P        — skenario SEBELUMNYA (manual)
   O        — ganti objek uji (mode ketik di overlay)
   K        — tambah keterangan utk sampel terakhir (mode ketik di overlay)
   Q / ESC  — keluar
============================================================
"""

import os
import cv2
import csv
import time
import threading
from datetime import datetime
from ultralytics import YOLO

# ── KONFIGURASI ────────────────────────────────────────────
MODEL_PATH   = "/home/wastemanagement2026/SmartBin/waste_results_3/weights/best.pt"
CAMERA_INDEX = 0
IMG_SIZE     = 320
CAP_WIDTH    = 640
CAP_HEIGHT   = 480

KELAS = {0: "organik", 1: "anorganik", 2: "anorganik_styrofoam"}
WARNA = {
    "organik":             (50,  200,  50),
    "anorganik":           (50,  150, 255),
    "anorganik_styrofoam": (255,  50,  50),
}

# Skenario gangguan sesuai tabel percobaan 2
SKENARIO = [
    {"no": 1,  "jenis": "Pencahayaan",   "kondisi": "Cahaya redup (< 50 lux)",           "objek_default": "Botol plastik"},
    {"no": 2,  "jenis": "Pencahayaan",   "kondisi": "Cahaya redup (< 50 lux)",           "objek_default": "Kulit buah"},
    {"no": 3,  "jenis": "Pencahayaan",   "kondisi": "Cahaya sangat terang (> 1000 lux)", "objek_default": "Botol plastik"},
    {"no": 4,  "jenis": "Pencahayaan",   "kondisi": "Cahaya sangat terang (> 1000 lux)", "objek_default": "Kulit buah"},
    {"no": 5,  "jenis": "Pencahayaan",   "kondisi": "Cahaya dari samping (backlight)",   "objek_default": "Botol plastik"},
    {"no": 6,  "jenis": "Latar Belakang","kondisi": "Latar warna gelap",                 "objek_default": "Botol plastik"},
    {"no": 7,  "jenis": "Latar Belakang","kondisi": "Latar warna gelap",                 "objek_default": "Kulit buah"},
    {"no": 8,  "jenis": "Latar Belakang","kondisi": "Latar warna terang",                "objek_default": "Botol plastik"},
    {"no": 9,  "jenis": "Latar Belakang","kondisi": "Latar warna terang",                "objek_default": "Kulit buah"},
    {"no": 10, "jenis": "Latar Belakang","kondisi": "Latar bermotif/ramai",              "objek_default": "Botol plastik"},
    {"no": 11, "jenis": "Posisi Objek",  "kondisi": "Objek miring ~45 derajat",          "objek_default": "Botol plastik"},
    {"no": 12, "jenis": "Posisi Objek",  "kondisi": "Objek miring ~45 derajat",          "objek_default": "Kulit buah"},
    {"no": 13, "jenis": "Posisi Objek",  "kondisi": "Objek terbalik",                    "objek_default": "Botol plastik"},
    {"no": 14, "jenis": "Posisi Objek",  "kondisi": "Objek sebagian tertutup",           "objek_default": "Botol plastik"},
    {"no": 15, "jenis": "Multi Objek",   "kondisi": "2 objek berbeda kelas",             "objek_default": "Botol + kulit buah"},
]

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hasil_percobaan2.csv")
CSV_HEADER = ["No", "Timestamp", "Jenis_Gangguan", "Kondisi_Gangguan",
              "Objek_Uji", "Kelas_Prediksi", "Confidence_Score_pct", "Keterangan"]


# ── CAMERA STREAM ──────────────────────────────────────────
class CameraStream:
    def __init__(self, index, w, h):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._stop.set()
        self.cap.release()


def init_csv():
    file_exists = os.path.exists(OUTPUT_CSV)
    f = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(CSV_HEADER)
        f.flush()
    return f, writer


def deteksi_frame(model, frame, names):
    results = model(frame, imgsz=IMG_SIZE, verbose=False)
    result  = results[0]

    kelas_list = []
    conf_val   = 0.0
    annotated  = frame.copy()

    if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy().astype(int)
        xyxys = result.boxes.xyxy.cpu().numpy()

        for i in range(len(confs)):
            x1,y1,x2,y2 = [int(v) for v in xyxys[i]]
            k = names.get(int(clss[i]), KELAS.get(int(clss[i]), "unknown"))
            warna = WARNA.get(k, (200,200,200))
            lbl = f"{k} {confs[i]*100:.1f}%"
            cv2.rectangle(annotated, (x1,y1), (x2,y2), warna, 2)
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1,y1-th-8),(x1+tw+4,y1), warna, -1)
            cv2.putText(annotated, lbl, (x1+2,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            kelas_list.append(f"{k}({confs[i]*100:.0f}%)")

        conf_val = float(confs.max())

    elif getattr(result, "probs", None) is not None:
        kid = int(result.probs.top1)
        conf_val = float(result.probs.top1conf)
        k = names.get(kid, KELAS.get(kid, "unknown"))
        kelas_list = [f"{k}({conf_val*100:.0f}%)"]

    kelas_str = ", ".join(kelas_list) if kelas_list else "tidak terdeteksi"
    return annotated, kelas_str, conf_val


# ============================================================
# [FIX 1] MODE KETIK DI OVERLAY — pengganti input() terminal
# ============================================================
#
# Daripada panggil input() yang membekukan jendela kamera, kita
# simpan status "sedang mengetik apa" di variabel, lalu render
# teksnya di atas frame kamera setiap loop tetap berjalan.
# Tombol huruf/angka yang ditekan ditambahkan ke buffer,
# BACKSPACE menghapus, ENTER menyelesaikan input.

TYPE_MODE_NONE    = None
TYPE_MODE_OBJEK   = "objek"
TYPE_MODE_KET     = "keterangan"

def gambar_prompt_ketik(frame, label, buffer):
    """Gambar kotak input mengetik di tengah-bawah frame, di atas frame kamera."""
    h, w = frame.shape[:2]
    teks = f"{label}: {buffer}_"
    box_h = 50
    cv2.rectangle(frame, (0, h - box_h), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, teks, (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)
    cv2.putText(frame, "ENTER=selesai | ESC=batal", (10, h - box_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def proses_tombol_ketik(key, buffer):
    """
    Update buffer teks berdasar keycode.
    Kembalikan (buffer_baru, selesai:bool, batal:bool)
    """
    if key == 13 or key == 10:        # ENTER
        return buffer, True, False
    if key == 27:                     # ESC
        return buffer, False, True
    if key == 8 or key == 127:        # BACKSPACE
        return buffer[:-1], False, False
    if 32 <= key <= 126:              # karakter cetak (ASCII)
        return buffer + chr(key), False, False
    return buffer, False, False


def main():
    print("=" * 55)
    print("  PERCOBAAN 2 — Deteksi dengan Variasi Gangguan")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    names = getattr(model, "names", None) or KELAS

    cam = CameraStream(CAMERA_INDEX, CAP_WIDTH, CAP_HEIGHT)
    time.sleep(0.5)

    f_csv, writer = init_csv()
    skenario_idx    = 0
    skenario_ini    = SKENARIO[skenario_idx]
    objek_override  = None   # None = pakai default dari skenario
    sampel_count    = 0      # jumlah sampel diambil di skenario AKTIF saat ini

    # [FIX 1] state mode ketik
    type_mode   = TYPE_MODE_NONE
    type_buffer = ""
    last_row_idx = None       # index baris CSV terakhir (untuk update keterangan via K)
    last_csv_path = OUTPUT_CSV

    print("\n[INFO] Kontrol:")
    print("  SPASI  — ambil sampel (boleh berkali-kali per skenario)")
    print("  N      — skenario BERIKUTNYA (manual)")
    print("  P      — skenario SEBELUMNYA (manual)")
    print("  O      — ganti objek uji (ketik di overlay)")
    print("  K      — tambah keterangan utk sampel terakhir (ketik di overlay)")
    print("  Q/ESC  — keluar\n")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        annotated, kelas_str, conf_val = deteksi_frame(model, frame, names)

        objek_tampil = objek_override or skenario_ini["objek_default"]
        skenario_no  = skenario_ini["no"]

        # Overlay info
        overlay = annotated.copy()
        info = [
            f"Skenario #{skenario_no}/{len(SKENARIO)}  (sampel: {sampel_count})",
            f"Jenis : {skenario_ini['jenis']}",
            f"Kondisi: {skenario_ini['kondisi'][:35]}",
            f"Objek : {objek_tampil}",
            f"Deteksi: {kelas_str[:35]}",
            f"Conf  : {conf_val*100:.1f}%",
            f"SPASI=catat | N/P=skenario | O=objek | K=ket | Q=keluar",
        ]
        cv2.rectangle(overlay, (0,0), (430, 30 + len(info)*26), (0,0,0), -1)
        y = 30
        for line in info:
            cv2.putText(overlay, line, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0,255,128) if "SPASI" in line else (220,220,220), 1)
            y += 26

        # [FIX 1] Kalau sedang dalam mode ketik, gambar prompt di bawah
        if type_mode == TYPE_MODE_OBJEK:
            gambar_prompt_ketik(overlay, "Nama objek uji", type_buffer)
        elif type_mode == TYPE_MODE_KET:
            gambar_prompt_ketik(overlay, "Keterangan sampel terakhir", type_buffer)

        cv2.imshow("Percobaan 2 — Variasi Gangguan (SPASI=catat, Q=keluar)", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:   # tidak ada tombol ditekan
            continue

        # --- Jika sedang mode ketik, semua tombol diarahkan ke buffer ---
        if type_mode is not None:
            type_buffer, selesai, batal = proses_tombol_ketik(key, type_buffer)

            if batal:
                print(f"[BATAL] mode ketik dibatalkan")
                type_mode, type_buffer = TYPE_MODE_NONE, ""
                continue

            if selesai:
                hasil = type_buffer.strip()
                if type_mode == TYPE_MODE_OBJEK:
                    objek_override = hasil if hasil else None
                    print(f"[OBJEK] diubah menjadi: {objek_override or skenario_ini['objek_default']} (default)")

                elif type_mode == TYPE_MODE_KET:
                    if last_row_idx is None:
                        print("[KET] belum ada sampel yang dicatat di skenario ini")
                    else:
                        _update_keterangan_csv(last_csv_path, last_row_idx, hasil)
                        print(f"[KET] keterangan diperbarui: '{hasil}'")

                type_mode, type_buffer = TYPE_MODE_NONE, ""
            continue   # selama mode ketik, JANGAN proses tombol lain di bawah

        # --- Tombol normal (di luar mode ketik) ---

        if key in (ord('q'), 27):
            break

        elif key == ord('n'):
            if skenario_idx < len(SKENARIO) - 1:
                skenario_idx += 1
                skenario_ini   = SKENARIO[skenario_idx]
                objek_override = None
                sampel_count   = 0
                last_row_idx   = None
                print(f"[SKENARIO] #{skenario_ini['no']}: {skenario_ini['jenis']} — {skenario_ini['kondisi']}")
            else:
                print("[SKENARIO] sudah di skenario terakhir")

        elif key == ord('p'):
            if skenario_idx > 0:
                skenario_idx -= 1
                skenario_ini   = SKENARIO[skenario_idx]
                objek_override = None
                sampel_count   = 0
                last_row_idx   = None
                print(f"[SKENARIO] #{skenario_ini['no']}: {skenario_ini['jenis']} — {skenario_ini['kondisi']}")
            else:
                print("[SKENARIO] sudah di skenario pertama")

        elif key == ord('o'):
            type_mode, type_buffer = TYPE_MODE_OBJEK, (objek_override or "")
            print("[MODE] ketik nama objek uji di jendela kamera...")

        elif key == ord('k'):
            type_mode, type_buffer = TYPE_MODE_KET, ""
            print("[MODE] ketik keterangan di jendela kamera...")

        elif key == ord(' '):
            # Pakai frame & hasil deteksi yang SUDAH ada di iterasi ini —
            # tidak perlu re-capture / re-infer (lebih cepat & konsisten
            # dengan apa yang kamu lihat di layar saat menekan SPASI).
            objek_final = objek_override or skenario_ini["objek_default"]

            writer.writerow([skenario_ini["no"],
                             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             skenario_ini["jenis"],
                             skenario_ini["kondisi"],
                             objek_final,
                             kelas_str,
                             f"{conf_val*100:.1f}",
                             ""])   # keterangan kosong dulu, bisa diisi via 'K'
            f_csv.flush()
            last_row_idx = _hitung_baris_terakhir(OUTPUT_CSV)
            sampel_count += 1

            print(f"[CATAT] #{skenario_ini['no']} sampel-{sampel_count} | "
                  f"{skenario_ini['jenis']} | {objek_final} | "
                  f"Pred={kelas_str} | Conf={conf_val*100:.1f}%")

            # Screenshot
            ss_dir = os.path.join(os.path.dirname(OUTPUT_CSV), "screenshots_p2")
            os.makedirs(ss_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(ss_dir,
                f"p2_{skenario_ini['no']}_{skenario_ini['jenis'].replace(' ','_')}_"
                f"s{sampel_count}_{ts}.jpg"), annotated)

            # [FIX 2] TIDAK ADA auto-maju skenario lagi.
            # Kamu sekarang kontrol penuh kapan pindah via tombol N/P.

    f_csv.close()
    cam.release()
    cv2.destroyAllWindows()
    print(f"\n[SELESAI] Data tersimpan di: {OUTPUT_CSV}")


# ============================================================
# Util CSV — update kolom Keterangan pada baris tertentu
# ============================================================

def _hitung_baris_terakhir(csv_path):
    """Hitung index baris data terakhir (tidak termasuk header)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1   # -1 untuk header


def _update_keterangan_csv(csv_path, row_idx, keterangan_baru):
    """
    Update kolom Keterangan pada baris ke-row_idx (1-indexed, tidak
    termasuk header). File CSV dibaca penuh, diubah, ditulis ulang —
    aman untuk file kecil seperti ini (puluhan-ratusan baris).
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    target = row_idx  # baris ke-0 = header, baris ke-row_idx = data terakhir
    if 0 < target < len(rows):
        rows[target][-1] = keterangan_baru

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()