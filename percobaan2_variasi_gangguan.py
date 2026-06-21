"""
============================================================
 percobaan2_variasi_gangguan.py
 Percobaan 2 — Deteksi dengan Variasi Gangguan
 SmartBin — Pengolahan Citra Digital

 Skenario gangguan yang diuji:
   - Pencahayaan (redup, sangat terang, backlight)
   - Latar belakang (gelap, terang, bermotif)
   - Posisi objek (miring, terbalik, tertutup sebagian)
   - Multi-objek (2 objek berbeda kelas)

 Prosedur:
   1. Pilih skenario gangguan dengan tombol S
   2. Atur kondisi lingkungan sesuai skenario
   3. Tekan SPASI untuk ambil sampel
   4. Hasil dicatat ke CSV otomatis

 Output:
   - CSV: hasil_percobaan2.csv
   - Screenshot setiap sampel

 Kontrol:
   SPASI  — ambil sampel deteksi
   S      — ganti skenario gangguan
   O      — ganti objek uji
   Q/ESC  — keluar
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
    skenario_idx  = 0
    skenario_ini  = SKENARIO[skenario_idx]
    objek_override = None   # None = pakai default dari skenario

    print("\n[INFO] Kontrol:")
    print("  SPASI  — ambil sampel")
    print("  S      — skenario berikutnya")
    print("  O      — ganti objek uji manual")
    print("  Q/ESC  — keluar\n")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        annotated, kelas_str, conf_val = deteksi_frame(model, frame, names)

        objek_tampil = objek_override or skenario_ini["objek_default"]
        skenario_no  = skenario_ini["no"]

        # Overlay
        overlay = annotated.copy()
        info = [
            f"Skenario #{skenario_no}/{len(SKENARIO)}",
            f"Jenis : {skenario_ini['jenis']}",
            f"Kondisi: {skenario_ini['kondisi'][:35]}",
            f"Objek : {objek_tampil}",
            f"Deteksi: {kelas_str[:35]}",
            f"Conf  : {conf_val*100:.1f}%",
            f"SPASI=catat | S=skenario | Q=keluar",
        ]
        cv2.rectangle(overlay, (0,0), (400, 30 + len(info)*26), (0,0,0), -1)
        y = 30
        for line in info:
            cv2.putText(overlay, line, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0,255,128) if "SPASI" in line else (220,220,220), 1)
            y += 26

        cv2.imshow("Percobaan 2 — Variasi Gangguan (SPASI=catat, Q=keluar)", overlay)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord('s'):
            skenario_idx = (skenario_idx + 1) % len(SKENARIO)
            skenario_ini = SKENARIO[skenario_idx]
            objek_override = None
            print(f"[SKENARIO] #{skenario_ini['no']}: {skenario_ini['jenis']} — {skenario_ini['kondisi']}")

        elif key == ord('o'):
            inp = input("\n[INPUT] Nama objek uji (kosong=default): ").strip()
            objek_override = inp if inp else None

        elif key == ord(' '):
            frame_s = cam.read()
            ann_s, kelas_s, conf_s = deteksi_frame(model, frame_s, names)

            ket = input(f"\n[INPUT] Keterangan (kosong=OK): ").strip()
            objek_final = objek_override or skenario_ini["objek_default"]

            writer.writerow([skenario_ini["no"],
                             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             skenario_ini["jenis"],
                             skenario_ini["kondisi"],
                             objek_final,
                             kelas_s,
                             f"{conf_s*100:.1f}",
                             ket])
            f_csv.flush()

            print(f"[CATAT] #{skenario_ini['no']} | {skenario_ini['jenis']} | "
                  f"{objek_final} | Pred={kelas_s} | Conf={conf_s*100:.1f}%")

            # Screenshot
            ss_dir = os.path.join(os.path.dirname(OUTPUT_CSV), "screenshots_p2")
            os.makedirs(ss_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(ss_dir,
                f"p2_{skenario_ini['no']}_{skenario_ini['jenis'].replace(' ','_')}_{ts}.jpg"), ann_s)

            # Auto-maju ke skenario berikutnya
            skenario_idx = min(skenario_idx + 1, len(SKENARIO) - 1)
            skenario_ini = SKENARIO[skenario_idx]
            objek_override = None
            print(f"[AUTO]  Maju ke skenario #{skenario_ini['no']}: {skenario_ini['kondisi']}")

    f_csv.close()
    cam.release()
    cv2.destroyAllWindows()
    print(f"\n[SELESAI] Data tersimpan di: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
