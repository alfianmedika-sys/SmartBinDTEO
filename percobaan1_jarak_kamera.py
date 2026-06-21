"""
============================================================
 percobaan1_jarak_kamera.py
 Percobaan 1 — Jarak Optimum Kamera Deteksi
 SmartBin — Pengolahan Citra Digital
============================================================
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import sys
import json
import cv2
import csv
import time
import threading
from datetime import datetime
from ultralytics import YOLO

# ── KONFIGURASI ────────────────────────────────────────────
MODEL_PATH   = "/home/wastemanagement2026/SmartBin/waste_results_3/weights/best.pt"
CAMERA_INDEX = 0
IMG_SIZE     = 160
CAP_WIDTH    = 640
CAP_HEIGHT   = 480

MIN_CONTOUR_AREA = 1500
MAX_RATIO        = 5.0
MIN_RATIO        = 0.2
MAX_REL_SIZE     = 0.9
ROI_PADDING      = 15
MAX_OBJECTS      = 3

OBJEK_UJI = ["Botol plastik", "Kulit buah"]
KELAS = {0: "organik", 1: "anorganik", 2: "anorganik_styrofoam"}
KELAS_MAPPING = {
    "organik": "organik", "anorganik": "anorganik",
    "anorganik_styrofoam": "anorganik_styrofoam",
    "R": "anorganik", "O": "organik",
    "organic": "organik", "inorganic": "anorganik",
    "styrofoam": "anorganik_styrofoam",
}
WARNA = {
    "organik":             (50,  200,  50),
    "anorganik":           (50,  150, 255),
    "anorganik_styrofoam": (255,  50,  50),
}
WARNA_DEFAULT = (200, 200, 200)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV     = os.path.join(BASE_DIR, "hasil_percobaan1.csv")
KALIBRASI_PATH = os.path.join(BASE_DIR, "kalibrasi_jarak.json")
CSV_HEADER = ["No", "Timestamp", "Jarak_Estimasi_cm", "Status_Kalibrasi",
              "Lebar_BBox_px", "Objek_Uji", "Kelas_Prediksi",
              "Confidence_Score_pct", "Bounding_Box_Terdeteksi",
              "Jumlah_Kandidat_Kontur", "Keterangan"]


# ── KALIBRASI ──────────────────────────────────────────────
def load_kalibrasi():
    if os.path.exists(KALIBRASI_PATH):
        try:
            with open(KALIBRASI_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_kalibrasi(data):
    try:
        with open(KALIBRASI_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[KALIBRASI] gagal simpan: {e}")

def estimasi_jarak(kalibrasi, objek, lebar_bbox_px):
    ref = kalibrasi.get(objek)
    if ref is None:
        return None, "Belum dikalibrasi (tekan C)"
    if lebar_bbox_px is None or lebar_bbox_px <= 0:
        return None, "Objek tidak terdeteksi"
    jarak = (ref["jarak_cm"] * ref["lebar_px"]) / lebar_bbox_px
    return jarak, "OK"


# ── CAMERA STREAM ──────────────────────────────────────────
class CameraStream:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera index {index} tidak bisa dibuka.")
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

    def wait_first_frame(self, timeout=5.0):
        """Tunggu sampai frame pertama benar-benar ada — fix layar hitam."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.read() is not None:
                return True
            time.sleep(0.05)
        return False

    def release(self):
        self._stop.set()
        self.cap.release()


# ── CONTOUR DETECTION ──────────────────────────────────────
def cari_kandidat_box(frame):
    frame_h, frame_w = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges  = cv2.dilate(edges, kernel, iterations=2)
    edges  = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h) if h > 0 else 0
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            continue
        if w > frame_w * MAX_REL_SIZE or h > frame_h * MAX_REL_SIZE:
            continue
        candidates.append((area, (x, y, w, h)))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [box for _, box in candidates[:MAX_OBJECTS]]


# ── DETEKSI ────────────────────────────────────────────────
def deteksi_frame(model, frame, names):
    frame_h, frame_w = frame.shape[:2]
    annotated  = frame.copy()
    candidates = cari_kandidat_box(frame)

    if not candidates:
        return annotated, "tidak terdeteksi", 0.0, "Tidak", 0, None

    rois, box_coords, lebar_kontur = [], [], []
    for (x, y, bw, bh) in candidates:
        x1 = max(0, x - ROI_PADDING)
        y1 = max(0, y - ROI_PADDING)
        x2 = min(frame_w, x + bw + ROI_PADDING)
        y2 = min(frame_h, y + bh + ROI_PADDING)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        rois.append(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)))
        box_coords.append((x1, y1, x2, y2))
        lebar_kontur.append(bw)

    if not rois:
        return annotated, "tidak terdeteksi", 0.0, "Tidak", len(candidates), None

    results    = model.predict(source=rois, imgsz=IMG_SIZE, verbose=False)
    detections = []
    for result, (x1, y1, x2, y2), bw in zip(results, box_coords, lebar_kontur):
        if getattr(result, "probs", None) is not None:
            kid   = int(result.probs.top1)
            conf  = float(result.probs.top1conf)
            kelas = KELAS_MAPPING.get(names.get(kid, ""), names.get(kid, "unknown"))
            detections.append({"xyxy": (x1,y1,x2,y2), "lebar_px": bw,
                                "kelas": kelas, "confidence": conf})
        elif getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            clss  = result.boxes.cls.cpu().numpy().astype(int)
            i     = confs.argmax()
            kelas = KELAS_MAPPING.get(names.get(int(clss[i]), ""), names.get(int(clss[i]), "unknown"))
            detections.append({"xyxy": (x1,y1,x2,y2), "lebar_px": bw,
                                "kelas": kelas, "confidence": float(confs[i])})

    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        k, c = d["kelas"], d["confidence"]
        warna = WARNA.get(k, WARNA_DEFAULT)
        lbl   = f"{k} {c*100:.1f}%"
        cv2.rectangle(annotated, (x1,y1), (x2,y2), warna, 2)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), warna, -1)
        cv2.putText(annotated, lbl, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if not detections:
        return annotated, "tidak terdeteksi", 0.0, \
               "Ya (kontur, tanpa klasifikasi)", len(candidates), \
               lebar_kontur[0] if lebar_kontur else None

    best = max(detections, key=lambda d: d["confidence"])
    return annotated, best["kelas"], best["confidence"], "Ya", len(candidates), best["lebar_px"]


# ── CSV ────────────────────────────────────────────────────
def init_csv():
    exists = os.path.exists(OUTPUT_CSV)
    f      = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if not exists:
        writer.writerow(CSV_HEADER)
        f.flush()
    return f, writer

def tulis_csv(writer, f, no, jarak_cm, status_kal, lebar_px,
              objek, kelas, conf, bbox, n_kandidat, ket=""):
    writer.writerow([
        no,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{jarak_cm:.1f}" if jarak_cm is not None else "-",
        status_kal,
        f"{lebar_px}" if lebar_px is not None else "-",
        objek, kelas,
        f"{conf*100:.1f}",
        bbox, n_kandidat, ket
    ])
    f.flush()

def flush_stdin():
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


# ── MAIN ───────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  PERCOBAAN 1 — Jarak Optimum Kamera Deteksi")
    print("=" * 55)
    print(f"  Output CSV : {OUTPUT_CSV}")
    print(f"  Kalibrasi  : {KALIBRASI_PATH}")
    print(f"  Model      : {MODEL_PATH}\n")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
        return

    print("[YOLO] Memuat model...")
    model = YOLO(MODEL_PATH)
    names = getattr(model, "names", None) or KELAS
    print(f"[YOLO] Kelas: {names}")

    kalibrasi = load_kalibrasi()
    if kalibrasi:
        print(f"[KALIBRASI] Dimuat: {kalibrasi}")
    else:
        print("[KALIBRASI] Belum ada data. Tekan C untuk kalibrasi.")

    # ── Buka kamera dan tunggu frame pertama ──────────────
    print("[CAM] Membuka kamera...")
    cam = CameraStream(CAMERA_INDEX, CAP_WIDTH, CAP_HEIGHT)
    print("[CAM] Menunggu frame pertama...")
    if not cam.wait_first_frame(timeout=5):
        print("[ERROR] Kamera tidak mengirim frame dalam 5 detik. Cek koneksi kamera.")
        cam.release()
        return
    print("[CAM] Frame pertama diterima.\n")

    f_csv, writer  = init_csv()
    no_percobaan   = 1
    objek_idx      = 0
    objek_saat_ini = OBJEK_UJI[objek_idx]

    print("[INFO] Kontrol:")
    print("  SPASI  — ambil sampel deteksi")
    print("  C      — kalibrasi jarak untuk objek uji saat ini")
    print("  O      — ganti objek uji")
    print("  Q/ESC  — keluar\n")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        annotated, kelas_pred, conf_val, bbox, n_kandidat, lebar_px = \
            deteksi_frame(model, frame, names)

        jarak_cm, status_kal = estimasi_jarak(kalibrasi, objek_saat_ini, lebar_px)
        jarak_text = f"{jarak_cm:.1f} cm" if jarak_cm is not None else "-"

        # Overlay info
        display = annotated.copy()
        info_lines = [
            f"Percobaan #{no_percobaan}",
            f"Objek      : {objek_saat_ini}  [O=ganti]",
            f"Jarak (est): {jarak_text}  [{status_kal}]",
            f"Kandidat   : {n_kandidat}",
            f"Deteksi    : {kelas_pred}",
            f"Conf       : {conf_val*100:.1f}%",
            f"SPASI=catat | C=kalibrasi | Q=keluar",
        ]
        cv2.rectangle(display, (0, 0), (370, 30 + len(info_lines)*26), (0,0,0), -1)
        y = 30
        for line in info_lines:
            cv2.putText(display, line, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,128) if "SPASI" in line else (220,220,220), 1)
            y += 26

        cv2.imshow("Percobaan 1 — Jarak Kamera (SPASI=catat, C=kalibrasi, Q=keluar)", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord('o'):
            objek_idx      = (objek_idx + 1) % len(OBJEK_UJI)
            objek_saat_ini = OBJEK_UJI[objek_idx]
            print(f"[OBJEK] Diubah ke: {objek_saat_ini}")
            if objek_saat_ini not in kalibrasi:
                print(f"[INFO] '{objek_saat_ini}' belum dikalibrasi. Tekan C dulu.")

        elif key == ord('c'):
            frame_kal = cam.read()
            _, _, _, _, _, lebar_kal = deteksi_frame(model, frame_kal, names)
            if lebar_kal is None:
                print("[KALIBRASI] Gagal: objek tidak terdeteksi. Coba lagi.")
                continue
            flush_stdin()
            print(f"\n[KALIBRASI] Lebar bbox: {lebar_kal}px | Objek: '{objek_saat_ini}'")
            try:
                jarak_kal = float(input("[KALIBRASI] Masukkan jarak objek ke kamera (cm): ").strip())
            except ValueError:
                print("[KALIBRASI] Input tidak valid, dibatalkan.")
                continue
            kalibrasi[objek_saat_ini] = {"lebar_px": lebar_kal, "jarak_cm": jarak_kal}
            save_kalibrasi(kalibrasi)
            print(f"[KALIBRASI] Tersimpan: '{objek_saat_ini}' -> {lebar_kal}px @ {jarak_kal}cm\n")

        elif key == ord(' '):
            frame_s = cam.read()
            ann, kelas_s, conf_s, bbox_s, n_kand_s, lebar_s = \
                deteksi_frame(model, frame_s, names)
            jarak_s, status_s = estimasi_jarak(kalibrasi, objek_saat_ini, lebar_s)

            flush_stdin()
            ket = input("[INPUT] Keterangan (kosong=OK): ").strip()

            tulis_csv(writer, f_csv, no_percobaan, jarak_s, status_s,
                      lebar_s, objek_saat_ini, kelas_s, conf_s,
                      bbox_s, n_kand_s, ket)

            jarak_log = f"{jarak_s:.1f}cm" if jarak_s is not None else "-"
            print(f"[CATAT] #{no_percobaan} | Jarak={jarak_log} | "
                  f"Objek={objek_saat_ini} | Prediksi={kelas_s} | "
                  f"Conf={conf_s*100:.1f}% | BBox={bbox_s}")
            no_percobaan += 1

            ss_dir = os.path.join(BASE_DIR, "screenshots_p1")
            os.makedirs(ss_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"p1_{no_percobaan-1}_{jarak_log}_{ts}.jpg"
            cv2.imwrite(os.path.join(ss_dir, fname), ann)
            print(f"[SS] Tersimpan: {fname}")

    f_csv.close()
    cam.release()
    cv2.destroyAllWindows()
    print(f"\n[SELESAI] Data: {OUTPUT_CSV}")
    print(f"[SELESAI] Total sampel: {no_percobaan - 1}")


if __name__ == "__main__":
    main()