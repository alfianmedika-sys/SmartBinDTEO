"""
============================================================
 FixCodeForWasteDetection.py
 Sistem Pemilahan Sampah Otomatis — Node Raspberry Pi 4
 Layer: AI & Decision
============================================================

 ALUR (sepenuhnya otomatis, tanpa SPASI / tanpa sensor staging):

   1. STANDBY — kamera terus mengawasi. Begitu ada objek yang
      DIAM (tidak bergerak) selama HOLD_SECONDS detik,
      sistem otomatis lanjut ke DETEKSI.

   2. DETEKSI — klasifikasi objek (organik / anorganik /
      anorganik_styrofoam). Kalau confidence kurang dari
      threshold, kembali ke STANDBY dan coba lagi otomatis.

   3. STYROFOAM_WAIT — kalau kelasnya styrofoam, beri waktu
      HOLD_SECONDS untuk memastikan tidak ada isu yang
      terjadi (cek ulang otomatis, TIDAK perlu tombol 'c').

   4. CEK_STATUS — tunggu ESP32 idle sebelum mengirim
      perintah servo.

   5. SORTING — kirim kelas ke ESP32 (via UART Serial), log,
      lalu AFTER_SORT_DELAY detik sebelum kembali ke
      STANDBY dan siap menerima objek berikutnya.

 Library: ultralytics, opencv-python, requests, pygame, pyserial
 Komunikasi ke ESP32-S3: UART via USB Type-C (/dev/ttyACM0)
============================================================
"""

import os
import platform
import csv
import threading
import time
import json
from datetime import datetime

import cv2
import requests
import serial as pyserial
import pygame
from ultralytics import YOLO


# ============================================================
# 1. KONFIGURASI GLOBAL
# ============================================================

# --- Model YOLO ---
MODEL_PATH         = "/home/wastemanagement2026/SmartBin/waste_results_3/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.70
CAMERA_INDEX       = 0
IMG_SIZE           = 160

CAP_WIDTH  = 320
CAP_HEIGHT = 240

# --- Serial UART ke ESP32 via USB Type-C ---
SERIAL_PORT    = "/dev/ttyACM0"
SERIAL_BAUD    = 115200
SERIAL_TIMEOUT = 2

# --- Flask dashboard (opsional) ---
FLASK_URL     = "http://127.0.0.1:5000"
FLASK_TIMEOUT = 3

# --- Timing siklus otomatis ---
HOLD_SECONDS        = 3.0    # objek harus diam selama ini sebelum trigger
STYROFOAM_TIMEOUT   = 30     # detik max tunggu konfirmasi styrofoam
CEK_STATUS_TIMEOUT  = 15     # detik max tunggu ESP32 idle
AFTER_SORT_DELAY    = 3.0    # detik tunggu setelah servo bergerak

# --- Contour / object detection ---
MIN_CONTOUR_AREA = 1500
MIN_RATIO        = 0.2
MAX_RATIO        = 5.0
MAX_REL_SIZE     = 0.9
MAX_OBJECTS      = 3
ROI_PADDING      = 15

# --- Stability tracker ---
STABLE_POSITION_TOLERANCE = 40

# --- Pemetaan kelas ---
KELAS = {0: "organik", 1: "anorganik", 2: "anorganik_styrofoam"}
KELAS_MAPPING = {
    "organik":             "organik",
    "anorganik":           "anorganik",
    "anorganik_styrofoam": "anorganik_styrofoam",
    "O": "organik",
    "R": "anorganik",
}

WARNA = {
    "organik":             (50,  200,  50),
    "anorganik":           (50,  150, 255),
    "anorganik_styrofoam": (255,  50,  50),
}
WARNA_DEFAULT = (200, 200, 200)

# --- Path ---
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
LOG_PATH  = os.path.join(LOG_DIR, "log_sampah.csv")


# ============================================================
# 2. CAMERA STREAM
# ============================================================

class CameraStream:
    def __init__(self, index, width, height):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Webcam index {index} tidak bisa dibuka. "
                f"Pastikan tidak ada proses lain yang menggunakan kamera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def actual_resolution(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()


# ============================================================
# 3. INFERENCE WORKER (thread terpisah)
# ============================================================

class InferenceWorker:
    def __init__(self, model):
        self.model   = model
        self._result = None
        self._lock   = threading.Lock()
        self._input  = None
        self._event  = threading.Event()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            if not self._event.wait(timeout=0.05):
                continue
            self._event.clear()
            frame, fw, fh = self._input
            result = self._infer(frame, fw, fh)
            with self._lock:
                self._result = result

    def _infer(self, frame, fw, fh):
        results = self.model(frame, imgsz=IMG_SIZE, verbose=False)
        result  = results[0]
        dets = []

        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            clss  = result.boxes.cls.cpu().numpy().astype(int)
            xyxys = result.boxes.xyxy.cpu().numpy()
            names = getattr(self.model, "names", KELAS)
            for i in range(len(confs)):
                raw   = names.get(int(clss[i]), KELAS.get(int(clss[i]), "unknown"))
                kelas = KELAS_MAPPING.get(raw, raw)
                dets.append({
                    "kelas":      kelas,
                    "kelas_asli": raw,
                    "confidence": float(confs[i]),
                    "xyxy":       xyxys[i],
                })

        elif getattr(result, "probs", None) is not None:
            names = getattr(self.model, "names", KELAS)
            kid   = int(result.probs.top1)
            conf  = float(result.probs.top1conf)
            raw   = names.get(kid, KELAS.get(kid, "unknown"))
            kelas = KELAS_MAPPING.get(raw, raw)
            dets.append({
                "kelas":      kelas,
                "kelas_asli": raw,
                "confidence": conf,
                "xyxy":       None,
            })

        return dets if dets else None

    def submit(self, frame, fw, fh):
        self._input = (frame.copy(), fw, fh)
        self._event.set()

    def get_result(self):
        with self._lock:
            r = self._result
            self._result = None
            return r

    def detect_blocking(self, frame, fw, fh):
        """Blocking inference — dipakai di state DETEKSI."""
        dets = self._infer(frame, fw, fh)
        if not dets:
            return None, 0.0
        best = max(dets, key=lambda d: d["confidence"])
        return best["kelas"], best["confidence"]

    def stop(self):
        self._stop.set()


# ============================================================
# 4. CONTOUR DETECTION
# ============================================================

def cari_kandidat_box(frame):
    frame_h, frame_w = frame.shape[:2]
    SCALE  = 0.5
    small  = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE,
                        interpolation=cv2.INTER_NEAREST)
    gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (3, 3), 0)
    edges  = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area_scaled = MIN_CONTOUR_AREA * (SCALE ** 2)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_scaled:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h) if h > 0 else 0
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            continue
        sw, sh = small.shape[1], small.shape[0]
        if w > sw * MAX_REL_SIZE or h > sh * MAX_REL_SIZE:
            continue
        x2 = int(x / SCALE); y2 = int(y / SCALE)
        w2 = int(w / SCALE); h2 = int(h / SCALE)
        candidates.append((area, (x2, y2, w2, h2)))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [box for _, box in candidates[:MAX_OBJECTS]]


# ============================================================
# 5. STABILITY TRACKER
# ============================================================

class StabilityTracker:
    def __init__(self):
        self._last_boxes = None
        self._stable_since = None

    def update(self, frame):
        boxes = cari_kandidat_box(frame)
        now   = time.time()

        if not boxes:
            self._last_boxes   = None
            self._stable_since = None
            return False, 0.0

        if self._last_boxes is None:
            self._last_boxes   = boxes
            self._stable_since = now
            return False, 0.0

        # Cek apakah box utama bergeser signifikan
        moved = False
        if len(boxes) != len(self._last_boxes):
            moved = True
        else:
            bx, by, bw, bh = boxes[0]
            lx, ly, lw, lh = self._last_boxes[0]
            if (abs(bx - lx) > STABLE_POSITION_TOLERANCE or
                    abs(by - ly) > STABLE_POSITION_TOLERANCE):
                moved = True

        if moved:
            self._stable_since = now

        self._last_boxes = boxes
        elapsed  = now - self._stable_since
        progress = min(elapsed / HOLD_SECONDS, 1.0)
        return elapsed >= HOLD_SECONDS, progress

    def reset(self):
        self._last_boxes   = None
        self._stable_since = None


# ============================================================
# 6. SERIAL CLIENT — UART ke ESP32
# ============================================================

class SerialClient:
    def __init__(self, port, baud, timeout):
        self.port    = port
        self.baud    = baud
        self.timeout = timeout
        self._ser    = None
        self._lock   = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            self._ser = pyserial.Serial(
                self.port, self.baud,
                timeout=self.timeout
            )
            time.sleep(2)
            print(f"[SERIAL] terhubung ke {self.port} @ {self.baud}")
        except pyserial.SerialException as e:
            print(f"[SERIAL] gagal konek: {e}")
            self._ser = None

    def health(self):
        return self._ser is not None and self._ser.is_open

    def get_status(self):
        """
        Kirim perintah cek status ke ESP32 dan baca responnya.
        ESP32 merespons dengan JSON: {"status": "standby", "sedang_bergerak": false}
        """
        if not self.health():
            return None
        try:
            cmd = json.dumps({"cmd": "status"}) + "\n"
            with self._lock:
                self._ser.write(cmd.encode("utf-8"))
                self._ser.flush()
                # Baca respons dengan timeout
                waktu_mulai = time.time()
                buffer = ""
                while time.time() - waktu_mulai < self.timeout:
                    if self._ser.in_waiting:
                        c = self._ser.read().decode("utf-8", errors="ignore")
                        if c == "\n":
                            break
                        buffer += c
                    else:
                        time.sleep(0.01)
            if buffer:
                return json.loads(buffer)
        except Exception as e:
            print(f"[SERIAL] gagal get_status: {e}")
            self._ser = None
        return None

    def kirim_servo(self, kelas):
        """Kirim perintah servo ke ESP32 via UART."""
        if not self.health():
            print("[SERIAL] tidak terhubung, skip kirim servo")
            return False
        payload = json.dumps({"kelas": kelas}) + "\n"
        try:
            with self._lock:
                self._ser.write(payload.encode("utf-8"))
                self._ser.flush()
            print(f"[SERIAL] servo terkirim: {payload.strip()}")
            return True
        except pyserial.SerialException as e:
            print(f"[SERIAL] gagal kirim servo: {e}")
            self._ser = None
            return False

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
            print("[SERIAL] port ditutup")


# ============================================================
# 7. AUDIO PLAYER
# ============================================================

class AudioPlayer:
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        os.makedirs(audio_dir, exist_ok=True)
        try:
            pygame.mixer.init()
            self.ready = True
            print(f"[AUDIO] inisialisasi berhasil, direktori: {audio_dir}")
        except Exception as e:
            print(f"[AUDIO] mixer gagal init: {e}")
            self.ready = False

    def play(self, nama_file, block=True):
        if not self.ready:
            return
        path = os.path.join(self.audio_dir, nama_file)
        if not os.path.exists(path):
            print(f"[AUDIO] file tidak ada: {path}")
            return
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            if block:
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        except Exception as e:
            print(f"[AUDIO] gagal putar {nama_file}: {e}")


# ============================================================
# 8. LOGGER CSV
# ============================================================

class Logger:
    HEADER = ["timestamp", "kelas", "kelas_asli", "confidence", "status"]

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        if not os.path.exists(path):
            self._tulis(self.HEADER)
        print(f"[LOG] direktori log: {os.path.dirname(path)}")

    def _tulis(self, row):
        try:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"[LOG] gagal tulis: {e}")

    def log(self, kelas, conf, status, kelas_asli=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._tulis([ts, kelas or "none", kelas_asli or "-",
                     f"{conf:.3f}", status])
        print(f"[LOG] {ts} | {kelas} | conf={conf:.3f} | {status}")


# ============================================================
# 9. OVERLAY HELPER
# ============================================================

def gambar_deteksi(frame, dets):
    if not dets:
        return
    for d in dets:
        if d["xyxy"] is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        warna = WARNA.get(d["kelas"], WARNA_DEFAULT)
        label = f"{d['kelas']} {d['confidence']*100:.0f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), warna, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ============================================================
# 10. FLASK INTEGRATION (opsional)
# ============================================================

_stream_lock   = threading.Lock()
_stream_latest = None
_stream_thread = None

def _stream_worker():
    global _stream_latest
    while True:
        with _stream_lock:
            data = _stream_latest
            _stream_latest = None
        if data is not None:
            try:
                requests.post(
                    f"{FLASK_URL}/api/frame",
                    data=data,
                    headers={"Content-Type": "image/jpeg"},
                    timeout=0.8,
                )
            except Exception:
                pass
        else:
            time.sleep(0.02)

def _start_stream_thread():
    global _stream_thread
    if _stream_thread is None or not _stream_thread.is_alive():
        _stream_thread = threading.Thread(target=_stream_worker, daemon=True)
        _stream_thread.start()

_frame_skip = 0

def kirim_frame_ke_flask(frame):
    global _frame_skip, _stream_latest
    _frame_skip += 1
    if _frame_skip % 6 != 0:
        return
    try:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        with _stream_lock:
            _stream_latest = buf.tobytes()
    except Exception:
        pass

def kirim_deteksi_ke_flask(class_name, confidence, routed_to):
    try:
        requests.post(
            f"{FLASK_URL}/api/detection",
            json={"class_name": class_name,
                  "confidence": round(confidence, 4),
                  "routed_to":  routed_to},
            timeout=FLASK_TIMEOUT,
        )
        print(f"[FLASK] deteksi terkirim: {class_name} -> {routed_to}")
    except Exception as e:
        print(f"[FLASK] gagal kirim: {e}")


# ============================================================
# 11. STATE MACHINE UTAMA
# ============================================================

class WasteSorter:
    def __init__(self):
        print(f"[YOLO] memuat model: {MODEL_PATH} ...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
        self.model  = YOLO(MODEL_PATH)
        self.names  = getattr(self.model, "names", None) or KELAS
        print(f"[YOLO] model dimuat. Kelas: {self.names}")

        self.audio  = AudioPlayer(AUDIO_DIR)
        self.logger = Logger(LOG_PATH)

        self.cam = CameraStream(CAMERA_INDEX, CAP_WIDTH, CAP_HEIGHT)
        time.sleep(0.5)
        self.frame_w, self.frame_h = self.cam.actual_resolution()
        print(f"[CAM] resolusi aktual: {self.frame_w}x{self.frame_h}")

        self.worker    = InferenceWorker(self.model)
        self.stability = StabilityTracker()
        self.serial    = SerialClient(SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT)
        self.running   = True
        self._ctx      = {}
        self._last_dets    = None
        self._frame_cnt    = 0
        self._detect_every = 5
        _start_stream_thread()

    # ---------- util ----------

    def baca_frame(self):
        return self.cam.read()

    def tampilkan(self, frame, label=""):
        if frame is None:
            return -1

        result = self.worker.get_result()
        if result is not None:
            self._last_dets = result

        if self._last_dets:
            gambar_deteksi(frame, self._last_dets)

        self._frame_cnt += 1
        if self._frame_cnt % self._detect_every == 0:
            self.worker.submit(frame, self.frame_w, self.frame_h)

        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
            cv2.putText(frame, label, (8, th + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Pemilahan Sampah - Smart Bin (q=keluar)", frame)
        kirim_frame_ke_flask(frame)
        return cv2.waitKey(1) & 0xFF

    # ---------- STATE: STANDBY ----------

    def standby(self):
        frame = self.baca_frame()
        if frame is None:
            return "STANDBY"

        stabil, progress = self.stability.update(frame)

        # Progress bar
        h, w = frame.shape[:2]
        bw, bh = 240, 14
        bx = (w - bw) // 2
        by = h - 50
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        fill_w = int(bw * progress)
        warna_bar = (50, 200, 50) if progress >= 1.0 else (50, 150, 255)
        cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bh), warna_bar, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (220, 220, 220), 1)
        cv2.putText(frame, "Menunggu objek diam...", (bx, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        label = f"STANDBY - letakkan & diamkan objek ({HOLD_SECONDS:.0f}s)"
        key   = self.tampilkan(frame, label)

        if key == ord('q'):
            self.running = False
            return None

        if stabil:
            print(f"[TRIGGER] objek diam {HOLD_SECONDS:.0f}s -> mulai deteksi")
            self.stability.reset()
            return "DETEKSI"

        return "STANDBY"

    # ---------- STATE: DETEKSI ----------

    def deteksi(self):
        frame = self.baca_frame()
        if frame is None:
            return "STANDBY"

        # Majority vote 3x
        N_VOTES   = 3
        votes     = []
        conf_list = []

        for i in range(N_VOTES):
            _f = self.baca_frame()
            f = _f if _f is not None else frame
            k, c = self.worker.detect_blocking(f, self.frame_w, self.frame_h)
            self.tampilkan(f, f"DETEKSI ({i+1}/{N_VOTES}): {k} ({c:.2f})" if k else f"DETEKSI ({i+1}/{N_VOTES}): -")
            if k is not None and c >= CONFIDENCE_THRESHOLD:
                votes.append(k)
                conf_list.append(c)

        if not votes:
            print("[DETEKSI] kurang yakin -> kembali standby")
            self.audio.play("scan_ulang.mp3", block=False)
            self.logger.log(None, 0.0, "gagal_confidence")
            self.stability.reset()
            return "STANDBY"

        from collections import Counter
        kelas = Counter(votes).most_common(1)[0][0]
        conf  = sum(c for k, c in zip(votes, conf_list) if k == kelas) / votes.count(kelas)

        print(f"[DETEKSI] vote={votes} -> kelas={kelas}, conf={conf:.2f}")
        self._ctx = {"kelas": kelas, "conf": conf}

        if kelas == "anorganik_styrofoam":
            self.stability.reset()
            return "STYROFOAM_WAIT"
        return "CEK_STATUS"

    # ---------- STATE: STYROFOAM_WAIT ----------

    def styrofoam_wait(self):
        frame = self.baca_frame()
        if frame is None:
            return "STANDBY"

        conf = self._ctx.get("conf", 0.0)

        if "_styro_mulai" not in self._ctx:
            self._ctx["_styro_mulai"] = time.time()

        elapsed = time.time() - self._ctx["_styro_mulai"]
        sisa    = max(0, int(STYROFOAM_TIMEOUT - elapsed))

        label = (f"STYROFOAM ({conf:.2f}) - "
                 f"menunggu konfirmasi ({sisa}s)")
        key = self.tampilkan(frame, label)

        if key == ord('q'):
            self.running = False
            return None

        if elapsed >= STYROFOAM_TIMEOUT:
            print("[STYROFOAM] timeout -> lanjut sorting")
            self._ctx.pop("_styro_mulai", None)
            return "CEK_STATUS"

        # Cek ulang otomatis setiap 5 detik
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            k, c = self.worker.detect_blocking(
                frame, self.frame_w, self.frame_h)
            if k and k != "anorganik_styrofoam":
                print(f"[STYROFOAM] re-check -> {k}, lanjut sorting")
                self._ctx["kelas"] = k
                self._ctx["conf"]  = c
                self._ctx.pop("_styro_mulai", None)
                return "CEK_STATUS"

        return "STYROFOAM_WAIT"

    # ---------- STATE: CEK_STATUS ----------

    def cek_status(self):
        if "_cek_mulai" not in self._ctx:
            self._ctx["_cek_mulai"] = time.time()

        status = self.serial.get_status()

        if status is None:
            # ESP32 tidak merespons — lanjut saja
            print("[CEK_STATUS] ESP32 tidak merespons, lanjut sorting")
            self._ctx.pop("_cek_mulai", None)
            return "SORTING"

        if status.get("sedang_bergerak", False):
            elapsed = time.time() - self._ctx["_cek_mulai"]
            if elapsed > CEK_STATUS_TIMEOUT:
                print(f"[CEK_STATUS] timeout {CEK_STATUS_TIMEOUT}s, lanjut")
                self._ctx.pop("_cek_mulai", None)
                return "SORTING"
            print("[CEK_STATUS] ESP32 sibuk, tunggu...")
            time.sleep(1)
            return "CEK_STATUS"

        self._ctx.pop("_cek_mulai", None)
        return "SORTING"

    # ---------- STATE: SORTING ----------

    def sorting(self):
        kelas = self._ctx["kelas"]
        conf  = self._ctx["conf"]

        audio_map = {
            "organik":             "organik.mp3",
            "anorganik":           "anorganik.mp3",
            "anorganik_styrofoam": "styrofoam.mp3",
        }
        self.audio.play(audio_map.get(kelas, "organik.mp3"), block=False)

        # Tentukan tong tujuan
        routed_to = "organik" if kelas == "organik" else "anorganik"

        print(f"[SORTING] mengirim kelas: {kelas} -> {routed_to}")
        ok = self.serial.kirim_servo(kelas)

        if not ok:
            print("[SORTING] perintah servo gagal")
            self.logger.log(kelas, conf, "gagal_servo")
            return "STANDBY"

        self.logger.log(kelas, conf, "sukses")
        kirim_deteksi_ke_flask(kelas, conf, routed_to)
        self.audio.play("terima_kasih.mp3", block=False)

        # Delay sebelum siap menerima objek berikutnya
        print(f"[SORTING] menunggu {AFTER_SORT_DELAY:.0f}s...")
        mulai = time.time()
        while time.time() - mulai < AFTER_SORT_DELAY:
            frame = self.baca_frame()
            sisa  = max(0.0, AFTER_SORT_DELAY - (time.time() - mulai))
            self.tampilkan(frame,
                           f"Sorting selesai. Siap lagi dalam {sisa:.0f}s")

        self.stability.reset()
        return "STANDBY"

    # ---------- MAIN LOOP ----------

    def run(self):
        if not self.serial.health():
            print(f"[INFO] ESP32 ({SERIAL_PORT}) belum terhubung. "
                  f"Deteksi tetap berjalan, perintah servo akan dilewati.")
        else:
            print("[INFO] ESP32 terhubung via UART!")

        state = "STANDBY"
        handlers = {
            "STANDBY":        self.standby,
            "DETEKSI":        self.deteksi,
            "STYROFOAM_WAIT": self.styrofoam_wait,
            "CEK_STATUS":     self.cek_status,
            "SORTING":        self.sorting,
        }

        print("=" * 55)
        print("  SmartBin - Sistem Pemilahan Otomatis (UART)")
        print(f"  Model  : {MODEL_PATH}")
        print(f"  Serial : {SERIAL_PORT} @ {SERIAL_BAUD}")
        print(f"  HOLD={HOLD_SECONDS}s | CONF_THR={CONFIDENCE_THRESHOLD}")
        print("  Tekan Q untuk keluar")
        print("=" * 55)

        while self.running:
            handler = handlers.get(state)
            if handler is None:
                break
            next_state = handler()
            if next_state is None:
                break
            state = next_state

        self.shutdown()

    def shutdown(self):
        print("[SHUTDOWN] membersihkan...")
        try:
            self.serial.kirim_servo("tengah")
        except Exception:
            pass
        try:
            self.serial.close()
        except Exception:
            pass
        try:
            self.worker.stop()
        except Exception:
            pass
        try:
            self.cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            pygame.mixer.quit()
        except Exception:
            pass


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        WasteSorter().run()
    except KeyboardInterrupt:
        print("\n[EXIT] dihentikan pengguna")
    except Exception as e:
        print(f"[FATAL] {e}")
        raise