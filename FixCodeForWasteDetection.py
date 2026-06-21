"""
============================================================
============================================================
 WasteSorter.py
 Sistem Pemilahan Sampah Otomatis - Node Raspberry Pi 4
 Layer: AI & Decision
 Versi: Disesuaikan untuk ESP32-S3 dengan endpoint /servo
============================================================
"""

import os
import csv
import queue
import threading
import time
from datetime import datetime

import cv2
import requests
import pygame
from ultralytics import YOLO


# ============================================================
# 1. KONFIGURASI GLOBAL
# ============================================================

# --- YOLO ---
MODEL_PATH           = "/home/wastemanagement2026/SmartBin/waste_results_3/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.70
CAMERA_INDEX         = 0
IMG_SIZE             = 160

CAP_WIDTH  = 640
CAP_HEIGHT = 480

# --- ESP32 (mode Access Point) ---
ESP32_IP      = "192.168.4.1"
ESP32_PORT    = 80
ESP32_TIMEOUT = 5

FLASK_TIMEOUT = 3

# --- Threshold & timeout ---
STYROFOAM_TIMEOUT     = 30      # detik

# --- Path (gunakan direktori lokal jika SD card tidak tersedia) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "log_sampah.csv")

# Mode pemicu STANDBY: "sensor" atau "manual"
MODE_TRIGGER = "manual"

# --- Contour detection ---
MIN_CONTOUR_AREA = 1500
MAX_RATIO        = 5.0
MIN_RATIO        = 0.2
MAX_REL_SIZE     = 0.9
ROI_PADDING      = 15
MAX_OBJECTS      = 3

# ============================================================
# 2. PEMETAAN KELAS (sesuaikan dengan model YOLO Anda)
# ============================================================

# Pemetaan id kelas -> nama (sesuaikan dengan hasil training Anda)
# Jika model Anda menggunakan label "R" untuk anorganik, sesuaikan di sini
KELAS = {
    0: "organik", 
    1: "anorganik", 
    2: "anorganik_styrofoam"
}

# Mapping untuk kelas yang mungkin keluar dari model (jika pakai nama langsung)
# Tambahkan mapping untuk semua kemungkinan nama kelas dari model Anda
KELAS_MAPPING = {
    "organik": "organik",
    "anorganik": "anorganik", 
    "anorganik_styrofoam": "anorganik",
    "R": "anorganik",  # Jika model mengeluarkan "R" untuk anorganik
    "O": "organik",    # Jika model mengeluarkan "O" untuk organik
    "organic": "organik",
    "inorganic": "anorganik",
    "styrofoam": "anorganik",
    # Tambahkan mapping lain sesuai kebutuhan
}

# ============================================================
# 3. CAMERA STREAM (thread terpisah)
# ============================================================

class CameraStream:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Webcam index {index} tidak bisa dibuka. Coba ganti CAMERA_INDEX."
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
# 4. CONTOUR DETECTION
# ============================================================

def cari_kandidat_box(frame):
    frame_h, frame_w = frame.shape[:2]

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges  = cv2.dilate(edges, kernel, iterations=2)
    edges  = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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


# ============================================================
# 5. INFERENCE WORKER
# ============================================================

def _jalankan_inferensi(frame, model, names, frame_w, frame_h):
    candidates = cari_kandidat_box(frame)
    if not candidates:
        return []

    rois       = []
    box_coords = []

    for (x, y, bw, bh) in candidates:
        x1 = max(0, x - ROI_PADDING)
        y1 = max(0, y - ROI_PADDING)
        x2 = min(frame_w, x + bw + ROI_PADDING)
        y2 = min(frame_h, y + bh + ROI_PADDING)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        roi_resized = cv2.resize(
            roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
        )
        rois.append(roi_resized)
        box_coords.append((x1, y1, x2, y2))

    if not rois:
        return []

    all_results = model.predict(source=rois, imgsz=IMG_SIZE, verbose=False)

    detections = []
    for result, (x1, y1, x2, y2) in zip(all_results, box_coords):
        # Model klasifikasi
        if getattr(result, "probs", None) is not None:
            kelas_id   = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            kelas      = names.get(kelas_id, KELAS.get(kelas_id, "unknown"))
            
            # Normalisasi kelas menggunakan mapping
            kelas_normal = KELAS_MAPPING.get(kelas, kelas)
            
            detections.append({
                "xyxy":       (x1, y1, x2, y2),
                "kelas":      kelas_normal,
                "kelas_asli": kelas,
                "confidence": confidence,
            })
            continue

        # Model deteksi objek
        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            confs    = result.boxes.conf.cpu().numpy()
            clss     = result.boxes.cls.cpu().numpy().astype(int)
            i        = confs.argmax()
            kelas_id = int(clss[i])
            confidence = float(confs[i])
            kelas    = names.get(kelas_id, KELAS.get(kelas_id, "unknown"))
            
            # Normalisasi kelas menggunakan mapping
            kelas_normal = KELAS_MAPPING.get(kelas, kelas)
            
            detections.append({
                "xyxy":       (x1, y1, x2, y2),
                "kelas":      kelas_normal,
                "kelas_asli": kelas,
                "confidence": confidence,
            })

    return detections


class InferenceWorker:
    def __init__(self, model_path, imgsz):
        print(f"[YOLO] memuat model: {model_path}")
        self.model = YOLO(model_path)
        self.names = getattr(self.model, "names", None) or KELAS
        self.imgsz = imgsz

        self.input_q  = queue.Queue(maxsize=1)
        self.output_q = queue.Queue(maxsize=1)
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                frame, fw, fh = self.input_q.get(timeout=0.1)
            except queue.Empty:
                continue
            result = _jalankan_inferensi(frame, self.model, self.names, fw, fh)
            try:
                self.output_q.get_nowait()
            except queue.Empty:
                pass
            self.output_q.put(result)

    def submit(self, frame, frame_w, frame_h):
        try:
            self.input_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.input_q.put_nowait((frame.copy(), frame_w, frame_h))
        except queue.Full:
            pass

    def get_result(self):
        try:
            return self.output_q.get_nowait()
        except queue.Empty:
            return None

    def detect_blocking(self, frame, frame_w, frame_h):
        detections = _jalankan_inferensi(
            frame, self.model, self.names, frame_w, frame_h
        )
        if not detections:
            return None, 0.0
        best = max(detections, key=lambda d: d["confidence"])
        return best["kelas"], best["confidence"]

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


# ============================================================
# 6. KLIEN ESP32
# ============================================================

class ESP32Client:
    def __init__(self, ip, port, timeout):
        self.base    = f"http://{ip}:{port}"
        self.timeout = timeout

    def health(self):
        try:
            r = requests.get(f"{self.base}/health", timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def get_status(self):
        try:
            r = requests.get(f"{self.base}/status", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"[ESP32] gagal GET /status: {e}")
            return None

    def post_servo(self, kelas):
        # Validasi kelas sebelum dikirim
        valid_classes = ["organik", "anorganik", "tengah", "netral"]
        
        if kelas not in valid_classes:
            print(f"[ESP32] kelas '{kelas}' tidak valid, gunakan default 'tengah'")
            kelas_send = "tengah"
        else:
            kelas_send = kelas

        try:
            r = requests.post(
                f"{self.base}/servo",
                json={"kelas": kelas_send},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"[ESP32] gagal POST /servo ({kelas_send}): {e}")
            return None


# ============================================================
# 7. PEMUTAR AUDIO
# ============================================================

class AudioPlayer:
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        # Buat direktori audio jika belum ada
        try:
            os.makedirs(audio_dir, exist_ok=True)
        except Exception as e:
            print(f"[AUDIO] gagal buat direktori: {e}")
        
        try:
            pygame.mixer.init()
            self.ready = True
            print(f"[AUDIO] inisialisasi berhasil, direktori: {audio_dir}")
        except Exception as e:
            print(f"[AUDIO] mixer gagal init: {e}")
            self.ready = False

    def play(self, nama_file, block=True):
        if not self.ready:
            print(f"[AUDIO] (lewati) {nama_file}")
            return
        
        path = os.path.join(self.audio_dir, nama_file)
        if not os.path.exists(path):
            print(f"[AUDIO] file tidak ada: {path}")
            # Coba cari file alternatif
            if nama_file == "organik.mp3" and os.path.exists(os.path.join(self.audio_dir, "anorganik.mp3")):
                path = os.path.join(self.audio_dir, "anorganik.mp3")
                print(f"[AUDIO] menggunakan alternatif: anorganik.mp3")
            elif nama_file == "anorganik.mp3" and os.path.exists(os.path.join(self.audio_dir, "organik.mp3")):
                path = os.path.join(self.audio_dir, "organik.mp3")
                print(f"[AUDIO] menggunakan alternatif: organik.mp3")
            else:
                # Buat file audio dummy jika tidak ada
                self._create_dummy_audio(path)
                if not os.path.exists(path):
                    return
        
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            if block:
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        except Exception as e:
            print(f"[AUDIO] gagal putar {nama_file}: {e}")

    def _create_dummy_audio(self, path):
        """Buat file audio dummy jika tidak ada (hanya untuk testing)"""
        try:
            # Buat file kosong sebagai placeholder
            with open(path, 'w') as f:
                f.write("# Dummy audio file - ganti dengan file MP3 asli")
            print(f"[AUDIO] dummy file dibuat: {path}")
        except Exception as e:
            print(f"[AUDIO] gagal buat dummy: {e}")


# ============================================================
# 8. LOGGER CSV
# ============================================================

class Logger:
    HEADER = ["timestamp", "kelas", "kelas_asli", "confidence", "status"]

    def __init__(self, path):
        self.path = path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"[LOG] direktori log: {os.path.dirname(path)}")
        except Exception as e:
            print(f"[LOG] gagal buat direktori: {e}")
            # Fallback ke direktori saat ini
            self.path = "log_sampah.csv"
            print(f"[LOG] menggunakan fallback: {self.path}")
        
        if not os.path.exists(self.path):
            self._write_row(self.HEADER)

    def _write_row(self, row):
        try:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"[LOG] gagal tulis: {e}")

    def log(self, kelas, confidence, status, kelas_asli=""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_row([ts, kelas, kelas_asli, f"{confidence:.2f}", status])
        print(f"[LOG] {ts} | {kelas} | conf={confidence:.2f} | {status}")


# ============================================================
# 9. PEMETAAN KEPUTUSAN
# ============================================================

def get_audio_file(kelas):
    """Pilih file audio berdasarkan kelas"""
    audio_files = {
        "organik": "organik.mp3",
        "anorganik": "anorganik.mp3",
        "anorganik_styrofoam": "anorganik.mp3",
    }
    return audio_files.get(kelas, "organik.mp3")


# ============================================================
# 10. OVERLAY HELPER
# ============================================================

WARNA_KELAS = {
    "organik":             ( 50, 200,  50),
    "anorganik":           ( 50, 150, 255),
    "anorganik_styrofoam": (255,  50,  50),
}
WARNA_TIDAK_YAKIN = (0, 0, 255)
WARNA_DEFAULT     = (200, 200, 200)

def gambar_deteksi(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        conf  = d["confidence"]
        kelas = d["kelas"]
        kelas_asli = d.get("kelas_asli", kelas)

        if conf >= CONFIDENCE_THRESHOLD:
            warna = WARNA_KELAS.get(kelas, WARNA_DEFAULT)
            label = f"{kelas} {conf * 100:.1f}%"
            if kelas_asli != kelas:
                label += f" ({kelas_asli})"
        else:
            warna = WARNA_TIDAK_YAKIN
            label = f"Tidak Yakin {conf * 100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), warna, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ============================================================

# ============================================================
# 10b. INTEGRASI FLASK - kirim deteksi ke dashboard
# ============================================================

FLASK_URL     = "http://127.0.0.1:5000"
FLASK_TIMEOUT = 3

def kirim_deteksi_ke_flask(class_name: str, confidence: float, routed_to: str):
    """
    Kirim hasil deteksi YOLO ke Flask API agar muncul di dashboard.
    Dipanggil dari sorting() setelah servo berhasil digerakkan.
    Gagal kirim tidak menghentikan program - hanya di-log ke terminal.
    """
    try:
        payload = {
            "class_name": class_name,
            "confidence":  round(confidence, 4),
            "routed_to":   routed_to,
        }
        r = requests.post(
            f"{FLASK_URL}/api/detection",
            json=payload,
            timeout=FLASK_TIMEOUT,
        )
        if r.status_code == 200:
            print(f"[FLASK] deteksi terkirim: {class_name} → {routed_to}")
        else:
            print(f"[FLASK] respons tidak OK: {r.status_code} {r.text}")
    except requests.RequestException as e:
        print(f"[FLASK] gagal kirim deteksi: {e}")

# 11. STATE MACHINE UTAMA
# ============================================================

class WasteSorter:
    def __init__(self):
        self.esp32  = ESP32Client(ESP32_IP, ESP32_PORT, ESP32_TIMEOUT)
        self.audio  = AudioPlayer(AUDIO_DIR)
        self.logger = Logger(LOG_PATH)

        # Kamera di thread terpisah
        self.cam = CameraStream(CAMERA_INDEX, CAP_WIDTH, CAP_HEIGHT)
        time.sleep(0.5)
        self.frame_w, self.frame_h = self.cam.actual_resolution()
        print(f"[CAM] resolusi aktual: {self.frame_w}x{self.frame_h}")

        # YOLO worker di thread terpisah
        self.worker = InferenceWorker(MODEL_PATH, IMG_SIZE)

        self.running     = True
        self._ctx        = {}
        self._last_dets  = []
        self._frame_cnt  = 0
        self._detect_every = 3

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
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(frame, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
            cv2.putText(frame, label, (8, th + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Pemilahan Sampah - Smart Bin (q=keluar)", frame)
        return cv2.waitKey(1) & 0xFF

    def standby(self):
        frame = self.baca_frame()
        key   = self.tampilkan(frame, "STANDBY - SPASI mulai | q keluar")

        if key == ord('q'):
            self.running = False
            return None

        if MODE_TRIGGER == "manual":
            if key == ord(' '):
                return "DETEKSI"

        return "STANDBY"

    def deteksi(self):
        frame = self.baca_frame()
        if frame is None:
            return "STANDBY"

        kelas, conf = self.worker.detect_blocking(
            frame, self.frame_w, self.frame_h
        )
        self.tampilkan(frame, f"DETEKSI: {kelas} ({conf:.2f})")

        # Log semua deteksi untuk debugging
        detections = self.worker.get_result()
        if detections:
            print(f"[DETEKSI] semua hasil: {[(d['kelas'], d['kelas_asli'], d['confidence']) for d in detections]}")

        if kelas is None or conf < CONFIDENCE_THRESHOLD:
            print(f"[DETEKSI] kurang yakin (kelas={kelas}, conf={conf:.2f}) -> scan ulang")
            self.audio.play("scan_ulang.mp3")
            self.logger.log(kelas or "none", conf, "gagal_confidence")
            return "STANDBY"

        print(f"[DETEKSI] kelas={kelas}, conf={conf:.2f}")
        self._ctx = {"kelas": kelas, "conf": conf}

        if kelas == "anorganik_styrofoam":
            return "STYROFOAM_WAIT"
        return "CEK_STATUS"

    def styrofoam_wait(self):
        self.audio.play("styrofoam.mp3")
        print(f"[STYROFOAM] menunggu sisa dibuang "
              f"(timeout {STYROFOAM_TIMEOUT}s, tekan 'c' untuk konfirmasi)")

        mulai = time.time()
        while time.time() - mulai < STYROFOAM_TIMEOUT:
            frame = self.baca_frame()
            sisa  = int(STYROFOAM_TIMEOUT - (time.time() - mulai))
            key   = self.tampilkan(
                frame, f"Buang sisa makanan... ({sisa}s) tekan 'c'"
            )
            if key == ord('c'):
                print("[STYROFOAM] dikonfirmasi, lanjut sorting")
                return "CEK_STATUS"
            if key == ord('q'):
                self.running = False
                return None

        print("[STYROFOAM] timeout, batalkan")
        self.logger.log("anorganik_styrofoam", self._ctx["conf"], "gagal_timeout_styrofoam")
        return "STANDBY"

    def cek_status(self):
        status = self.esp32.get_status()
        if status is None:
            print("[CEK_STATUS] ESP32 tak terjangkau, batal")
            return "STANDBY"

        if status.get("sedang_bergerak", False):
            print("[CEK_STATUS] ESP32 sedang sibuk, tunggu...")
            time.sleep(1)
            return "CEK_STATUS"

        self._ctx["status"] = status
        return "SORTING"

    def sorting(self):
        kelas = self._ctx["kelas"]
        audio_file = get_audio_file(kelas)
        
        # Validasi kelas untuk ESP32
        if kelas not in ["organik", "anorganik"]:
            print(f"[SORTING] kelas '{kelas}' tidak dikenali ESP32, menggunakan 'organik'")
            kelas_esp32 = "organik"
        else:
            kelas_esp32 = kelas

        print(f"[SORTING] mengirim kelas: {kelas_esp32} (asli: {kelas})")
        self.audio.play(audio_file, block=False)

        resp = self.esp32.post_servo(kelas_esp32)
        if resp is None:
            print("[SORTING] perintah servo gagal, batal")
            self.logger.log(kelas, self._ctx["conf"], "gagal_servo", kelas)
            return "STANDBY"

        print(f"[SORTING] respon ESP32: {resp}")
        self.logger.log(kelas, self._ctx["conf"], "sukses", kelas)
        # Kirim ke Flask dashboard
        routed = "organik" if kelas == "organik" else "anorganik"
        kirim_deteksi_ke_flask(kelas, self._ctx["conf"], routed)
        self.audio.play("terima_kasih.mp3")

        print("[SORTING] menunggu servo kembali ke netral...")
        time.sleep(4.5)

        return "STANDBY"

    def run(self):
        # Cek koneksi ESP32
        if not self.esp32.health():
            print(f"[INFO] ESP32 ({ESP32_IP}) belum terjangkau. "
                  f"Pastikan RPi terhubung ke WiFi AP ESP32.")
        else:
            print("[INFO] ESP32 terhubung!")

        state = "STANDBY"
        handlers = {
            "STANDBY":        self.standby,
            "DETEKSI":        self.deteksi,
            "STYROFOAM_WAIT": self.styrofoam_wait,
            "CEK_STATUS":     self.cek_status,
            "SORTING":        self.sorting,
        }
        print("=== Sistem siap. Fokus jendela kamera, tekan SPASI mulai, q keluar ===")
        print(f"[INFO] Kelas yang dikenali ESP32: organik, anorganik, tengah")
        print(f"[INFO] Mapping kelas: {KELAS_MAPPING}")

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
            self.esp32.post_servo("tengah")
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