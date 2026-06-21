"""
============================================================
 WasteSorter_remote_laravel.py
 Sistem Pemilahan Sampah Otomatis - Node Raspberry Pi 4
 Layer: AI, Decision, Remote Laravel Integration
 Versi: ESP32-S3 BLE + Laravel /api/waste-detection
============================================================
"""

import asyncio
import csv
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import pygame
import requests
from ultralytics import YOLO

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    BleakClient = None
    BleakScanner = None


# ============================================================
# 1. KONFIGURASI GLOBAL
# ============================================================

# --- YOLO ---
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/wastemanagement2026/Smart Bin/waste_results_3/weights/best.pt",
)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "160"))

CAP_WIDTH = int(os.getenv("CAP_WIDTH", "640"))
CAP_HEIGHT = int(os.getenv("CAP_HEIGHT", "480"))

# --- ESP32 BLE ---
ESP32_NAME = os.getenv("ESP32_NAME", "SmartBin-ESP32")
ESP32_SERVICE_UUID = os.getenv(
    "ESP32_SERVICE_UUID",
    "12345678-1234-1234-1234-123456789abc",
)
ESP32_CMD_CHAR_UUID = os.getenv(
    "ESP32_CMD_CHAR_UUID",
    "12345678-1234-1234-1234-123456789001",
)
ESP32_BLE_SCAN_TIMEOUT = float(os.getenv("ESP32_BLE_SCAN_TIMEOUT", "5"))

# --- Laravel remote host ---
LARAVEL_BASE_URL = os.getenv(
    "LARAVEL_BASE_URL",
    "https://web-production-23834.up.railway.app",
)
SENSOR_API_KEY = os.getenv("SENSOR_API_KEY", "smart-waste-test-key")
LOCATION = os.getenv("SMART_WASTE_LOCATION", "BB102")
LARAVEL_DEVICE_ID = os.getenv("LARAVEL_DEVICE_ID", "RPI-BB102-CAM")
LARAVEL_TIMEOUT = float(os.getenv("LARAVEL_TIMEOUT", "10"))
SEND_IMAGE_TO_LARAVEL = os.getenv("SEND_IMAGE_TO_LARAVEL", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# --- Threshold & timeout ---
STYROFOAM_TIMEOUT = int(os.getenv("STYROFOAM_TIMEOUT", "30"))

# --- Path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.getenv("AUDIO_DIR", os.path.join(BASE_DIR, "audio"))
LOG_DIR = os.getenv("LOG_DIR", os.path.join(BASE_DIR, "logs"))
LOG_PATH = os.getenv("LOG_PATH", os.path.join(LOG_DIR, "log_sampah.csv"))

# Mode pemicu STANDBY: "sensor" atau "manual"
MODE_TRIGGER = os.getenv("MODE_TRIGGER", "manual")

# --- Contour detection ---
MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", "1500"))
MAX_RATIO = float(os.getenv("MAX_RATIO", "5.0"))
MIN_RATIO = float(os.getenv("MIN_RATIO", "0.2"))
MAX_REL_SIZE = float(os.getenv("MAX_REL_SIZE", "0.9"))
ROI_PADDING = int(os.getenv("ROI_PADDING", "15"))
MAX_OBJECTS = int(os.getenv("MAX_OBJECTS", "3"))


# ============================================================
# 2. PEMETAAN KELAS
# ============================================================

KELAS = {
    0: "organik",
    1: "anorganik",
    2: "anorganik_styrofoam",
}

KELAS_MAPPING = {
    "organik": "organik",
    "anorganik": "anorganik",
    "anorganik_styrofoam": "anorganik",
    "R": "anorganik",
    "O": "organik",
    "organic": "organik",
    "inorganic": "anorganik",
    "styrofoam": "anorganik",
}


# ============================================================
# 3. CAMERA STREAM
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
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
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return width, height

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()


# ============================================================
# 4. CONTOUR DETECTION
# ============================================================

def cari_kandidat_box(frame):
    frame_h, frame_w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, width, height = cv2.boundingRect(cnt)
        ratio = width / float(height) if height > 0 else 0
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            continue
        if width > frame_w * MAX_REL_SIZE or height > frame_h * MAX_REL_SIZE:
            continue

        candidates.append((area, (x, y, width, height)))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    return [box for _, box in candidates[:MAX_OBJECTS]]


# ============================================================
# 5. INFERENCE WORKER
# ============================================================

def _jalankan_inferensi(frame, model, names, frame_w, frame_h):
    candidates = cari_kandidat_box(frame)
    if not candidates:
        return []

    rois = []
    box_coords = []

    for (x, y, width, height) in candidates:
        x1 = max(0, x - ROI_PADDING)
        y1 = max(0, y - ROI_PADDING)
        x2 = min(frame_w, x + width + ROI_PADDING)
        y2 = min(frame_h, y + height + ROI_PADDING)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        rois.append(roi_resized)
        box_coords.append((x1, y1, x2, y2))

    if not rois:
        return []

    all_results = model.predict(source=rois, imgsz=IMG_SIZE, verbose=False)

    detections = []
    for result, (x1, y1, x2, y2) in zip(all_results, box_coords):
        if getattr(result, "probs", None) is not None:
            kelas_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            kelas = names.get(kelas_id, KELAS.get(kelas_id, "unknown"))
            kelas_normal = KELAS_MAPPING.get(kelas, kelas)

            detections.append({
                "xyxy": (x1, y1, x2, y2),
                "kelas": kelas_normal,
                "kelas_asli": kelas,
                "confidence": confidence,
            })
            continue

        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            index = confs.argmax()
            kelas_id = int(clss[index])
            confidence = float(confs[index])
            kelas = names.get(kelas_id, KELAS.get(kelas_id, "unknown"))
            kelas_normal = KELAS_MAPPING.get(kelas, kelas)

            detections.append({
                "xyxy": (x1, y1, x2, y2),
                "kelas": kelas_normal,
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

        self.input_q = queue.Queue(maxsize=1)
        self.output_q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                frame, frame_w, frame_h = self.input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            result = _jalankan_inferensi(frame, self.model, self.names, frame_w, frame_h)
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
        detections = _jalankan_inferensi(frame, self.model, self.names, frame_w, frame_h)
        if not detections:
            return None, 0.0

        best = max(detections, key=lambda detection: detection["confidence"])

        return best["kelas"], best["confidence"]

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


# ============================================================
# 6. KLIEN ESP32 BLE
# ============================================================

class ESP32BleClient:
    def __init__(self, name, command_char_uuid, scan_timeout):
        self.name = name
        self.command_char_uuid = command_char_uuid
        self.scan_timeout = scan_timeout
        self.address = None
        self.client = None
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)

        return future.result()

    async def _find_esp32(self):
        if BleakScanner is None:
            print("[BLE] bleak belum terinstall. Jalankan: pip install bleak")

            return None

        print(f"[BLE] scanning untuk {self.name}...")
        devices = await BleakScanner.discover(timeout=self.scan_timeout)
        for device in devices:
            if device.name == self.name:
                print(f"[BLE] ditemukan {device.name}: {device.address}")

                return device.address

        print("[BLE] ESP32 tidak ditemukan")

        return None

    async def _connect(self):
        if BleakClient is None:
            print("[BLE] bleak belum terinstall. Jalankan: pip install bleak")

            return None

        if self.client and self.client.is_connected:
            return self.client

        if not self.address:
            self.address = await self._find_esp32()

        if not self.address:
            return None

        self.client = BleakClient(self.address)
        await self.client.connect()
        print("[BLE] terhubung ke ESP32")

        return self.client

    async def _send(self, kelas):
        client = await self._connect()
        if not client:
            print("[BLE] tidak bisa kirim, ESP32 belum terhubung")

            return False

        try:
            command = kelas.lower().encode()
            await client.write_gatt_char(self.command_char_uuid, command, response=True)
            print(f"[BLE] terkirim: {kelas.lower()}")

            return True
        except Exception as error:
            print(f"[BLE] gagal kirim '{kelas}': {error}")
            self.address = None
            self.client = None

            return False

    async def _disconnect(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("[BLE] disconnected")

        self.client = None

    def init(self):
        with self._lock:
            client = self._run_async(self._connect())

        return client is not None and client.is_connected

    def send_servo(self, kelas):
        valid_classes = ["organik", "anorganik", "tengah", "netral"]
        kelas_send = kelas if kelas in valid_classes else "tengah"

        if kelas != kelas_send:
            print(f"[BLE] kelas '{kelas}' tidak valid, gunakan default 'tengah'")

        with self._lock:
            return self._run_async(self._send(kelas_send))

    def close(self):
        with self._lock:
            self._run_async(self._disconnect())

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)


# ============================================================
# 7. KLIEN LARAVEL REMOTE
# ============================================================

class LaravelClient:
    def __init__(self, base_url, api_key, timeout):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self):
        headers = {}
        if self.api_key:
            headers["X-Sensor-Key"] = self.api_key

        return headers

    def send_waste_detection(self, event_id, waste_type, confidence, frame=None, sequence_number=None):
        payload = {
            "event_id": event_id,
            "waste_type": waste_type,
            "confidence": confidence,
            "location": LOCATION,
            "device_id": LARAVEL_DEVICE_ID,
            "device_timestamp": time.time(),
        }

        if sequence_number is not None:
            payload["sequence_number"] = sequence_number

        try:
            if SEND_IMAGE_TO_LARAVEL and frame is not None:
                ok, encoded = cv2.imencode(".jpg", frame)
                if ok:
                    files = {
                        "image": ("detected-frame.jpg", encoded.tobytes(), "image/jpeg"),
                    }
                    response = requests.post(
                        f"{self.base_url}/api/waste-detection",
                        data=payload,
                        files=files,
                        headers=self._headers(),
                        timeout=self.timeout,
                    )
                else:
                    response = requests.post(
                        f"{self.base_url}/api/waste-detection",
                        json=payload,
                        headers=self._headers(),
                        timeout=self.timeout,
                    )
            else:
                response = requests.post(
                    f"{self.base_url}/api/waste-detection",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )

            response.raise_for_status()
            print(f"[LARAVEL] waste-detection terkirim: HTTP {response.status_code}")

            return response.json()
        except requests.RequestException as error:
            print(f"[LARAVEL] gagal kirim /api/waste-detection: {error}")

            return None

# ============================================================
# 8. PEMUTAR AUDIO
# ============================================================

class AudioPlayer:
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        try:
            os.makedirs(audio_dir, exist_ok=True)
        except OSError as error:
            print(f"[AUDIO] gagal buat direktori: {error}")

        try:
            pygame.mixer.init()
            self.ready = True
            print(f"[AUDIO] inisialisasi berhasil, direktori: {audio_dir}")
        except Exception as error:
            print(f"[AUDIO] mixer gagal init: {error}")
            self.ready = False

    def play(self, nama_file, block=True):
        if not self.ready:
            print(f"[AUDIO] (lewati) {nama_file}")
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
        except Exception as error:
            print(f"[AUDIO] gagal putar {nama_file}: {error}")


# ============================================================
# 9. LOGGER CSV
# ============================================================

class Logger:
    HEADER = [
        "timestamp",
        "event_id",
        "kelas",
        "kelas_asli",
        "confidence",
        "status",
        "laravel_detection_id",
    ]

    def __init__(self, path):
        self.path = path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"[LOG] direktori log: {os.path.dirname(path)}")
        except OSError as error:
            print(f"[LOG] gagal buat direktori: {error}")
            self.path = "log_sampah.csv"
            print(f"[LOG] menggunakan fallback: {self.path}")

        if not os.path.exists(self.path):
            self._write_row(self.HEADER)

    def _write_row(self, row):
        try:
            with open(self.path, "a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(row)
        except OSError as error:
            print(f"[LOG] gagal tulis: {error}")

    def log(
        self,
        event_id,
        kelas,
        confidence,
        status,
        kelas_asli="",
        laravel_detection_id="",
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_row([
            timestamp,
            event_id,
            kelas,
            kelas_asli,
            f"{confidence:.2f}",
            status,
            laravel_detection_id,
        ])
        print(f"[LOG] {timestamp} | {event_id} | {kelas} | conf={confidence:.2f} | {status}")


# ============================================================
# 10. PEMETAAN KEPUTUSAN
# ============================================================

def get_audio_file(kelas):
    audio_files = {
        "organik": "organik.mp3",
        "anorganik": "anorganik.mp3",
        "anorganik_styrofoam": "anorganik.mp3",
    }

    return audio_files.get(kelas, "organik.mp3")


def make_event_id(sequence_number):
    return f"RPI-{LOCATION}-{sequence_number:06d}-{int(time.time() * 1000)}"


# ============================================================
# 11. OVERLAY HELPER
# ============================================================

WARNA_KELAS = {
    "organik": (50, 200, 50),
    "anorganik": (50, 150, 255),
    "anorganik_styrofoam": (255, 50, 50),
}
WARNA_TIDAK_YAKIN = (0, 0, 255)
WARNA_DEFAULT = (200, 200, 200)


def gambar_deteksi(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["xyxy"]]
        confidence = detection["confidence"]
        kelas = detection["kelas"]
        kelas_asli = detection.get("kelas_asli", kelas)

        if confidence >= CONFIDENCE_THRESHOLD:
            warna = WARNA_KELAS.get(kelas, WARNA_DEFAULT)
            label = f"{kelas} {confidence * 100:.1f}%"
            if kelas_asli != kelas:
                label += f" ({kelas_asli})"
        else:
            warna = WARNA_TIDAK_YAKIN
            label = f"Tidak Yakin {confidence * 100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )
        cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width + 4, y1), warna, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


# ============================================================
# 12. STATE MACHINE UTAMA
# ============================================================

class WasteSorter:
    def __init__(self):
        self.esp32 = ESP32BleClient(
            ESP32_NAME,
            ESP32_CMD_CHAR_UUID,
            ESP32_BLE_SCAN_TIMEOUT,
        )
        self.laravel = LaravelClient(LARAVEL_BASE_URL, SENSOR_API_KEY, LARAVEL_TIMEOUT)
        self.audio = AudioPlayer(AUDIO_DIR)
        self.logger = Logger(LOG_PATH)

        self.cam = CameraStream(CAMERA_INDEX, CAP_WIDTH, CAP_HEIGHT)
        time.sleep(0.5)
        self.frame_w, self.frame_h = self.cam.actual_resolution()
        print(f"[CAM] resolusi aktual: {self.frame_w}x{self.frame_h}")

        self.worker = InferenceWorker(MODEL_PATH, IMG_SIZE)

        self.running = True
        self._ctx = {}
        self._last_dets = []
        self._frame_cnt = 0
        self._detect_every = 3
        self._sequence_number = 0

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
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )
            cv2.rectangle(frame, (0, 0), (text_width + 16, text_height + 16), (0, 0, 0), -1)
            cv2.putText(
                frame,
                label,
                (8, text_height + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Pemilahan Sampah - Smart Bin (q=keluar)", frame)

        return cv2.waitKey(1) & 0xFF

    def standby(self):
        frame = self.baca_frame()
        key = self.tampilkan(frame, "STANDBY - SPASI mulai | q keluar")

        if key == ord("q"):
            self.running = False

            return None

        if MODE_TRIGGER == "manual" and key == ord(" "):
            return "DETEKSI"

        return "STANDBY"

    def deteksi(self):
        frame = self.baca_frame()
        if frame is None:
            return "STANDBY"

        kelas, confidence = self.worker.detect_blocking(frame, self.frame_w, self.frame_h)
        self.tampilkan(frame, f"DETEKSI: {kelas} ({confidence:.2f})")

        detections = self.worker.get_result()
        if detections:
            print(
                "[DETEKSI] semua hasil: "
                f"{[(d['kelas'], d['kelas_asli'], d['confidence']) for d in detections]}"
            )

        if kelas is None or confidence < CONFIDENCE_THRESHOLD:
            print(f"[DETEKSI] kurang yakin (kelas={kelas}, conf={confidence:.2f}) -> scan ulang")
            self.audio.play("scan_ulang.mp3")
            self.logger.log("", kelas or "none", confidence, "gagal_confidence")

            return "STANDBY"

        self._sequence_number += 1
        event_id = make_event_id(self._sequence_number)
        print(f"[DETEKSI] event_id={event_id} kelas={kelas}, conf={confidence:.2f}")

        self._ctx = {
            "event_id": event_id,
            "sequence_number": self._sequence_number,
            "kelas": kelas,
            "conf": confidence,
            "frame": frame.copy(),
        }

        if kelas == "anorganik_styrofoam":
            return "STYROFOAM_WAIT"

        return "CEK_STATUS"

    def styrofoam_wait(self):
        self.audio.play("styrofoam.mp3")
        print(
            f"[STYROFOAM] menunggu sisa dibuang "
            f"(timeout {STYROFOAM_TIMEOUT}s, tekan 'c' untuk konfirmasi)"
        )

        mulai = time.time()
        while time.time() - mulai < STYROFOAM_TIMEOUT:
            frame = self.baca_frame()
            sisa = int(STYROFOAM_TIMEOUT - (time.time() - mulai))
            key = self.tampilkan(frame, f"Buang sisa makanan... ({sisa}s) tekan 'c'")
            if key == ord("c"):
                print("[STYROFOAM] dikonfirmasi, lanjut sorting")

                return "CEK_STATUS"
            if key == ord("q"):
                self.running = False

                return None

        print("[STYROFOAM] timeout, batalkan")
        self.logger.log(
            self._ctx.get("event_id", ""),
            "anorganik_styrofoam",
            self._ctx["conf"],
            "gagal_timeout_styrofoam",
        )

        return "STANDBY"

    def cek_status(self):
        self._ctx["status"] = {}
        return "SORTING"

    def _send_laravel_records(self, kelas_esp32):
        return self.laravel.send_waste_detection(
            event_id=self._ctx["event_id"],
            waste_type=kelas_esp32,
            confidence=self._ctx["conf"],
            frame=self._ctx.get("frame"),
            sequence_number=self._ctx.get("sequence_number"),
        )

    def sorting(self):
        kelas = self._ctx["kelas"]
        audio_file = get_audio_file(kelas)

        if kelas not in ["organik", "anorganik"]:
            print(f"[SORTING] kelas '{kelas}' tidak dikenali ESP32, menggunakan 'organik'")
            kelas_esp32 = "organik"
        else:
            kelas_esp32 = kelas

        print(f"[SORTING] mengirim kelas: {kelas_esp32} (asli: {kelas})")
        self.audio.play(audio_file, block=False)

        sent = self.esp32.send_servo(kelas_esp32)
        if not sent:
            print("[SORTING] perintah BLE servo gagal, batal")
            self.logger.log(self._ctx["event_id"], kelas, self._ctx["conf"], "gagal_servo", kelas)

            return "STANDBY"

        print("[SORTING] perintah BLE servo terkirim")

        detection_response = self._send_laravel_records(kelas_esp32)
        laravel_detection_id = ""

        if detection_response:
            laravel_detection_id = detection_response.get("detection_id", "")

        status = "sukses_laravel" if detection_response else "sukses_lokal_laravel_gagal"
        self.logger.log(
            self._ctx["event_id"],
            kelas,
            self._ctx["conf"],
            status,
            kelas,
            laravel_detection_id,
        )
        self.audio.play("terima_kasih.mp3")

        print("[SORTING] menunggu servo kembali ke netral...")
        time.sleep(4.5)

        return "STANDBY"

    def run(self):
        if not self.esp32.init():
            print(f"[INFO] ESP32 BLE ({ESP32_NAME}) belum terjangkau.")
        else:
            print("[INFO] ESP32 BLE terhubung!")

        print(f"[INFO] Laravel remote host: {LARAVEL_BASE_URL}")
        print(f"[INFO] Laravel location: {LOCATION}")
        print(f"[INFO] ESP32 BLE name: {ESP32_NAME}")
        print(f"[INFO] Kirim image ke Laravel: {SEND_IMAGE_TO_LARAVEL}")

        state = "STANDBY"
        handlers = {
            "STANDBY": self.standby,
            "DETEKSI": self.deteksi,
            "STYROFOAM_WAIT": self.styrofoam_wait,
            "CEK_STATUS": self.cek_status,
            "SORTING": self.sorting,
        }
        print("=== Sistem siap. Fokus jendela kamera, tekan SPASI mulai, q keluar ===")
        print("[INFO] Kelas yang dikenali ESP32: organik, anorganik, tengah")
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
            self.esp32.send_servo("tengah")
        except Exception:
            pass
        try:
            self.esp32.close()
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
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        WasteSorter().run()
    except KeyboardInterrupt:
        print("\n[EXIT] dihentikan pengguna")
    except Exception as error:
        print(f"[FATAL] {error}")
