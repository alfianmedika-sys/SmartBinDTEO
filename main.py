"""
main.py — Entry point tunggal SmartBin
Jalankan: python3 main.py

Menjalankan Flask API + WasteSorter dalam SATU proses:
  - Flask jalan di thread terpisah (daemon)
  - WasteSorter jalan di main thread
  - Frame kamera di-share via buffer memori langsung (tanpa HTTP)
  - Tidak perlu cv2.imshow, tidak perlu monitor
"""

import threading
import time
import cv2
import numpy as np

# ── Shared frame buffer (diakses App.py dan WasteSorter) ──────────
_frame_lock  = threading.Lock()
_frame_bytes = None   # JPEG bytes terbaru dari kamera

def set_frame(frame_bgr):
    """Dipanggil WasteSorter setiap ada frame baru."""
    global _frame_bytes
    if frame_bgr is None:
        return
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 65])
    with _frame_lock:
        _frame_bytes = buf.tobytes()

def get_frame():
    """Dipanggil Flask generator untuk stream ke browser."""
    with _frame_lock:
        return _frame_bytes


# ── Patch App.py agar pakai shared buffer di atas ────────────────
import App
App._mjpeg_lock  = _frame_lock
# Override get_frame di App agar pakai fungsi get_frame kita
import types

def _patched_gen_frames():
    blank = _buat_blank()
    while True:
        data = get_frame() or blank
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.04)

def _buat_blank():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Menunggu kamera...", (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

App._gen_frames = _patched_gen_frames


# ── Jalankan Flask di thread background ──────────────────────────
def jalankan_flask():
    print("[FLASK] Menjalankan Flask di port 5000...")
    App.app.run(host="0.0.0.0", port=5000, debug=False, threaded=True,
                use_reloader=False)

flask_thread = threading.Thread(target=jalankan_flask, daemon=True)
flask_thread.start()
time.sleep(2)   # tunggu Flask siap
print("[FLASK] Flask siap di http://0.0.0.0:5000")


# ── Patch WasteSorter: ganti tampilkan() agar tidak pakai imshow ──
import FixCodeForWasteDetection as WS

_original_tampilkan = WS.WasteSorter.tampilkan

def _headless_tampilkan(self, frame, label=""):
    """Versi headless: tidak pakai cv2.imshow, kirim frame ke buffer."""
    if frame is None:
        return -1

    # Ambil hasil deteksi terbaru
    result = self.worker.get_result()
    if result is not None:
        self._last_dets = result

    if self._last_dets:
        WS.gambar_deteksi(frame, self._last_dets)

    self._frame_cnt += 1
    if self._frame_cnt % self._detect_every == 0:
        self.worker.submit(frame, self.frame_w, self.frame_h)

    # Gambar label status di frame
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Share ke Flask stream — tidak ada cv2.imshow
    set_frame(frame)
    return -1   # headless: tidak ada key input

WS.WasteSorter.tampilkan = _headless_tampilkan


# ── Jalankan WasteSorter di main thread ──────────────────────────
print("[WASTE] Menjalankan WasteSorter...")
try:
    WS.WasteSorter().run()
except KeyboardInterrupt:
    print("\n[EXIT] Dihentikan pengguna")
except Exception as e:
    print(f"[FATAL] {e}")
    raise