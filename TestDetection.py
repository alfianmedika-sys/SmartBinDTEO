import os
import csv
import queue
import threading
import time
from datetime import datetime

import cv2
import requests
import pygame
import torch
from ultralytics import YOLO

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
# 4. REAL-TIME DETECTION
# ============================================================

MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
    os.path.join(os.path.dirname(__file__), "weights", "last.pt"),
    "weights/best.pt",
    "weights/last.pt",
]

CAMERA_INDEX = 0
CAP_WIDTH = 640
CAP_HEIGHT = 360
INFERENCE_SIZE = 320
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
WINDOW_TITLE = "Smart Bin - Kamera Real-time"


def find_model_path():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Model YOLO tidak ditemukan. Letakkan file .pt di folder weights/ best.pt atau last.pt."
    )


def draw_detections(frame, result, names):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confidences, class_ids):
        if conf < CONFIDENCE_THRESHOLD:
            continue
        label = names.get(class_id, str(class_id))
        text = f"{label} {conf * 100:.1f}%"
        color = (0, 255, 0)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    cam = CameraStream(CAMERA_INDEX, CAP_WIDTH, CAP_HEIGHT)
    time.sleep(0.5)
    width, height = cam.actual_resolution()
    print(f"[CAM] resolusi aktual: {width} x {height}")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow(WINDOW_TITLE, gray_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[DONE] Interupsi keyboard diterima.")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
