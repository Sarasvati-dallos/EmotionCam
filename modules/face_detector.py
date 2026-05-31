"""
modules/face_detector.py
Detecta rostros con Haar Cascades y recorta la región facial.
"""

import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect(self, frame_bgr):
        """Retorna lista de (x, y, w, h)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        return list(faces) if len(faces) > 0 else []

    def crop_face(self, frame_bgr, rect, size=(224, 224)):
        """Recorta y redimensiona la cara a RGB."""
        x, y, w, h = rect
        m = int(0.1 * min(w, h))
        x1 = max(0, x - m);  y1 = max(0, y - m)
        x2 = min(frame_bgr.shape[1], x + w + m)
        y2 = min(frame_bgr.shape[0], y + h + m)
        crop = frame_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, size)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
