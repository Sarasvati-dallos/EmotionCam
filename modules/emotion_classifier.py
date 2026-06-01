"""
modules/emotion_classifier.py
Clasifica emociones usando el modelo de Yeison.
Si no encuentra el modelo, usa placeholder aleatorio.
"""

import numpy as np
import json
import random
import os


EMOJIS = {
    'angry':    '😠',
    'disgust':  '🤢',
    'fear':     '😨',
    'happy':    '😄',
    'neutral':  '😐',
    'sad':      '😢',
    'surprise': '😮',
}


class PlaceholderClassifier:
    LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def predict(self, face_img_rgb):
        emotion = random.choice(self.LABELS)
        confidence = round(random.uniform(0.55, 0.99), 2)
        return emotion, confidence, EMOJIS.get(emotion, '😐')


class EmotionClassifier:
    def __init__(self, model_path, labels_path):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

        # Inspect model input shape to adapt preprocessing
        try:
            self.input_shape = self.model.input_shape  # (None, H, W, C) or similar
            _, h, w, c = self.input_shape
            self.target_size = (int(w), int(h))
            self.channels = int(c) if c is not None else 1
        except Exception:
            # Fallback defaults
            self.input_shape = None
            self.target_size = (224, 224)
            self.channels = 3

        print(f"[EmotionClassifier] Modelo cargado OK  Etiquetas: {self.labels}  target={self.target_size} channels={self.channels}")

    def predict(self, face_img_rgb):
        import tensorflow as tf
        import cv2
        # Prepare image according to model input channels
        target_w, target_h = self.target_size
        gray = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2GRAY)

        if self.channels == 1:
            # Model truly expects grayscale input.
            img = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img = img.astype('float32')
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            img = img.reshape(1, target_h, target_w, 1)
        else:
            # Training used grayscale content, but MobileNet base expects 3 channels.
            # Keep the image grayscale, then replicate to RGB so the model sees the same
            # luminance-only content it was trained on.
            img = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.astype('float32')
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            img = img.reshape(1, target_h, target_w, 3)
        pred = self.model.predict(img, verbose=0)[0]
        idx = int(np.argmax(pred))
        emotion = self.labels[idx]
        confidence = float(np.max(pred))
        return emotion, confidence, EMOJIS.get(emotion, '😐')


def get_classifier(model_path='models/emotion_model.h5',
                   labels_path='models/emotion_labels.json'):
    if os.path.exists(model_path) and os.path.exists(labels_path):
        try:
            return EmotionClassifier(model_path, labels_path)
        except Exception as e:
            print(f"[get_classifier] Error cargando modelo: {e}")
    print("[get_classifier] Usando PLACEHOLDER")
    return PlaceholderClassifier()
