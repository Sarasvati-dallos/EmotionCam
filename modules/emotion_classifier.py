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
        print(f"[EmotionClassifier] Modelo cargado ✓  Etiquetas: {self.labels}")

    def predict(self, face_img_rgb):
        import tensorflow as tf
        import cv2
        img = cv2.resize(face_img_rgb, (224, 224))
        img = img.astype('float32')
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = img.reshape(1, 224, 224, 3)
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
