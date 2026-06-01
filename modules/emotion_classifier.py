"""
modules/emotion_classifier.py
Clasifica emociones usando DeepFace.
"""

import numpy as np
import random

EMOJIS = {
    'angry':    '😠',
    'disgust':  '🤢',
    'fear':     '😨',
    'happy':    '😄',
    'neutral':  '😐',
    'sad':      '😢',
    'surprise': '😮',
}

LABEL_MAP = {
    'angry':    'angry',
    'disgust':  'disgust',
    'fear':     'fear',
    'happy':    'happy',
    'neutral':  'neutral',
    'sad':      'sad',
    'surprise': 'surprise',
}


class PlaceholderClassifier:
    def predict(self, face_img_rgb):
        emotion = random.choice(list(EMOJIS.keys()))
        confidence = round(random.uniform(0.55, 0.99), 2)
        return emotion, confidence, EMOJIS.get(emotion, '😐')


class DeepFaceClassifier:
    def __init__(self):
        from deepface import DeepFace
        self.DeepFace = DeepFace
        print('[DeepFaceClassifier] Listo')

    def predict(self, face_img_rgb):
        try:
            result = self.DeepFace.analyze(
                face_img_rgb,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
            )
            emotions = result[0]['emotion']
            # Normalizar scores a 0-1
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v / total for k, v in emotions.items()}

            emotion_key = max(emotions, key=lambda k: emotions[k])
            confidence = float(emotions[emotion_key])
            emotion_key = LABEL_MAP.get(emotion_key.lower(), emotion_key.lower())
            return emotion_key, confidence, EMOJIS.get(emotion_key, '😐')
        except Exception as e:
            print(f'[DeepFaceClassifier] Error: {e}')
            return 'neutral', 0.5, '😐'


def get_classifier(model_path=None, labels_path=None):
    try:
        return DeepFaceClassifier()
    except Exception as e:
        print(f'[get_classifier] Error cargando DeepFace: {e}')
        print('[get_classifier] Usando PLACEHOLDER')
        return PlaceholderClassifier()