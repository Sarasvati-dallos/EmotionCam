"""
app.py
Servidor Flask de EmotionCam.
Corre en la PC y se accede desde el teléfono por WiFi.
"""

import cv2
import time
import base64
import numpy as np
from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO

from modules.face_detector import FaceDetector
from modules.emotion_classifier import get_classifier
from modules import database as db

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotioncam2024'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# ── Módulos IA ──────────────────────────────────────────────
detector   = FaceDetector()
classifier = get_classifier()

# ── Estado global de sesión ─────────────────────────────────
session_state = {
    'active':     False,
    'session_id': None,
    'start_time': None,
    'last_emotion': '—',
    'last_emoji':   '😐',
    'last_conf':    0.0,
    'last_save':    0.0,   # timestamp del último save a DB
}

cap = None  # cámara global


def open_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def release_camera():
    global cap
    if cap and cap.isOpened():
        cap.release()
        cap = None


def generate_frames():
    """Generador MJPEG para el <img> del navegador."""
    open_camera()
    while session_state['active']:
        if cap is None or not cap.isOpened():
            break
        ret, frame = cap.read()
        if not ret:
            continue

        faces = detector.detect(frame)
        labels = []

        for rect in faces:
            face_rgb = detector.crop_face(frame, rect)
            emotion, conf, emoji = classifier.predict(face_rgb)

            session_state['last_emotion'] = emotion.capitalize()
            session_state['last_emoji']   = emoji
            session_state['last_conf']    = conf

            # Guardar en DB máximo una vez por segundo
            now = time.time()
            if session_state['session_id'] and now - session_state['last_save'] >= 1.0:
                db.save_detection(session_state['session_id'], emotion, conf)
                session_state['last_save'] = now

            labels.append(f"{emoji} {emotion} {conf:.0%}")

        # Dibujar bounding boxes
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 2)
            if i < len(labels):
                cv2.rectangle(frame, (x, y-28), (x+len(labels[i])*11, y), (0,0,0), -1)
                cv2.putText(frame, labels[i], (x+3, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)

        # Emitir estado actual por SocketIO
        socketio.emit('emotion_update', {
            'emotion': session_state['last_emotion'],
            'emoji':   session_state['last_emoji'],
            'conf':    round(session_state['last_conf'] * 100),
        })

        # Encodear como JPEG y enviar
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    release_camera()


# ── Rutas ────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    sessions = db.get_all_sessions()
    return render_template('history.html', sessions=sessions)


@app.route('/stats')
def stats():
    raw   = db.get_emotion_stats()
    total = sum(raw.values()) if raw else 1
    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emojis = {'angry':'😠','disgust':'🤢','fear':'😨','happy':'😄',
              'neutral':'😐','sad':'😢','surprise':'😮'}
    data = []
    for e in all_emotions:
        cnt = raw.get(e, 0)
        data.append({
            'emotion': e.capitalize(),
            'emoji':   emojis.get(e, '❓'),
            'count':   cnt,
            'pct':     round(cnt / total * 100, 1),
        })
    data.sort(key=lambda x: x['count'], reverse=True)
    top = data[0] if data and data[0]['count'] > 0 else None
    return render_template('stats.html', data=data, top=top)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def api_start():
    if not session_state['active']:
        session_state['active']     = True
        session_state['session_id'] = db.start_session()
        session_state['start_time'] = time.time()
        session_state['last_save']  = 0.0
    return jsonify(ok=True)


@app.route('/api/stop', methods=['POST'])
def api_stop():
    if session_state['active']:
        session_state['active'] = False
        if session_state['session_id'] and session_state['start_time']:
            duration = int(time.time() - session_state['start_time'])
            db.end_session(session_state['session_id'], duration)
        session_state['session_id'] = None
        session_state['start_time'] = None
        release_camera()
    return jsonify(ok=True)


@app.route('/api/status')
def api_status():
    return jsonify(
        active  = session_state['active'],
        emotion = session_state['last_emotion'],
        emoji   = session_state['last_emoji'],
        conf    = round(session_state['last_conf'] * 100),
    )


if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  EmotionCam corriendo en:")
    print(f"  PC:      http://localhost:5000")
    print(f"  Teléfono: http://{local_ip}:5000")
    print(f"  (ambos deben estar en el mismo WiFi)")
    print(f"{'='*50}\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
