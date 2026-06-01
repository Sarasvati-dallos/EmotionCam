"""
app.py
Servidor Flask de EmotionCam.
Corre en la PC y se accede desde el teléfono por WiFi.
"""

import cv2
import time
import base64
import ipaddress
import os
import threading
import numpy as np
from collections import deque
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, jsonify, Response, request, send_file
from flask_socketio import SocketIO
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial

from modules.face_detector import FaceDetector
from modules.emotion_classifier import get_classifier, EMOJIS
from modules import database as db

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotioncam2024'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

BASE_DIR = os.path.dirname(__file__)
CERT_DIR = os.path.join(BASE_DIR, 'certs')
CA_CERT_PATH = os.path.join(CERT_DIR, 'emotioncam-ca.crt')
CA_KEY_PATH = os.path.join(CERT_DIR, 'emotioncam-ca.key')
SERVER_CERT_PATH = os.path.join(CERT_DIR, 'emotioncam-server.crt')
SERVER_KEY_PATH = os.path.join(CERT_DIR, 'emotioncam-server.key')

# ── Módulos IA ──────────────────────────────────────────────
detector   = FaceDetector()
classifier = get_classifier()

# ── Estado global de sesión ─────────────────────────────────
session_state = {
    'active':             False,
    'session_id':         None,
    'start_time':         None,
    'prediction_window':  deque(maxlen=8),
    'stable_emotion_key': 'neutral',
    'last_emotion':       '—',
    'last_emoji':         '😐',
    'last_conf':          0.0,
    'last_save':          0.0,
}

cap = None  # cámara global


def _get_lan_ip():
    """Devuelve la IP local usada para salir a la red, evitando adaptadores virtuales."""
    import socket
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(('8.8.8.8', 80))
        ip = probe.getsockname()[0]
        probe.close()
        return ip
    except OSError:
        return '127.0.0.1'


def _load_certificate(path):
    with open(path, 'rb') as handle:
        return x509.load_pem_x509_certificate(handle.read())


def _server_cert_matches_ip(cert_path, local_ip):
    if not os.path.exists(cert_path):
        return False
    try:
        cert = _load_certificate(cert_path)
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        dns_names = set(san.get_values_for_type(x509.DNSName))
        ip_names = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
        return {'localhost', 'emotioncam.local'}.issubset(dns_names) and {'127.0.0.1', local_ip}.issubset(ip_names)
    except Exception:
        return False


def _ensure_https_assets(local_ip):
    os.makedirs(CERT_DIR, exist_ok=True)

    if not os.path.exists(CA_KEY_PATH) or not os.path.exists(CA_CERT_PATH):
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        ca_name = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, 'CO'),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'EmotionCam Local CA'),
            x509.NameAttribute(NameOID.COMMON_NAME, 'EmotionCam Local CA'),
        ])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )
        with open(CA_KEY_PATH, 'wb') as handle:
            handle.write(
                ca_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
        with open(CA_CERT_PATH, 'wb') as handle:
            handle.write(ca_cert.public_bytes(serialization.Encoding.PEM))

    if _server_cert_matches_ip(SERVER_CERT_PATH, local_ip) and os.path.exists(SERVER_KEY_PATH):
        return SERVER_CERT_PATH, SERVER_KEY_PATH

    with open(CA_KEY_PATH, 'rb') as handle:
        ca_key = serialization.load_pem_private_key(handle.read(), password=None)
    ca_cert = _load_certificate(CA_CERT_PATH)

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, 'CO'),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'EmotionCam Local Server'),
        x509.NameAttribute(NameOID.COMMON_NAME, 'EmotionCam Local Server'),
    ])
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=825))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName('localhost'),
                x509.DNSName('emotioncam.local'),
                x509.IPAddress(ipaddress.ip_address('127.0.0.1')),
                x509.IPAddress(ipaddress.ip_address(local_ip)),
            ]),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    with open(SERVER_KEY_PATH, 'wb') as handle:
        handle.write(
            server_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(SERVER_CERT_PATH, 'wb') as handle:
        handle.write(server_cert.public_bytes(serialization.Encoding.PEM))

    return SERVER_CERT_PATH, SERVER_KEY_PATH


def _start_ca_http_server():
    """Sirve la carpeta de certificados por HTTP para instalar la CA sin confiar todavía en HTTPS."""
    handler = partial(SimpleHTTPRequestHandler, directory=CERT_DIR)
    server = ThreadingHTTPServer(('0.0.0.0', 5001), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _reset_prediction_smoothing():
    session_state['prediction_window'].clear()
    session_state['stable_emotion_key'] = 'neutral'
    session_state['last_emotion'] = 'Neutral'
    session_state['last_emoji'] = '😐'
    session_state['last_conf'] = 0.0


def _update_smoothed_emotion(emotion_key, confidence, emoji):
    """
    Voting window simple: los últimos 8 frames votan ponderados por confianza.
    Gana la emoción con mayor confianza acumulada. Sin umbrales artificiales.
    """
    window = session_state['prediction_window']
    window.append((emotion_key, confidence, emoji))

    # Sumar confianza por emoción en la ventana
    scores = {}
    for k, conf, _ej in window:
        scores[k] = scores.get(k, 0.0) + float(conf)

    # Ganador = mayor score acumulado
    winner_key = max(scores, key=lambda k: scores[k])
    winner_items = [item for item in window if item[0] == winner_key]
    avg_conf = sum(i[1] for i in winner_items) / len(winner_items)
    winner_emoji = winner_items[-1][2]

    session_state['stable_emotion_key'] = winner_key
    session_state['last_emotion'] = winner_key.capitalize()
    session_state['last_emoji'] = winner_emoji
    session_state['last_conf'] = float(avg_conf)

    return session_state['last_emotion'], session_state['last_conf'], session_state['last_emoji']


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


def _decode_frame_payload(request_obj):
    if 'frame' in request_obj.files:
        file_storage = request_obj.files['frame']
        data = np.frombuffer(file_storage.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('No se pudo decodificar la imagen enviada')
        return frame

    payload = request_obj.get_json(silent=True) or {}
    image_b64 = payload.get('image') or payload.get('frame')
    if not image_b64:
        raise ValueError('No se recibió ninguna imagen')

    if ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]

    data = base64.b64decode(image_b64)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('No se pudo decodificar la imagen enviada')
    return frame


def _process_frame(frame_bgr):
    faces = detector.detect(frame_bgr)
    detections = []

    for rect in faces:
        face_rgb = detector.crop_face(frame_bgr, rect)
        emotion, conf, emoji = classifier.predict(face_rgb)
        emotion_key = emotion.lower().strip()
        stable_emotion, stable_conf, stable_emoji = _update_smoothed_emotion(emotion_key, conf, emoji)
        detections.append({
            'rect':    rect,
            'emotion': stable_emotion,
            'emoji':   stable_emoji,
            'conf':    stable_conf,
        })

    if detections:
        top_detection = detections[-1]
        if session_state['session_id']:
            now = time.time()
            if now - session_state['last_save'] >= 1.0:
                db.save_detection(session_state['session_id'], session_state['stable_emotion_key'], session_state['last_conf'])
                session_state['last_save'] = now

        session_state['last_emotion'] = top_detection['emotion']
        session_state['last_emoji']   = top_detection['emoji']
        session_state['last_conf']    = top_detection['conf']

    return frame_bgr, detections


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
            _update_smoothed_emotion(emotion.lower(), conf, emoji)

            # Guardar en DB máximo una vez por segundo
            now = time.time()
            if session_state['session_id'] and now - session_state['last_save'] >= 1.0:
                db.save_detection(session_state['session_id'], session_state['stable_emotion_key'], session_state['last_conf'])
                session_state['last_save'] = now

            labels.append(f"{session_state['last_emoji']} {session_state['last_emotion']} {session_state['last_conf']:.0%}")

        # Dibujar bounding boxes
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 2)
            if i < len(labels):
                cv2.rectangle(frame, (x, y-28), (x+len(labels[i])*11, y), (0, 0, 0), -1)
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
    emojis = {'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😄',
              'neutral': '😐', 'sad': '😢', 'surprise': '😮'}
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


@app.route('/api/debug')
def api_debug():
    window = list(session_state.get('prediction_window', []))
    return jsonify(
        ok=True,
        window=[{'k': w[0], 'conf': w[1], 'emoji': w[2]} for w in window],
        stable=session_state['stable_emotion_key'],
    )


@app.route('/api/reset-smoothing', methods=['POST'])
def api_reset_smoothing():
    _reset_prediction_smoothing()
    return jsonify(ok=True)


@app.route('/api/frame', methods=['POST'])
def api_frame():
    try:
        frame = _decode_frame_payload(request)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400

    _, detections = _process_frame(frame)
    if not detections:
        return jsonify(
            ok=True,
            emotion=session_state['last_emotion'],
            emoji=session_state['last_emoji'],
            conf=round(session_state['last_conf'] * 100),
            faces=0,
        )

    return jsonify(
        ok=True,
        emotion=session_state['last_emotion'],
        emoji=session_state['last_emoji'],
        conf=round(session_state['last_conf'] * 100),
        faces=len(detections),
    )


@app.route('/download-ca')
def download_ca():
    local_ip = _get_lan_ip()
    _ensure_https_assets(local_ip)
    return send_file(
        CA_CERT_PATH,
        as_attachment=True,
        download_name='emotioncam-ca.crt',
        mimetype='application/x-x509-ca-cert',
    )


if __name__ == '__main__':
    local_ip = _get_lan_ip()
    cert_path, key_path = _ensure_https_assets(local_ip)
    ca_server = _start_ca_http_server()
    print(f"\n{'='*50}")
    print(f"  EmotionCam corriendo en:")
    print(f"  PC:       https://localhost:5000")
    print(f"  Teléfono: https://{local_ip}:5000")
    print(f"  (ambos deben estar en el mismo WiFi)")
    print(f"  CA HTTP:  http://{local_ip}:5001/emotioncam-ca.crt")
    print(f"  CA HTTPS: https://{local_ip}:5000/download-ca")
    print(f"{'='*50}\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, ssl_context=(cert_path, key_path))