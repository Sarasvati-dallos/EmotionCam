"""
main.py — EmotionCam Android
Usa la cámara del TELÉFONO, detecta emociones con el modelo de Yeison.
"""

import os
import json
import time
import threading
import numpy as np

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.core.window import Window
from kivy.utils import platform

# Android: pedir permisos de cámara
if platform == 'android':
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])

import cv2

# ── Colores ────────────────────────────────────────────────
BG      = (0.05, 0.05, 0.1, 1)
CARD    = (0.1,  0.1,  0.18, 1)
ACCENT  = (0.13, 0.83, 0.93, 1)
GREEN   = (0.2,  0.83, 0.4,  1)
MUTED   = (0.4,  0.4,  0.5,  1)
WHITE   = (0.9,  0.9,  1,    1)

EMOJIS = {
    'angry':'😠','disgust':'🤢','fear':'😨',
    'happy':'😄','neutral':'😐','sad':'😢','surprise':'😮'
}

# ── Clasificador ───────────────────────────────────────────
class EmotionEngine:
    def __init__(self):
        self.model     = None
        self.labels    = []
        self.cascade   = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._load_model()

    def _load_model(self):
        paths = [
            'emotion_model.h5',
            os.path.join(os.path.dirname(__file__), 'emotion_model.h5'),
            '/sdcard/emotion_model.h5',
        ]
        label_paths = [
            'emotion_labels.json',
            os.path.join(os.path.dirname(__file__), 'emotion_labels.json'),
        ]
        try:
            import tensorflow as tf
            for p in paths:
                if os.path.exists(p):
                    self.model = tf.keras.models.load_model(p)
                    print(f"[Engine] Modelo cargado: {p}")
                    break
            for lp in label_paths:
                if os.path.exists(lp):
                    with open(lp) as f:
                        self.labels = json.load(f)
                    break
        except Exception as e:
            print(f"[Engine] Error cargando modelo: {e}")

    def predict(self, frame_bgr):
        """
        Detecta caras y predice emoción.
        Retorna: (frame_con_box, emotion, confidence, emoji)
        """
        faces = self.cascade.detectMultiScale(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        emotion, conf, emoji = 'neutral', 0.0, '😐'

        if len(faces) > 0 and self.model is not None:
            import tensorflow as tf
            x, y, w, h = faces[0]
            face = frame_bgr[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype('float32')
            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
            pred = self.model.predict(face.reshape(1,224,224,3), verbose=0)[0]
            idx  = int(np.argmax(pred))
            emotion = self.labels[idx] if self.labels else 'happy'
            conf    = float(np.max(pred))
            emoji   = EMOJIS.get(emotion, '😐')
            # Dibujar bounding box
            cv2.rectangle(frame_bgr, (x,y), (x+w,y+h), (0,220,100), 2)
            label = f"{emoji} {emotion} {conf:.0%}"
            cv2.rectangle(frame_bgr, (x, y-28), (x+len(label)*11, y), (0,0,0), -1)
            cv2.putText(frame_bgr, label, (x+3,y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,100), 2)
        elif len(faces) > 0:
            # placeholder
            import random
            emotion = random.choice(list(EMOJIS.keys()))
            conf    = round(random.uniform(0.6, 0.95), 2)
            emoji   = EMOJIS[emotion]
            x,y,w,h = faces[0]
            cv2.rectangle(frame_bgr,(x,y),(x+w,y+h),(0,220,100),2)

        return frame_bgr, emotion, conf, emoji


# ── Base screen con fondo ──────────────────────────────────
class BaseScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(*BG)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=lambda *a: setattr(self._bg, 'pos', self.pos),
                  size=lambda *a: setattr(self._bg, 'size', self.size))


def make_btn(text, color=ACCENT, on_press=None):
    btn = Button(
        text=text, font_size='17sp', bold=True,
        size_hint_y=None, height='52dp',
        background_normal='', background_color=(0,0,0,0),
        color=(0.05,0.05,0.1,1) if color==ACCENT else WHITE,
    )
    with btn.canvas.before:
        c = Color(*color)
        rr = RoundedRectangle(pos=btn.pos, size=btn.size, radius=[13])
    btn.bind(pos=lambda *a: setattr(rr,'pos',btn.pos),
             size=lambda *a: setattr(rr,'size',btn.size))
    if on_press:
        btn.bind(on_press=on_press)
    return btn


# ── HomeScreen ─────────────────────────────────────────────
class HomeScreen(BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding='32dp', spacing='18dp')

        layout.add_widget(Widget())
        layout.add_widget(Label(text='😊', font_size='72sp', size_hint_y=None, height='90dp'))
        layout.add_widget(Label(text='EmotionCam', font_size='32sp', bold=True,
                                color=ACCENT, size_hint_y=None, height='48dp'))
        layout.add_widget(Label(text='Detector de emociones faciales',
                                font_size='15sp', color=MUTED,
                                size_hint_y=None, height='28dp'))
        layout.add_widget(Widget(size_hint_y=None, height='20dp'))
        layout.add_widget(make_btn('📷  Iniciar cámara', ACCENT,
                                   lambda *a: setattr(self.manager, 'current', 'camera')))
        layout.add_widget(make_btn('📋  Historial', (0.1,0.4,0.6,1),
                                   lambda *a: setattr(self.manager, 'current', 'history')))
        layout.add_widget(make_btn('📊  Estadísticas', (0.08,0.3,0.5,1),
                                   lambda *a: setattr(self.manager, 'current', 'stats')))
        layout.add_widget(Widget())
        self.add_widget(layout)


# ── CameraScreen ───────────────────────────────────────────
class CameraScreen(BaseScreen):
    def __init__(self, engine, db, **kwargs):
        super().__init__(**kwargs)
        self.engine     = engine
        self.db         = db
        self.cap        = None
        self._running   = False
        self._thread    = None
        self._session_id   = None
        self._session_start = None
        self._last_save    = 0.0

        # Estabilizador de emoción (evita parpadeo)
        self._emo_history = []
        self._stable_emotion = 'neutral'
        self._stable_conf    = 0.0
        self._stable_emoji   = '😐'

        layout = BoxLayout(orientation='vertical')

        # Header
        header = BoxLayout(size_hint_y=None, height='48dp',
                           padding=['10dp',0], spacing='8dp')
        with header.canvas.before:
            Color(0.03,0.03,0.08,1)
            self._hbg = Rectangle(pos=header.pos, size=header.size)
        header.bind(pos=lambda *a: setattr(self._hbg,'pos',header.pos),
                    size=lambda *a: setattr(self._hbg,'size',header.size))
        back_btn = Button(text='← Inicio', size_hint_x=0.35,
                          background_normal='', background_color=(0.2,0.2,0.3,1),
                          color=WHITE, font_size='14sp', bold=True)
        back_btn.bind(on_press=lambda *a: self._stop_and_go('home'))
        self.fps_lbl = Label(text='FPS: —', color=GREEN, font_size='13sp')
        self.status_lbl = Label(text='● inactivo', color=MUTED, font_size='12sp', size_hint_x=0.35)
        header.add_widget(back_btn)
        header.add_widget(self.fps_lbl)
        header.add_widget(self.status_lbl)
        layout.add_widget(header)

        # Video
        self.img_widget = Image(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.img_widget)

        # Panel emoción
        panel = BoxLayout(orientation='vertical', size_hint_y=None,
                          height='105dp', padding=['16dp','8dp'], spacing='6dp')
        with panel.canvas.before:
            Color(*CARD)
            self._pbg = RoundedRectangle(pos=panel.pos, size=panel.size, radius=[14])
        panel.bind(pos=lambda *a: setattr(self._pbg,'pos',panel.pos),
                   size=lambda *a: setattr(self._pbg,'size',panel.size))

        emo_row = BoxLayout(size_hint_y=None, height='44dp')
        self.emoji_lbl   = Label(text='😐', font_size='34sp', size_hint_x=None, width='52dp')
        self.emotion_lbl = Label(text='—', font_size='22sp', bold=True, color=ACCENT,
                                 halign='left', text_size=(None,None))
        self.conf_lbl    = Label(text='0%', font_size='17sp', color=WHITE,
                                 size_hint_x=None, width='56dp')
        emo_row.add_widget(self.emoji_lbl)
        emo_row.add_widget(self.emotion_lbl)
        emo_row.add_widget(self.conf_lbl)
        panel.add_widget(emo_row)

        # Barra confianza
        self.bar_container = Widget(size_hint_y=None, height='14dp')
        with self.bar_container.canvas:
            Color(0.18,0.18,0.28,1)
            self._bar_bg   = RoundedRectangle(pos=self.bar_container.pos,
                                               size=self.bar_container.size, radius=[6])
            Color(*GREEN)
            self._bar_fill = RoundedRectangle(pos=self.bar_container.pos,
                                               size=(0, self.bar_container.height), radius=[6])
        self.bar_container.bind(pos=self._update_bar, size=self._update_bar)
        panel.add_widget(self.bar_container)
        layout.add_widget(panel)

        # Footer botones
        footer = BoxLayout(size_hint_y=None, height='56dp',
                           padding=['10dp','6dp'], spacing='10dp')
        with footer.canvas.before:
            Color(0.03,0.03,0.08,1)
            self._fbg = Rectangle(pos=footer.pos, size=footer.size)
        footer.bind(pos=lambda *a: setattr(self._fbg,'pos',footer.pos),
                    size=lambda *a: setattr(self._fbg,'size',footer.size))
        stop_btn = make_btn('⏹ Detener', (0.85,0.2,0.2,1), lambda *a: self._stop_and_go('home'))
        snap_btn = make_btn('📸 Snapshot', (0.8,0.5,0.1,1), lambda *a: self._snapshot())
        footer.add_widget(stop_btn)
        footer.add_widget(snap_btn)
        layout.add_widget(footer)

        self.add_widget(layout)

    def _update_bar(self, *args):
        self._bar_bg.pos  = self.bar_container.pos
        self._bar_bg.size = self.bar_container.size

    def on_enter(self):
        self._running = True
        self.status_lbl.text  = '● activo'
        self.status_lbl.color = GREEN
        self._session_id    = self.db.start_session()
        self._session_start = time.time()
        self._last_save     = 0.0
        self._emo_history   = []
        # Abrir cámara en hilo separado
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

    def on_leave(self):
        self._running = False
        if self._session_id and self._session_start:
            self.db.end_session(self._session_id, int(time.time()-self._session_start))
        self._session_id = None

    def _camera_loop(self):
        # En Android usar cámara frontal (índice 1), en PC usar 0
        cam_idx = 1 if platform == 'android' else 0
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        t_frames = []
        while self._running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame, emotion, conf, emoji = self.engine.predict(frame)

            # Estabilizar: guardar últimas 8 predicciones
            self._emo_history.append(emotion)
            if len(self._emo_history) > 8:
                self._emo_history.pop(0)
            # La emoción estable es la más frecuente
            from collections import Counter
            most_common = Counter(self._emo_history).most_common(1)[0][0]
            self._stable_emotion = most_common
            self._stable_conf    = conf
            self._stable_emoji   = EMOJIS.get(most_common, '😐')

            # Guardar en DB máx 1 vez/segundo
            now = time.time()
            if self._session_id and now - self._last_save >= 1.0:
                self.db.save_detection(self._session_id, most_common, conf)
                self._last_save = now

            # FPS
            t_frames.append(time.time()-t0)
            if len(t_frames) > 20: t_frames.pop(0)
            fps = 1/( sum(t_frames)/len(t_frames) ) if t_frames else 0

            # Actualizar UI en hilo principal
            self._update_ui(frame, most_common, conf,
                            EMOJIS.get(most_common,'😐'), fps)

        if self.cap:
            self.cap.release()
            self.cap = None

    @mainthread
    def _update_ui(self, frame, emotion, conf, emoji, fps):
        # Textura
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_flip = cv2.flip(frame_rgb, 0)
        buf = frame_flip.tobytes()
        tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        tex.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = tex

        # Labels
        self.emoji_lbl.text   = emoji
        self.emotion_lbl.text = emotion.capitalize()
        self.conf_lbl.text    = f'{conf:.0%}'
        self.fps_lbl.text     = f'FPS: {fps:.0f}'

        # Barra
        self._bar_fill.pos  = self.bar_container.pos
        self._bar_fill.size = (self.bar_container.width * conf, self.bar_container.height)

    def _stop_and_go(self, screen):
        self._running = False
        self.manager.current = screen

    def _snapshot(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                name = f'snapshot_{int(time.time())}.jpg'
                cv2.imwrite(name, frame)
                print(f'[Snapshot] Guardado: {name}')


# ── HistoryScreen ──────────────────────────────────────────
class HistoryScreen(BaseScreen):
    def __init__(self, db, **kwargs):
        super().__init__(**kwargs)
        self.db = db
        self.layout = BoxLayout(orientation='vertical', padding='16dp', spacing='10dp')

        header = BoxLayout(size_hint_y=None, height='50dp')
        back = make_btn('← Inicio', (0.2,0.2,0.3,1),
                        lambda *a: setattr(self.manager,'current','home'))
        back.size_hint_x = 0.35
        header.add_widget(back)
        header.add_widget(Label(text='📋 Historial', font_size='22sp',
                                bold=True, color=ACCENT))
        self.layout.add_widget(header)

        from kivy.uix.scrollview import ScrollView
        self.scroll_content = BoxLayout(orientation='vertical', spacing='8dp',
                                        size_hint_y=None)
        self.scroll_content.bind(minimum_height=self.scroll_content.setter('height'))
        sv = ScrollView()
        sv.add_widget(self.scroll_content)
        self.layout.add_widget(sv)
        self.add_widget(self.layout)

    def on_enter(self):
        self.scroll_content.clear_widgets()
        sessions = self.db.get_all_sessions()
        if not sessions:
            self.scroll_content.add_widget(
                Label(text='Sin sesiones aún.\n¡Abre la cámara!',
                      color=MUTED, font_size='16sp',
                      size_hint_y=None, height='80dp', halign='center'))
            return
        for s in sessions:
            emoji = EMOJIS.get(s['dominant'].lower(), '❓')
            row = BoxLayout(size_hint_y=None, height='70dp',
                            padding=['12dp','8dp'], spacing='10dp')
            with row.canvas.before:
                Color(*CARD)
                rr = RoundedRectangle(pos=row.pos, size=row.size, radius=[10])
            row.bind(pos=lambda w,*a: setattr(rr,'pos',w.pos),
                     size=lambda w,*a: setattr(rr,'size',w.size))
            row.add_widget(Label(text=emoji, font_size='28sp',
                                 size_hint_x=None, width='44dp'))
            info = BoxLayout(orientation='vertical')
            info.add_widget(Label(text=s['dominant'], font_size='16sp',
                                  bold=True, color=ACCENT, halign='left',
                                  text_size=(None,None)))
            info.add_widget(Label(text=s['timestamp'], font_size='11sp',
                                  color=MUTED, halign='left', text_size=(None,None)))
            row.add_widget(info)
            row.add_widget(Label(text=s['duration'], font_size='13sp',
                                 color=WHITE, size_hint_x=None, width='56dp'))
            self.scroll_content.add_widget(row)


# ── StatsScreen ────────────────────────────────────────────
class StatsScreen(BaseScreen):
    def __init__(self, db, **kwargs):
        super().__init__(**kwargs)
        self.db = db
        self.layout = BoxLayout(orientation='vertical', padding='16dp', spacing='10dp')

        header = BoxLayout(size_hint_y=None, height='50dp')
        back = make_btn('← Inicio', (0.2,0.2,0.3,1),
                        lambda *a: setattr(self.manager,'current','home'))
        back.size_hint_x = 0.35
        header.add_widget(back)
        header.add_widget(Label(text='📊 Estadísticas', font_size='22sp',
                                bold=True, color=ACCENT))
        self.layout.add_widget(header)

        self.top_label = Label(text='—', font_size='20sp', bold=True,
                               color=ACCENT, size_hint_y=None, height='40dp')
        self.layout.add_widget(self.top_label)

        from kivy.uix.scrollview import ScrollView
        self.bars_box = BoxLayout(orientation='vertical', spacing='10dp',
                                  size_hint_y=None)
        self.bars_box.bind(minimum_height=self.bars_box.setter('height'))
        sv = ScrollView()
        sv.add_widget(self.bars_box)
        self.layout.add_widget(sv)
        self.add_widget(self.layout)

    def on_enter(self):
        self.bars_box.clear_widgets()
        raw   = self.db.get_emotion_stats()
        if not raw:
            self.top_label.text = 'Sin datos aún'
            return
        total     = sum(raw.values())
        max_count = max(raw.values())
        all_emo   = ['angry','disgust','fear','happy','neutral','sad','surprise']
        sorted_e  = sorted(all_emo, key=lambda e: raw.get(e,0), reverse=True)
        self.top_label.text = f"Más detectada: {EMOJIS.get(sorted_e[0],'❓')} {sorted_e[0].capitalize()}"

        BAR_COLORS = {
            'angry':(0.95,0.3,0.3,1),'disgust':(0.5,0.9,0.3,1),
            'fear':(0.75,0.3,0.95,1),'happy':(0.98,0.85,0.2,1),
            'neutral':(0.4,0.75,0.95,1),'sad':(0.3,0.5,0.95,1),
            'surprise':(0.95,0.6,0.1,1),
        }
        for emo in sorted_e:
            cnt  = raw.get(emo, 0)
            pct  = cnt/total if total>0 else 0
            bw   = cnt/max_count if max_count>0 else 0
            row  = BoxLayout(size_hint_y=None, height='44dp', spacing='8dp',
                             padding=['4dp',0])
            row.add_widget(Label(text=EMOJIS.get(emo,'❓'), font_size='20sp',
                                 size_hint_x=None, width='32dp'))
            row.add_widget(Label(text=emo.capitalize(), font_size='13sp',
                                 color=WHITE, size_hint_x=None, width='80dp',
                                 halign='left', text_size=(80,None)))
            bar_w = Widget()
            bc = BAR_COLORS.get(emo, GREEN)
            with bar_w.canvas:
                Color(0.18,0.18,0.28,1)
                bg_rr = RoundedRectangle(pos=bar_w.pos, size=bar_w.size, radius=[6])
                Color(*bc)
                fill_rr = RoundedRectangle(pos=bar_w.pos,
                                           size=(bar_w.width*bw, bar_w.height), radius=[6])
            def update_bar(w, *a, _bw=bw, _bg=bg_rr, _fill=fill_rr):
                _bg.pos=w.pos; _bg.size=w.size
                _fill.pos=w.pos; _fill.size=(w.width*_bw, w.height)
            bar_w.bind(pos=update_bar, size=update_bar)
            row.add_widget(bar_w)
            row.add_widget(Label(text=f'{pct:.0%}', font_size='12sp',
                                 color=MUTED, size_hint_x=None, width='38dp'))
            self.bars_box.add_widget(row)


# ── App principal ──────────────────────────────────────────
class EmotionCamApp(App):
    def build(self):
        Window.clearcolor = BG

        from modules.database import Database
        self.db     = Database()
        self.engine = EmotionEngine()

        sm = ScreenManager(transition=FadeTransition(duration=0.15))
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(CameraScreen(engine=self.engine, db=self.db, name='camera'))
        sm.add_widget(HistoryScreen(db=self.db, name='history'))
        sm.add_widget(StatsScreen(db=self.db, name='stats'))
        return sm


if __name__ == '__main__':
    EmotionCamApp().run()
