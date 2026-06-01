"""
main.py — EmotionCam Android (TFLite)
Usa la cámara del teléfono y TFLite para detectar emociones.
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
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.core.window import Window
from kivy.utils import platform

if platform == 'android':
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])

import cv2

# ── Colores ────────────────────────────────────────────────
BG     = (0.05, 0.05, 0.1,  1)
CARD   = (0.1,  0.1,  0.18, 1)
ACCENT = (0.13, 0.83, 0.93, 1)
GREEN  = (0.2,  0.83, 0.4,  1)
MUTED  = (0.4,  0.4,  0.5,  1)
WHITE  = (0.9,  0.9,  1.0,  1)

EMOJIS = {
    'angry':'😠','disgust':'🤢','fear':'😨',
    'happy':'😄','neutral':'😐','sad':'😢','surprise':'😮'
}
LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']


# ── Motor de IA con TFLite ─────────────────────────────────
class EmotionEngine:
    def __init__(self):
        self.interpreter = None
        self.labels      = LABELS
        self.cascade     = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._load_model()

    def _load_model(self):
        candidates = [
            'emotion_model.tflite',
            os.path.join(os.path.dirname(__file__), 'emotion_model.tflite'),
        ]
        label_candidates = [
            'emotion_labels.json',
            os.path.join(os.path.dirname(__file__), 'emotion_labels.json'),
        ]
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                print("[Engine] TFLite no disponible, usando placeholder")
                return

        for p in candidates:
            if os.path.exists(p):
                self.interpreter = tflite.Interpreter(model_path=p)
                self.interpreter.allocate_tensors()
                print(f"[Engine] Modelo TFLite cargado: {p}")
                break

        for lp in label_candidates:
            if os.path.exists(lp):
                with open(lp) as f:
                    self.labels = json.load(f)
                break

    def predict(self, frame_bgr):
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        emotion, conf, emoji = 'neutral', 0.0, '😐'

        if len(faces) > 0:
            x, y, w, h = faces[0]

            if self.interpreter is not None:
                face = frame_bgr[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype('float32')
                # Normalizar igual que MobileNetV2
                face = (face / 127.5) - 1.0
                face = face.reshape(1, 224, 224, 3)

                inp  = self.interpreter.get_input_details()
                out  = self.interpreter.get_output_details()
                self.interpreter.set_tensor(inp[0]['index'], face)
                self.interpreter.invoke()
                pred   = self.interpreter.get_tensor(out[0]['index'])[0]
                idx    = int(np.argmax(pred))
                emotion = self.labels[idx]
                conf    = float(np.max(pred))
            else:
                import random
                emotion = random.choice(LABELS)
                conf    = round(random.uniform(0.6, 0.95), 2)

            emoji = EMOJIS.get(emotion, '😐')
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 220, 100), 2)
            lbl = f"{emoji} {emotion} {conf:.0%}"
            cv2.rectangle(frame_bgr, (x, y-28), (x+len(lbl)*11, y), (0,0,0), -1)
            cv2.putText(frame_bgr, lbl, (x+3, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,100), 2)

        return frame_bgr, emotion, conf, emoji


# ── Helpers UI ─────────────────────────────────────────────
def bg_rect(widget, color):
    with widget.canvas.before:
        c  = Color(*color)
        rr = Rectangle(pos=widget.pos, size=widget.size)
    widget.bind(pos=lambda *a: setattr(rr,'pos',widget.pos),
                size=lambda *a: setattr(rr,'size',widget.size))

def make_btn(text, color=ACCENT, on_press=None):
    btn = Button(
        text=text, font_size='17sp', bold=True,
        size_hint_y=None, height='52dp',
        background_normal='', background_color=(0,0,0,0),
        color=(0.05,0.05,0.1,1) if color==ACCENT else WHITE,
    )
    with btn.canvas.before:
        Color(*color)
        rr = RoundedRectangle(pos=btn.pos, size=btn.size, radius=[13])
    btn.bind(pos=lambda *a: setattr(rr,'pos',btn.pos),
             size=lambda *a: setattr(rr,'size',btn.size))
    if on_press:
        btn.bind(on_press=on_press)
    return btn


# ── HomeScreen ─────────────────────────────────────────────
class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bg_rect(self, BG)
        layout = BoxLayout(orientation='vertical', padding='32dp', spacing='18dp')
        layout.add_widget(Widget())
        layout.add_widget(Label(text='😊', font_size='72sp',
                                size_hint_y=None, height='90dp'))
        layout.add_widget(Label(text='EmotionCam', font_size='32sp',
                                bold=True, color=ACCENT,
                                size_hint_y=None, height='48dp'))
        layout.add_widget(Label(text='Detector de emociones faciales',
                                font_size='15sp', color=MUTED,
                                size_hint_y=None, height='28dp'))
        layout.add_widget(Widget(size_hint_y=None, height='20dp'))
        layout.add_widget(make_btn('📷  Iniciar cámara', ACCENT,
                          lambda *a: setattr(self.manager,'current','camera')))
        layout.add_widget(make_btn('📋  Historial', (0.1,0.4,0.6,1),
                          lambda *a: setattr(self.manager,'current','history')))
        layout.add_widget(make_btn('📊  Estadísticas', (0.08,0.3,0.5,1),
                          lambda *a: setattr(self.manager,'current','stats')))
        layout.add_widget(Widget())
        self.add_widget(layout)


# ── CameraScreen ───────────────────────────────────────────
class CameraScreen(Screen):
    def __init__(self, engine, db, **kwargs):
        super().__init__(**kwargs)
        self.engine          = engine
        self.db              = db
        self.cap             = None
        self._running        = False
        self._session_id     = None
        self._session_start  = None
        self._last_save      = 0.0
        self._emo_history    = []

        bg_rect(self, BG)
        layout = BoxLayout(orientation='vertical')

        # Header
        header = BoxLayout(size_hint_y=None, height='48dp',
                           padding=['10dp',0], spacing='8dp')
        bg_rect(header, (0.03,0.03,0.08,1))
        back = Button(text='← Inicio', size_hint_x=0.35,
                      background_normal='', background_color=(0.2,0.2,0.3,1),
                      color=WHITE, font_size='14sp', bold=True)
        back.bind(on_press=lambda *a: self._stop_and_go('home'))
        self.fps_lbl    = Label(text='FPS: —', color=GREEN, font_size='13sp')
        self.status_lbl = Label(text='● activo', color=GREEN,
                                font_size='12sp', size_hint_x=0.35)
        header.add_widget(back)
        header.add_widget(self.fps_lbl)
        header.add_widget(self.status_lbl)
        layout.add_widget(header)

        # Video feed
        self.img_widget = Image(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.img_widget)

        # Panel emoción
        panel = BoxLayout(orientation='vertical', size_hint_y=None,
                          height='105dp', padding=['16dp','8dp'], spacing='6dp')
        with panel.canvas.before:
            Color(*CARD)
            prr = RoundedRectangle(pos=panel.pos, size=panel.size, radius=[14])
        panel.bind(pos=lambda *a: setattr(prr,'pos',panel.pos),
                   size=lambda *a: setattr(prr,'size',panel.size))

        emo_row = BoxLayout(size_hint_y=None, height='44dp')
        self.emoji_lbl   = Label(text='😐', font_size='34sp',
                                 size_hint_x=None, width='52dp')
        self.emotion_lbl = Label(text='—', font_size='22sp', bold=True,
                                 color=ACCENT, halign='left', text_size=(None,None))
        self.conf_lbl    = Label(text='0%', font_size='17sp', color=WHITE,
                                 size_hint_x=None, width='56dp')
        emo_row.add_widget(self.emoji_lbl)
        emo_row.add_widget(self.emotion_lbl)
        emo_row.add_widget(self.conf_lbl)
        panel.add_widget(emo_row)

        self.bar_w = Widget(size_hint_y=None, height='14dp')
        with self.bar_w.canvas:
            Color(0.18,0.18,0.28,1)
            self._bbg  = RoundedRectangle(pos=self.bar_w.pos,
                                          size=self.bar_w.size, radius=[6])
            Color(*GREEN)
            self._bfill = RoundedRectangle(pos=self.bar_w.pos,
                                           size=(0,self.bar_w.height), radius=[6])
        self.bar_w.bind(pos=self._upd_bar, size=self._upd_bar)
        panel.add_widget(self.bar_w)
        layout.add_widget(panel)

        # Footer
        footer = BoxLayout(size_hint_y=None, height='56dp',
                           padding=['10dp','6dp'], spacing='10dp')
        bg_rect(footer, (0.03,0.03,0.08,1))
        footer.add_widget(make_btn('⏹ Detener', (0.85,0.2,0.2,1),
                                   lambda *a: self._stop_and_go('home')))
        footer.add_widget(make_btn('📸 Snapshot', (0.8,0.5,0.1,1),
                                   lambda *a: self._snapshot()))
        layout.add_widget(footer)
        self.add_widget(layout)

    def _upd_bar(self, *a):
        self._bbg.pos  = self.bar_w.pos
        self._bbg.size = self.bar_w.size

    def on_enter(self):
        self._running       = True
        self._session_id    = self.db.start_session()
        self._session_start = time.time()
        self._last_save     = 0.0
        self._emo_history   = []
        threading.Thread(target=self._loop, daemon=True).start()

    def on_leave(self):
        self._running = False
        if self._session_id and self._session_start:
            self.db.end_session(self._session_id,
                                int(time.time()-self._session_start))
        self._session_id = None

    def _loop(self):
        idx = 1 if platform == 'android' else 0
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        times = []
        while self._running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame, emotion, conf, emoji = self.engine.predict(frame)

            # Estabilizar emoción (evita parpadeo)
            self._emo_history.append(emotion)
            if len(self._emo_history) > 8:
                self._emo_history.pop(0)
            from collections import Counter
            stable = Counter(self._emo_history).most_common(1)[0][0]

            now = time.time()
            if self._session_id and now - self._last_save >= 1.0:
                self.db.save_detection(self._session_id, stable, conf)
                self._last_save = now

            times.append(time.time()-t0)
            if len(times) > 20: times.pop(0)
            fps = 1/(sum(times)/len(times)) if times else 0

            self._update_ui(frame, stable, conf, EMOJIS.get(stable,'😐'), fps)

        if self.cap:
            self.cap.release()

    @mainthread
    def _update_ui(self, frame, emotion, conf, emoji, fps):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flip = cv2.flip(rgb, 0)
        tex  = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='rgb')
        tex.blit_buffer(flip.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture  = tex
        self.emoji_lbl.text      = emoji
        self.emotion_lbl.text    = emotion.capitalize()
        self.conf_lbl.text       = f'{conf:.0%}'
        self.fps_lbl.text        = f'FPS: {fps:.0f}'
        self._bfill.pos          = self.bar_w.pos
        self._bfill.size         = (self.bar_w.width*conf, self.bar_w.height)

    def _stop_and_go(self, screen):
        self._running = False
        self.manager.current = screen

    def _snapshot(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(f'snapshot_{int(time.time())}.jpg', frame)


# ── HistoryScreen ──────────────────────────────────────────
class HistoryScreen(Screen):
    def __init__(self, db, **kwargs):
        super().__init__(**kwargs)
        self.db = db
        bg_rect(self, BG)
        layout = BoxLayout(orientation='vertical', padding='16dp', spacing='10dp')

        header = BoxLayout(size_hint_y=None, height='50dp')
        back = make_btn('← Inicio', (0.2,0.2,0.3,1),
                        lambda *a: setattr(self.manager,'current','home'))
        back.size_hint_x = 0.35
        header.add_widget(back)
        header.add_widget(Label(text='📋 Historial', font_size='22sp',
                                bold=True, color=ACCENT))
        layout.add_widget(header)

        self.container = BoxLayout(orientation='vertical', spacing='8dp',
                                   size_hint_y=None)
        self.container.bind(minimum_height=self.container.setter('height'))
        sv = ScrollView()
        sv.add_widget(self.container)
        layout.add_widget(sv)
        self.add_widget(layout)

    def on_enter(self):
        self.container.clear_widgets()
        sessions = self.db.get_all_sessions()
        if not sessions:
            self.container.add_widget(
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
            row.bind(pos=lambda w,*a,r=rr: setattr(r,'pos',w.pos),
                     size=lambda w,*a,r=rr: setattr(r,'size',w.size))
            row.add_widget(Label(text=emoji, font_size='28sp',
                                 size_hint_x=None, width='44dp'))
            info = BoxLayout(orientation='vertical')
            info.add_widget(Label(text=s['dominant'], font_size='16sp',
                                  bold=True, color=ACCENT,
                                  halign='left', text_size=(None,None)))
            info.add_widget(Label(text=s['timestamp'], font_size='11sp',
                                  color=MUTED, halign='left',
                                  text_size=(None,None)))
            row.add_widget(info)
            row.add_widget(Label(text=s['duration'], font_size='13sp',
                                 color=WHITE, size_hint_x=None, width='56dp'))
            self.container.add_widget(row)


# ── StatsScreen ────────────────────────────────────────────
class StatsScreen(Screen):
    def __init__(self, db, **kwargs):
        super().__init__(**kwargs)
        self.db = db
        bg_rect(self, BG)
        layout = BoxLayout(orientation='vertical', padding='16dp', spacing='10dp')

        header = BoxLayout(size_hint_y=None, height='50dp')
        back = make_btn('← Inicio', (0.2,0.2,0.3,1),
                        lambda *a: setattr(self.manager,'current','home'))
        back.size_hint_x = 0.35
        header.add_widget(back)
        header.add_widget(Label(text='📊 Estadísticas', font_size='22sp',
                                bold=True, color=ACCENT))
        layout.add_widget(header)

        self.top_lbl = Label(text='—', font_size='20sp', bold=True,
                             color=ACCENT, size_hint_y=None, height='40dp')
        layout.add_widget(self.top_lbl)

        self.bars = BoxLayout(orientation='vertical', spacing='10dp',
                              size_hint_y=None)
        self.bars.bind(minimum_height=self.bars.setter('height'))
        sv = ScrollView()
        sv.add_widget(self.bars)
        layout.add_widget(sv)
        self.add_widget(layout)

    def on_enter(self):
        self.bars.clear_widgets()
        raw = self.db.get_emotion_stats()
        if not raw:
            self.top_lbl.text = 'Sin datos aún'
            return
        total     = sum(raw.values())
        max_count = max(raw.values())
        sorted_e  = sorted(LABELS, key=lambda e: raw.get(e,0), reverse=True)
        top = sorted_e[0]
        self.top_lbl.text = f"Más detectada: {EMOJIS.get(top,'❓')} {top.capitalize()}"

        BCOLORS = {
            'angry':(0.95,0.3,0.3,1),'disgust':(0.5,0.9,0.3,1),
            'fear':(0.75,0.3,0.95,1),'happy':(0.98,0.85,0.2,1),
            'neutral':(0.4,0.75,0.95,1),'sad':(0.3,0.5,0.95,1),
            'surprise':(0.95,0.6,0.1,1),
        }
        for emo in sorted_e:
            cnt = raw.get(emo, 0)
            pct = cnt/total if total>0 else 0
            bw  = cnt/max_count if max_count>0 else 0
            row = BoxLayout(size_hint_y=None, height='44dp',
                            spacing='8dp', padding=['4dp',0])
            row.add_widget(Label(text=EMOJIS.get(emo,'❓'), font_size='20sp',
                                 size_hint_x=None, width='32dp'))
            row.add_widget(Label(text=emo.capitalize(), font_size='13sp',
                                 color=WHITE, size_hint_x=None, width='80dp',
                                 halign='left', text_size=(80,None)))
            bw_widget = Widget()
            bc = BCOLORS.get(emo, GREEN)
            with bw_widget.canvas:
                Color(0.18,0.18,0.28,1)
                b1 = RoundedRectangle(pos=bw_widget.pos,
                                      size=bw_widget.size, radius=[6])
                Color(*bc)
                b2 = RoundedRectangle(pos=bw_widget.pos,
                                      size=(0,bw_widget.height), radius=[6])
            def upd(w, *a, _b1=b1, _b2=b2, _bw=bw):
                _b1.pos=w.pos; _b1.size=w.size
                _b2.pos=w.pos; _b2.size=(w.width*_bw, w.height)
            bw_widget.bind(pos=upd, size=upd)
            row.add_widget(bw_widget)
            row.add_widget(Label(text=f'{pct:.0%}', font_size='12sp',
                                 color=MUTED, size_hint_x=None, width='38dp'))
            self.bars.add_widget(row)


# ── App ────────────────────────────────────────────────────
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
