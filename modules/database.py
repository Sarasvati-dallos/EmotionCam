"""
modules/database.py
SQLite para la versión Android.
"""
import sqlite3
import os
from datetime import datetime


class Database:
    def __init__(self):
        try:
            from kivy.utils import platform
        except Exception:
            # Kivy not available (typical on PC/server). Assume non-Android.
            platform = 'linux'

        if platform == 'android':
            try:
                from android.storage import app_storage_path
                db_dir = app_storage_path()
            except Exception:
                # Fallback to local data folder if android.storage isn't available
                db_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        else:
            db_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, 'emotioncam.db')
        self._init()

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self):
        c = self._conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                duration INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)
        c.commit(); c.close()

    def start_session(self):
        c = self._conn()
        cur = c.execute("INSERT INTO sessions (timestamp) VALUES (?)",
                        (datetime.now().isoformat(),))
        sid = cur.lastrowid
        c.commit(); c.close()
        return sid

    def end_session(self, sid, duration):
        c = self._conn()
        c.execute("UPDATE sessions SET duration=? WHERE id=?", (duration, sid))
        c.commit(); c.close()

    def save_detection(self, sid, emotion, confidence):
        c = self._conn()
        c.execute("INSERT INTO detections (session_id,emotion,confidence,timestamp) VALUES (?,?,?,?)",
                  (sid, emotion, confidence, datetime.now().isoformat()))
        c.commit(); c.close()

    def get_all_sessions(self):
        c = self._conn()
        rows = c.execute("SELECT id,timestamp,duration FROM sessions ORDER BY id DESC").fetchall()
        result = []
        for r in rows:
            emo = c.execute(
                "SELECT emotion FROM detections WHERE session_id=? GROUP BY emotion ORDER BY COUNT(*) DESC LIMIT 1",
                (r['id'],)).fetchone()
            mins, secs = divmod(r['duration'], 60)
            result.append({
                'id': r['id'],
                'timestamp': r['timestamp'][:19].replace('T',' '),
                'duration': f"{mins}m {secs}s" if mins else f"{secs}s",
                'dominant': emo['emotion'].capitalize() if emo else 'N/A',
            })
        c.close()
        return result

    def get_emotion_stats(self):
        c = self._conn()
        rows = c.execute(
            "SELECT emotion, COUNT(*) as cnt FROM detections GROUP BY emotion ORDER BY cnt DESC"
        ).fetchall()
        c.close()
        return {r['emotion']: r['cnt'] for r in rows}


# --- Módulo: exposición de API simple y creación perezosa de la BD ---
_db = None

def _get_db():
    global _db
    if _db is None:
        _db = Database()
    return _db


def start_session():
    return _get_db().start_session()


def end_session(sid, duration):
    return _get_db().end_session(sid, duration)


def save_detection(sid, emotion, confidence):
    return _get_db().save_detection(sid, emotion, confidence)


def get_all_sessions():
    return _get_db().get_all_sessions()


def get_emotion_stats():
    return _get_db().get_emotion_stats()
