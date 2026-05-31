"""
modules/database.py
Persistencia SQLite para sesiones y detecciones.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'emotioncam.db')


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            duration  INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS detections (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            emotion    TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp  TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    c.commit(); c.close()


def start_session():
    c = _conn()
    cur = c.execute("INSERT INTO sessions (timestamp) VALUES (?)",
                    (datetime.now().isoformat(),))
    sid = cur.lastrowid
    c.commit(); c.close()
    return sid


def end_session(sid, duration):
    c = _conn()
    c.execute("UPDATE sessions SET duration=? WHERE id=?", (duration, sid))
    c.commit(); c.close()


def save_detection(sid, emotion, confidence):
    c = _conn()
    c.execute("INSERT INTO detections (session_id,emotion,confidence,timestamp) VALUES (?,?,?,?)",
              (sid, emotion, confidence, datetime.now().isoformat()))
    c.commit(); c.close()


def get_all_sessions():
    c = _conn()
    rows = c.execute("SELECT id,timestamp,duration FROM sessions ORDER BY id DESC").fetchall()
    result = []
    for r in rows:
        emo = c.execute(
            "SELECT emotion FROM detections WHERE session_id=? GROUP BY emotion ORDER BY COUNT(*) DESC LIMIT 1",
            (r['id'],)).fetchone()
        mins, secs = divmod(r['duration'], 60)
        result.append({
            'id': r['id'],
            'timestamp': r['timestamp'][:19].replace('T', ' '),
            'duration': f"{mins}m {secs}s" if mins else f"{secs}s",
            'dominant': emo['emotion'].capitalize() if emo else 'N/A',
        })
    c.close()
    return result


def get_emotion_stats():
    c = _conn()
    rows = c.execute(
        "SELECT emotion, COUNT(*) as cnt FROM detections GROUP BY emotion ORDER BY cnt DESC"
    ).fetchall()
    c.close()
    return {r['emotion']: r['cnt'] for r in rows}


init_db()
