"""
storage.py — журнал событий (SQLite) и кадры-доказательства.

Требование кейса: сохранять события и поддерживать фильтрацию по типу
нарушения, камере, зоне и времени. Здесь — минимальная реализация
без внешних зависимостей: одна таблица + каталог с jpg-кадрами.

Особенности:
  • SQLite-файл лежит рядом с кодом (data/events.db), а не «где запустили»;
  • на каждую операцию открывается своё соединение (просто и безопасно
    для Streamlit-потока);
  • доказательство сохраняется как jpg рядом с журналом;
  • статусы события: new → confirmed | false_positive (решение оператора).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "data" / "events.db"
DEFAULT_EVIDENCE = BASE_DIR / "data" / "evidence"

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_type TEXT    NOT NULL,             -- код типа
    title          TEXT    NOT NULL,             -- название для оператора
    camera_id      TEXT    NOT NULL,
    zone_id        TEXT    NOT NULL,
    started_at     TEXT    NOT NULL,             -- реальные часы (ISO)
    media_time_s   REAL    NOT NULL,             -- время на записи/в сцене
    frame_id       INTEGER NOT NULL,
    confidence     REAL    NOT NULL,
    status         TEXT    NOT NULL DEFAULT 'new',   -- new|confirmed|false_positive
    evidence_path  TEXT,
    backend        TEXT    NOT NULL              -- demo | linq
);
"""

_STATUS_VALUES = ("new", "confirmed", "false_positive")


def _as_date(value) -> date:
    """Строка 'YYYY-MM-DD' или date → date."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


class EventStorage:
    """Журнал событий. Создаётся один раз, дальше — save/list/update."""

    def __init__(self, db_path=DEFAULT_DB, evidence_dir=DEFAULT_EVIDENCE):
        self.db_path = Path(db_path)
        self.evidence_dir = Path(evidence_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript(_DDL)
            conn.commit()

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------
    def save_event(self, event, backend: str, evidence_frame=None):
        """Сохранить событие; если передан кадр — сохранить как доказательство.

        Возвращает id события или None, если что-то не удалось.
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                "INSERT INTO events (violation_type, title, camera_id, zone_id,"
                " started_at, media_time_s, frame_id, confidence, backend)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event.violation_type, event.title, event.camera_id,
                 event.zone_id, event.started_at.isoformat(),
                 event.media_time_s, event.frame_id, event.confidence,
                 backend),
            )
            event_id = cur.lastrowid
            if evidence_frame is not None:
                path = self.save_evidence(event_id, evidence_frame)
                if path:
                    conn.execute(
                        "UPDATE events SET evidence_path = ? WHERE id = ?",
                        (path, event_id))
                else:
                    # доказательство не сохранилось — событие без него
                    # в журнал не записываем (оператору нужна доказательная база)
                    conn.execute("DELETE FROM events WHERE id = ?",
                                 (event_id,))
                    conn.commit()
                    raise RuntimeError(
                        "Не удалось сохранить кадр-доказательство — "
                        "событие не записано в журнал")
            conn.commit()
            return event_id

    def save_evidence(self, event_id: int, frame_bgr):
        """Записать кадр-доказательство в jpg. Возвращает путь или None."""
        path = self.evidence_dir / f"event_{int(event_id):04d}.jpg"
        ok = cv2.imwrite(str(path), frame_bgr)
        return str(path) if ok else None

    def update_status(self, event_id: int, status: str) -> bool:
        """Оператор подтвердил нарушение или пометил как ложное."""
        if status not in _STATUS_VALUES:
            raise ValueError(f"Недопустимый статус: {status}")
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute("UPDATE events SET status = ? WHERE id = ?",
                               (status, event_id))
            conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------
    def list_events(self, violation_type=None, camera_id=None, zone_id=None,
                   date_from=None, date_to=None) -> list[dict]:
        """Журнал с фильтрами (все фильтры необязательны).

        date_from / date_to — объекты datetime.date или 'YYYY-MM-DD'.
        «По дату» включает весь выбранный день ДО ПОЛУНОЧИ следующего
        (граница строгая, чтобы не терять события с микросекундами
        в последнюю секунду дня).
        """
        query = "SELECT * FROM events WHERE 1=1"
        params: list = []
        if violation_type:
            query += " AND violation_type = ?"
            params.append(violation_type)
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if zone_id:
            query += " AND zone_id = ?"
            params.append(zone_id)
        if date_from:
            query += " AND started_at >= ?"
            params.append(f"{_as_date(date_from)}T00:00:00")
        if date_to:
            # строго меньше полуночи СЛЕДУЮЩЕГО дня
            end = _as_date(date_to) + timedelta(days=1)
            query += " AND started_at < ?"
            params.append(f"{end.isoformat()}T00:00:00")
        query += " ORDER BY id DESC"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def distinct(self, column: str) -> list[str]:
        """Уникальные значения столбца — для выпадающих фильтров."""
        allowed = {"violation_type", "camera_id", "zone_id", "status"}
        if column not in allowed:
            raise ValueError(f"Фильтровать по {column} нельзя")
        with closing(sqlite3.connect(self.db_path)) as conn:
            return [r[0] for r in conn.execute(
                f"SELECT DISTINCT {column} FROM events ORDER BY 1")]

    def stats(self) -> dict:
        """Короткая сводка для панели: всего событий / непросмотренных."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            fresh = conn.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'new'").fetchone()[0]
        return {"total": total, "new": fresh}
