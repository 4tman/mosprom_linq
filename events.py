"""
events.py — подтверждение нарушений и подавление повторов.

Ключевая идея кейса: детектор что-то нашёл — это ещё НЕ нарушение.
Было ли нарушение на самом деле, решает этот модуль:

  1. Попадает ли найденный объект в запрещённую зону (полигон на кадре)?
  2. Длится ли ситуация достаточно долго (один случайный кадр не считается)?
  3. Если да — создаём РОВНО ОДНО событие и не дублируем его, пока
     нарушение продолжается.

Модуль — «чистая логика»: он не знает ни про Streamlit, ни про LinQ,
ни про базу данных. Его можно тестировать отдельно (см. tests/).

Автомат состояний зоны:

    CLEAR (норма)
       │ человек появился в зоне
       ▼
    CONFIRMING (проверяем, не разовый ли кадр)
       │  НЕПРЕРЫВНО пробыл в зоне >= confirm_seconds
       ▼
    ACTIVE (НАРУШЕНИЕ подтверждено → одно событие)
       │ зона пуста >= clear_seconds
       ▼
    CLEAR

Правила толерантности к пропускам детектора:
  • CONFIRMING сбрасывается при пропадании — подтверждение требует
    НЕПРЕРЫВНОГО присутствия (две короткие детекции с паузой не
    «склеиваются» в одно нарушение);
  • ACTIVE переживает короткие пропуски — чтобы длящееся нарушение
    не порождало дубликаты, если детектор моргнул.

Время берём «медиа»-время (время на записи), а не часы обработки,
чтобы поведение не зависело от скорости проигрывания и инференса.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ViolationRule:
    """Правило одного типа нарушения.

    zone — полигон в ДОЛЯХ кадра [(x, y), ...], координаты 0..1.
    Такой формат не зависит от разрешения камеры.
    """
    violation_type: str              # код для журнала, напр. "restricted_zone"
    title: str                       # человекочитаемое название
    labels: tuple                    # какие метки детектора считаются объектом
    zone: list                       # полигон в нормализованных координатах
    confirm_seconds: float = 2.0     # сколько секунд держим подтверждение
    clear_seconds: float = 4.0       # сколько секунд зона должна быть пустой


DEFAULT_RULE = ViolationRule(
    violation_type="restricted_zone",
    title="Нахождение человека в опасной зоне",
    labels=("person", "worker"),
    zone=[(0.40, 0.30), (0.72, 0.30), (0.72, 0.90), (0.40, 0.90)],
)


@dataclass
class Event:
    """Подтверждённое нарушение — то, что попадёт в журнал."""
    violation_type: str
    title: str
    camera_id: str
    zone_id: str
    started_at: datetime    # реальные часы (когда увидели)
    media_time_s: float     # время на записи / в сцене
    frame_id: int
    confidence: float


def point_in_polygon(px: float, py: float, polygon) -> bool:
    """Классический ray-casting: точка внутри многоугольника?"""
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            x_int = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < x_int:
                inside = not inside
        j = i
    return inside


def detection_in_zone(detection: dict, frame_w: int, frame_h: int, zone) -> bool:
    """Проверка попадания детекции в зону.

    Берём НИЗ рамки по центру (точка «на полу») — для напольной
    запрещённой зоны это правильнее, чем центр всего тела.
    bbox приходит в ПИКСЕЛЯХ, зона — в долях кадра: нормируем здесь.
    """
    x1, y1, x2, y2 = detection["bbox"]
    px = ((x1 + x2) / 2) / frame_w
    py = y2 / frame_h
    return point_in_polygon(px, py, zone)


class EventTracker:
    """Следит за потоком кадров и решает, когда создать событие.

    Один трекер = одна камера + одна зона + один тип нарушения.
    Для нескольких зон создайте несколько трекеров (будущее расширение).
    """

    CLEAR = "clear"
    CONFIRMING = "confirming"
    ACTIVE = "active"

    def __init__(self, rule: ViolationRule | None = None,
                 camera_id: str = "camera-1", zone_id: str = "zone-1"):
        self.rule = rule or DEFAULT_RULE
        self.camera_id = camera_id
        self.zone_id = zone_id
        self.state = self.CLEAR
        self._first_seen = None   # когда человек появился в зоне (медиа-время)
        self._last_seen = None    # когда видели в зоне в последний раз

    def process_frame(self, frame_id: int, detections: list,
                      media_time: timedelta, frame_w: int, frame_h: int,
                      now: datetime | None = None) -> list[Event]:
        """Обработать один кадр. Возвращает список НОВЫХ событий (0 или 1)."""
        now = now or datetime.now()

        # 1. есть ли в кадре объект нужного типа внутри зоны?
        candidate = None
        for d in detections:
            if d["label"] in self.rule.labels and \
               detection_in_zone(d, frame_w, frame_h, self.rule.zone):
                candidate = d
                break

        if candidate is not None:
            self._last_seen = media_time
            if self.state == self.CLEAR:
                self.state = self.CONFIRMING
                self._first_seen = media_time
            elif self.state == self.CONFIRMING:
                # 2. подтверждение во времени: мало одного кадра
                if media_time - self._first_seen >= timedelta(
                        seconds=self.rule.confirm_seconds):
                    self.state = self.ACTIVE
                    return [self._make_event(now, frame_id, candidate)]
            # 3. ACTIVE: ничего не делаем — это и есть ПОДАВЛЕНИЕ ПОВТОРОВ:
            #    длящееся нарушение не порождает новых событий.
            return []

        # объекта в зоне нет
        if self.state == self.CLEAR:
            return []
        if self.state == self.CONFIRMING:
            # подтверждение требует НЕПРЕРЫВНОГО присутствия:
            # пропал до подтверждения — начинаем заново
            self.state = self.CLEAR
            return []
        # ACTIVE: пережидаем короткие пропуски детекции; если зона пуста
        # достаточно долго — эпизод закрываем (зона «остыла»)
        if media_time - self._last_seen >= timedelta(seconds=self.rule.clear_seconds):
            self.state = self.CLEAR
        return []

    def _make_event(self, now: datetime, frame_id: int, detection: dict) -> Event:
        return Event(
            violation_type=self.rule.violation_type,
            title=self.rule.title,
            camera_id=self.camera_id,
            zone_id=self.zone_id,
            started_at=now,
            media_time_s=self._last_seen.total_seconds(),
            frame_id=frame_id,
            confidence=float(detection.get("confidence", 0.0)),
        )
