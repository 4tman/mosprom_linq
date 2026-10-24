"""
processor.py — источники кадров и «детекторы».

Этот файл отвечает на два вопроса:
  1) ОТКУДА берутся кадры (SyntheticFrameSource / VideoFrameSource);
  2) ЧТО мы о кадре узнали (DemoBackend / LinQBackend).

Панель оператора (app.py) и журнал (storage.py) не знают, как устроен
детектор, — они работают только с результатом.

КОНТРАКТ КООРДИНАТ (договоритесь с напарником один раз и не меняйте):

    detection = {
        "label": "person",        # метка класса
        "confidence": 0.93,       # уверенность 0..1
        "bbox": [x1, y1, x2, y2], # ПИКСЕЛИ исходного кадра
        "track_id": 1,            # опционально, для трекинга
    }

Любой реальный бэкенд (LinQ) должен вернуть список таких словарей —
тогда остальная система не изменится.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Результат обработки одного кадра
# ---------------------------------------------------------------------------
@dataclass
class ProcessResult:
    detections: list        # список detection-словарей (контракт выше)
    latency_ms: float       # время обработки кадра
    backend: str            # "demo" | "linq"


# ---------------------------------------------------------------------------
# Источники кадров
# ---------------------------------------------------------------------------
class SyntheticFrameSource:
    """Синтетическая «камера»: рисуем цех и человека, идущего через зону.

    Кадры полностью детерминированы (зависят только от номера кадра),
    поэтому демо воспроизводимо: при каждом запуске — один и тот же сценарий.
    Используется, чтобы собрать и проверить интерфейс ДО подключения модели.
    """

    MEDIA_FPS = 10.0  # «время сцены»: 10 кадров = 1 секунда

    def __init__(self, width: int = 960, height: int = 540, length: int = 240):
        self.width = width
        self.height = height
        self.length = length

    def __iter__(self):
        for i in range(self.length):
            yield i, self._make_frame(i)

    def _make_frame(self, frame_id: int) -> np.ndarray:
        w, h = self.width, self.height
        # фон-цех: стены, пол, «станки»
        frame = np.full((h, w, 3), 45, dtype=np.uint8)
        cv2.rectangle(frame, (0, int(h * 0.45)), (w, h), (58, 58, 58), -1)  # пол
        cv2.rectangle(frame, (int(w * 0.04), int(h * 0.15)),
                     (int(w * 0.22), int(h * 0.45)), (78, 78, 88), -1)       # станок 1
        cv2.rectangle(frame, (int(w * 0.80), int(h * 0.10)),
                     (int(w * 0.96), int(h * 0.45)), (72, 82, 72), -1)       # станок 2
        # шум «камеры» (детерминированный)
        rng = np.random.default_rng(frame_id)
        frame = cv2.add(frame, rng.integers(0, 8, frame.shape, dtype=np.uint8))
        # человек
        x1, y1, x2, y2 = demo_person_bbox(frame_id, w, h)
        _draw_person(frame, x1, y1, x2, y2)
        return frame


class VideoFrameSource:
    """Чтение видеофайла покадрово; по окончании — зацикливание."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.media_fps = fps if fps and fps > 1 else 25.0
        self._frame_id = 0

    def __iter__(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:  # конец файла — начинаем заново
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    return
            yield self._frame_id, frame
            self._frame_id += 1

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------
# Демо-движок: положение человека (общее для источника и «детектора»)
# ---------------------------------------------------------------------------
def demo_person_bbox(frame_id: int, width: int, height: int):
    """Человек идёт слева направо и каждые 120 кадров проходит опасную зону."""
    u = (frame_id % 120) / 120.0
    cx = (0.10 + 0.80 * u) * width
    return [int(cx - 0.04 * width), int(0.35 * height),
            int(cx + 0.04 * width), int(0.85 * height)]


def _draw_person(frame, x1, y1, x2, y2):
    """Простой силуэт: голова + туловище (только для демо-кадров)."""
    cx = (x1 + x2) // 2
    head_r = max(4, (x2 - x1) // 3)
    cv2.rectangle(frame, (x1, y1 + head_r), (x2, y2), (180, 120, 60), -1)
    cv2.circle(frame, (cx, y1 + head_r - 2), head_r, (150, 150, 160), -1)


# ---------------------------------------------------------------------------
# Бэкенды «детектора»
# ---------------------------------------------------------------------------
class DemoBackend:
    """Имитация детектора для отладки интерфейса.

    ВАЖНО: детекции СКРИПТОВЫЕ — они вычисляются из номера кадра и НЕ зависят
    от содержимого кадра. Если подать сюда реальное видео, оно НЕ анализируется.
    Это позволяет собрать интерфейс, журнал и логику событий до готовности
    реальной модели на LinQ.
    """

    def process(self, frame: np.ndarray, frame_id: int) -> ProcessResult:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = demo_person_bbox(frame_id, w, h)
        detections = [{
            "label": "person",
            "confidence": round(0.90 + 0.05 * ((frame_id % 7) / 7), 3),
            "bbox": [x1, y1, x2, y2],
            "track_id": 1,
        }]
        latency_ms = (time.perf_counter() - t0) * 1000
        return ProcessResult(detections=detections,
                             latency_ms=round(latency_ms, 2),
                             backend="demo")


class LinQBackend:
    """Реальный инференс на LinQ (транспорт — linq_client.py).

    Что заполнить под вашу модель — всё в трёх местах:
      1. LinQConfig в linq_client.py: логин, имя .tpu, имена тензоров;
      2. preprocess() ниже: кадр → входной тензор модели;
      3. postprocess() ниже: выходной тензор → список detection-словарей
         по контракту из шапки этого файла.

    Проверка связи со стендом — stand_check.py (запускать в Jupyter).
    """

    def __init__(self, preprocess=None, postprocess=None, config=None):
        from linq_client import LinQClient, LinQConfig
        self.client = LinQClient(config or LinQConfig())
        self._preprocess = preprocess or self._default_preprocess
        self._postprocess = postprocess or self._default_postprocess

    def process(self, frame: np.ndarray, frame_id: int) -> ProcessResult:
        t0 = time.perf_counter()
        tensors = self._preprocess(frame)
        response = self.client.infer(tensors)
        detections = self._postprocess(response)
        return ProcessResult(
            detections=detections,
            latency_ms=response.latency_ms,
            backend="linq",
        )

    def _default_preprocess(self, frame: np.ndarray) -> dict:
        raise NotImplementedError(
            "Заполните preprocess для ВАШЕЙ модели: кадр → тензор. "
            "Образец — функция preprocess в example.ipynb со стенда "
            "(изменение размера, нормализация, порядок осей). "
            "Тензор обёртывается в словарь: "
            "{LinQConfig.input_tensor_name: массив}")

    def _default_postprocess(self, response) -> list:
        raise NotImplementedError(
            "Заполните postprocess для ВАШЕЙ модели: выходной тензор → "
            "список detection-словарей по контракту из шапки processor.py "
            "(label, confidence, bbox в пикселях). У YOLO выход — "
            "координаты боксов + классы + уверенности, см. документацию "
            "модели.")


# ---------------------------------------------------------------------------
# Отрисовка аннотаций
# ---------------------------------------------------------------------------
def draw_annotations(frame: np.ndarray, detections: list, rule) -> np.ndarray:
    """Рисует на КОПИИ кадра полигон зоны правила и рамки детекций."""
    out = frame.copy()
    h, w = out.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in rule.zone],
                   dtype=np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    # подпись зоны не рисуем: OpenCV не выводит кириллицу,
    # а легенда по-русски есть в интерфейсе рядом с кадром
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f'{d["label"]} {d["confidence"]:.2f}',
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out
