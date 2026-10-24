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

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Результат обработки одного кадра
# ---------------------------------------------------------------------------
@dataclass
class ProcessResult:
    detections: list        # список detection-словарей (контракт выше)
    latency_ms: float       # время обработки кадра
    backend: str            # "demo" | "linq"
    worker: dict = field(default_factory=dict)  # служебная информация стенда


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
                      (int(w * 0.22), int(h * 0.45)), (78, 78, 88), -1)      # станок 1
        cv2.rectangle(frame, (int(w * 0.80), int(h * 0.10)),
                      (int(w * 0.96), int(h * 0.45)), (72, 82, 72), -1)      # станок 2
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


# ---------------------------------------------------------------------------
# LinQ: подготовка входа и разбор выхода YOLOv8 (1, 4 + классы, 8400)
# ---------------------------------------------------------------------------
INPUT_SIZE = 640
PAD_COLOR = (114, 114, 114)
CLASS_NAMES_FILE = ROOT / "class_names.json"


def default_linq_config():
    """Те же параметры, с которыми прошёл check_safetyvision.py."""
    from linq_client import LinQConfig
    return LinQConfig(
        login="hackathon55190",
        program="safetyvision_linq.tpu",
        input_tensor_name="images",
        output_tensor_name="output0",
        input_shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
        dtype="float32",
        request_timeout_s=120,
    )


def letterbox(frame_bgr: np.ndarray, size: int = INPUT_SIZE):
    """Кадр BGR → тензор (1, 3, size, size) float32 0..1.

    Повторяет подготовку из скрипта сборки: RGB, PIL BILINEAR,
    вписывание с сохранением пропорций, серые поля 114 по центру.
    Возвращает тензор и параметры для обратного пересчёта рамок.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    width, height = image.size
    scale = min(size / width, size / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    canvas.paste(resized, (pad_x, pad_y))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None].copy()

    meta = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "frame_w": width,
        "frame_h": height,
    }
    return tensor, meta


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """Подавление перекрывающихся рамок (boxes: N×4 в формате x1,y1,x2,y2)."""
    order = np.argsort(scores)[::-1]
    areas = ((boxes[:, 2] - boxes[:, 0]).clip(0)
             * (boxes[:, 3] - boxes[:, 1]).clip(0))
    keep = []

    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if not rest.size:
            break
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        wh = (rb - lt).clip(0)
        inter = wh[:, 0] * wh[:, 1]
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]

    return keep


def load_class_names(num_classes: int) -> list:
    """Имена классов из class_names.json (список строк), иначе class_N."""
    if CLASS_NAMES_FILE.exists():
        try:
            names = json.loads(CLASS_NAMES_FILE.read_text(encoding="utf-8"))
            if isinstance(names, list) and len(names) == num_classes:
                return [str(n) for n in names]
        except (OSError, json.JSONDecodeError):
            pass
    return [f"class_{i}" for i in range(num_classes)]


def decode_yolov8(output: np.ndarray, meta: dict,
                  confidence_threshold: float = 0.25,
                  iou_threshold: float = 0.45) -> list:
    """Выход (1, 4 + C, N) → список detection-словарей в пикселях кадра."""
    output = np.asarray(output, dtype=np.float32)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] <= 4:
        raise RuntimeError(f"Неожиданная форма выхода модели: {output.shape}")

    pred = output[0].T                 # (N, 4 + C)
    boxes_cxcywh = pred[:, :4]
    scores = pred[:, 4:]

    # Если оценки похожи на логиты (вне 0..1) — переводим в вероятности.
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))

    class_ids = scores.argmax(axis=1)
    confidence = scores.max(axis=1)

    mask = confidence >= confidence_threshold
    if not mask.any():
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    confidence = confidence[mask]

    cx, cy, w, h = boxes_cxcywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # Из входа 640×640 обратно в пиксели исходного кадра.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta["pad_x"]) / meta["scale"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta["pad_y"]) / meta["scale"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, meta["frame_w"] - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, meta["frame_h"] - 1)

    names = load_class_names(scores.shape[1])
    detections = []

    # NMS отдельно для каждого класса.
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        for k in nms(boxes[idx], confidence[idx], iou_threshold):
            i = idx[k]
            x1, y1, x2, y2 = boxes[i]
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue  # рамка целиком попала в серые поля
            detections.append({
                "label": names[int(cls)],
                "class_id": int(cls),
                "confidence": float(confidence[i]),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


class LinQBackend:
    """Реальный инференс на LinQ (транспорт — linq_client.py).

    preprocess:  кадр BGR → ({имя_входа: тензор}, meta)
    postprocess: (response, meta) → список detection-словарей по контракту.
    Оба можно подменить через аргументы конструктора.
    """

    def __init__(self, preprocess=None, postprocess=None, config=None,
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        from linq_client import LinQClient
        self.config = config or default_linq_config()
        self.client = LinQClient(self.config)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._preprocess = preprocess or self._default_preprocess
        self._postprocess = postprocess or self._default_postprocess

    def process(self, frame: np.ndarray, frame_id: int) -> ProcessResult:
        tensors, meta = self._preprocess(frame)
        response = self.client.infer(tensors)

        if not response.status_ok:
            raise RuntimeError(f"Ошибка выполнения на стенде: {response.worker}")

        detections = self._postprocess(response, meta)
        return ProcessResult(
            detections=detections,
            latency_ms=float(response.latency_ms),
            backend="linq",
            worker=dict(response.worker or {}),
        )

    def close(self):
        if self.client is not None:
            self.client.close()

    def _default_preprocess(self, frame: np.ndarray):
        tensor, meta = letterbox(frame, INPUT_SIZE)
        return {self.config.input_tensor_name: tensor}, meta

    def _default_postprocess(self, response, meta: dict) -> list:
        name = self.config.output_tensor_name
        if name not in response.data:
            raise RuntimeError(
                f"В ответе нет {name}, ключи: {list(response.data)}")
        return decode_yolov8(
            response.data[name], meta,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )


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