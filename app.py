"""
app.py — панель оператора «Пост контроля безопасности» (Streamlit).

Запуск:
    python -m streamlit run app.py

Этот файл — ТОЛЬКО интерфейс: кнопки, кадр, метрики, журнал.
Вся логика в соседних модулях:
    processor.py — откуда кадры и что «нашёл детектор»;
    events.py    — когда нарушение считается подтверждённым;
    storage.py   — журнал и кадры-доказательства.

Поток данных:
    processor (кадр + детекции) → events (событие?) → storage (журнал)
                                    → app (показать оператору)
"""
from __future__ import annotations

import dataclasses
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import streamlit as st

from events import DEFAULT_RULE, EventTracker
from processor import (DemoBackend, LinQBackend, SyntheticFrameSource,
                       VideoFrameSource, draw_annotations)
from storage import EventStorage

from image_panel import render_image_panel


st.set_page_config(page_title="Пост контроля безопасности", layout="wide")

SOURCE_SYNTH = "Синтетическая сцена (демо)"
SOURCE_VIDEO = "Загрузить видео"
BACKEND_DEMO = "Демо-детектор"
BACKEND_LINQ = "LinQ (заготовка)"

STATE_RU = {"clear": "норма", "confirming": "проверка…", "active": "НАРУШЕНИЕ"}
STATUS_RU = {"new": "новое", "confirmed": "подтверждено",
             "false_positive": "ложное"}


# ---------------------------------------------------------------------------
# Состояние сессии (переживает перерисовки Streamlit)
# ---------------------------------------------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("running", False)
    ss.setdefault("frame_id", 0)
    ss.setdefault("source", None)        # объект-источник кадров
    ss.setdefault("source_iter", None)  # итератор по кадрам
    ss.setdefault("source_mode", None)
    ss.setdefault("backend", None)
    ss.setdefault("backend_name", "—")
    ss.setdefault("tracker", None)
    ss.setdefault("media_fps", 10.0)
    ss.setdefault("_last_tick", 0.0)
    # последнее состояние обработки — чтобы кадр не мигал между тиками
    ss.setdefault("last_frame_rgb", None)   # аннотированный кадр (RGB)
    ss.setdefault("last_stats", None)        # метрики последнего кадра
    ss.setdefault("last_event_msg", None)    # последнее уведомление
    if "storage" not in ss:
        ss.storage = EventStorage()


init_state()
ss = st.session_state

from image_panel import render_image_panel

page = st.sidebar.radio(
    "Раздел",
    ["Фотография — LinQ", "Видео и демо"],
)

if page == "Фотография — LinQ":
    ss.running = False

    if ss.source is not None:
        if hasattr(ss.source, "close"):
            ss.source.close()

        ss.source = None
        ss.source_iter = None

    render_image_panel()
    st.stop()


# ---------------------------------------------------------------------------
# Панель слева: параметры и управление
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Параметры")

    source_mode = st.radio("Источник кадров", [SOURCE_SYNTH, SOURCE_VIDEO])
    uploaded = None
    if source_mode == SOURCE_VIDEO:
        uploaded = st.file_uploader(
            "Видеофайл", type=["mp4", "avi", "mov", "mkv"])

    backend_mode = st.radio(
        "Обработчик", [BACKEND_DEMO, BACKEND_LINQ],
        help="LinQ подключим, когда будет .tpu-программа модели")

    confirm_s = st.slider(
        "Подтверждение нарушения, сек", 1.0, 6.0, 2.0, 0.5,
        help="Сколько секунд нарушение должно непрерывно длиться, "
             "чтобы создать событие")
    clear_s = st.slider(
        "Снятие нарушения, сек", 2.0, 10.0, 4.0, 0.5,
        help="Сколько секунд зона должна быть пустой, чтобы эпизод закрылся")
    play_fps = st.slider(
        "Скорость проигрывания, кадров/с", 2, 25, 10)

    col_start, col_stop = st.columns(2)
    start_pressed = col_start.button("▶ Начать", type="primary")
    stop_pressed = col_stop.button("■ Стоп")


def _close_source():
    if ss.source is not None and hasattr(ss.source, "close"):
        ss.source.close()
    ss.source = None
    ss.source_iter = None


# --- обработка кнопки «Начать» ---
if start_pressed:
    try:
        if source_mode == SOURCE_SYNTH:
            source = SyntheticFrameSource()
            ss.media_fps = source.MEDIA_FPS
        elif uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            source = VideoFrameSource(tmp_path)
            ss.media_fps = source.media_fps
        else:
            st.sidebar.error("Выберите файл видео")
            source = None

        if source is not None:
            _close_source()
            ss.source = source
            ss.source_iter = iter(source)
            ss.source_mode = source_mode
            rule = dataclasses.replace(
                DEFAULT_RULE, confirm_seconds=confirm_s, clear_seconds=clear_s)
            ss.tracker = EventTracker(rule=rule)
            if backend_mode == BACKEND_DEMO:
                ss.backend = DemoBackend()
                ss.backend_name = "demo"
            else:
                ss.backend = LinQBackend()
                ss.backend_name = "linq"
            ss.frame_id = 0
            ss.running = True
    except (RuntimeError, ValueError) as e:  # например, видео не открылось
        st.sidebar.error(str(e))
        ss.running = False

# --- обработка кнопки «Стоп» ---
if stop_pressed:
    ss.running = False
    _close_source()


# ---------------------------------------------------------------------------
# Заголовок и честная пометка режима
# ---------------------------------------------------------------------------
st.title("Пост контроля безопасности")

if ss.backend_name == "demo":
    st.warning(
        "ДЕМО-режим: детекции скриптовые, реальное видео НЕ анализируется. "
        "Так собираем интерфейс до подключения модели на LinQ.")
elif ss.backend_name == "linq":
    st.info(
        "LinQ-бэкенд ещё НЕ подключён: заполните LinQBackend.process в "
        "processor.py по образцу example.ipynb со стенда (см. README).")


# ---------------------------------------------------------------------------
# Обработка кадров: фрагмент, обновляющийся сам.
# ВАЖНО: рисуем на каждом проходе из session_state, а обрабатываем новый
# кадр не чаще, чем выбрал оператор — иначе кадр «мигал» бы между тиками.
# ---------------------------------------------------------------------------
def _process_one_frame() -> None:
    """Взять следующий кадр, прогнать через детектор и логику событий."""
    try:
        frame_id, frame = next(ss.source_iter)
    except StopIteration:
        ss.running = False
        return

    ss.frame_id = frame_id
    h, w = frame.shape[:2]
    media_time = timedelta(seconds=frame_id / ss.media_fps)

    # 1. «детектор»
    try:
        result = ss.backend.process(frame, frame_id)
    except NotImplementedError as e:
        ss.running = False
        ss.last_event_msg = f"Ошибка: {e}"
        return

    # 2. логика нарушения
    new_events = ss.tracker.process_frame(
        frame_id=frame_id,
        detections=result.detections,
        media_time=media_time,
        frame_w=w,
        frame_h=h,
        now=datetime.now(),
    )

    # 3. аннотированный кадр (он же — доказательство)
    annotated = draw_annotations(frame, result.detections, ss.tracker.rule)
    for ev in new_events:
        try:
            ss.storage.save_event(ev, backend=result.backend,
                                  evidence_frame=annotated)
        except RuntimeError as e:
            ss.last_event_msg = f"Ошибка сохранения события: {e}"
        else:
            ss.last_event_msg = (f"Подтверждено нарушение: {ev.title} "
                                 f"(кадр {frame_id})")

    # 4. запоминаем состояние для показа
    lat_label = ("Время демо-обработки, мс" if ss.backend_name == "demo"
                 else "Задержка инференса, мс")
    ss.last_frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    ss.last_stats = {
        "frame": frame_id,
        "latency": result.latency_ms,
        "latency_label": lat_label,
        "state": STATE_RU[ss.tracker.state],
        "total": ss.storage.stats()["total"],
    }


@st.fragment(run_every=0.05)
def processing_fragment():
    # темп: обрабатываем новый кадр не чаще, чем выбрал оператор
    if ss.running and ss.source_iter is not None:
        now = time.perf_counter()
        if now - ss._last_tick >= 1.0 / play_fps:
            ss._last_tick = now
            _process_one_frame()

    # рисуем ВСЕГДА — из сохранённого состояния (кадр не мигает)
    if ss.last_frame_rgb is None:
        st.info("Нажмите «▶ Начать» в панели слева, чтобы запустить обработку.")
        return

    col_frame, col_stats = st.columns([3, 1])
    with col_frame:
        st.image(ss.last_frame_rgb, caption=f"Кадр {ss.last_stats['frame']}")
        st.caption("Красный полигон — опасная зона; "
                   "жёлтая рамка — детекция человека.")
    with col_stats:
        st.metric("Кадр", ss.last_stats["frame"])
        st.metric(ss.last_stats["latency_label"],
                  f"{ss.last_stats['latency']:.1f}")
        st.metric("Событий в журнале", ss.last_stats["total"])
        st.metric("Состояние зоны", ss.last_stats["state"])

    if ss.last_event_msg:
        st.error(ss.last_event_msg)


processing_fragment()


# ---------------------------------------------------------------------------
# Журнал событий (с фильтрами — требование кейса).
# Фильтры — вне фрагмента (меняются редко), список — во фрагменте,
# чтобы журнал обновлялся сам во время обработки.
# ---------------------------------------------------------------------------
st.header("Журнал событий")

f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
with f_col1:
    f_type = st.selectbox(
        "Тип нарушения", ["Все"] + ss.storage.distinct("violation_type"))
with f_col2:
    f_cam = st.selectbox("Камера", ["Все"] + ss.storage.distinct("camera_id"))
with f_col3:
    f_zone = st.selectbox("Зона", ["Все"] + ss.storage.distinct("zone_id"))
with f_col4:
    f_from = st.date_input("С даты", value=None)
with f_col5:
    f_to = st.date_input("По дату", value=None)


@st.fragment(run_every=1.0)
def journal_fragment(f_type, f_cam, f_zone, f_from, f_to):
    rows = ss.storage.list_events(
        violation_type=None if f_type == "Все" else f_type,
        camera_id=None if f_cam == "Все" else f_cam,
        zone_id=None if f_zone == "Все" else f_zone,
        date_from=f_from,
        date_to=f_to,
    )

    if not rows:
        st.info("Пока нет событий — журнал пуст.")
        return

    st.dataframe(
        [{"№": r["id"],
          "Время": r["started_at"][:19].replace("T", " "),
          "Камера": r["camera_id"],
          "Зона": r["zone_id"],
          "Тип": r["title"],
          "Статус": STATUS_RU.get(r["status"], r["status"])} for r in rows],
        hide_index=True,
    )

    st.subheader("Карточки событий")
    for r in rows[:10]:  # последние 10, чтобы не тормозить
        with st.expander(
                f"Событие №{r['id']} — {r['started_at'][:19].replace('T', ' ')}"
                f" — {r['title']}"):
            st.write(f"Камера: {r['camera_id']}, зона: {r['zone_id']}, "
                     f"кадр: {r['frame_id']}")
            st.write(f"Уверенность детекции: {r['confidence']:.2f}, "
                     f"источник: {r['backend']}, "
                     f"статус: {STATUS_RU.get(r['status'], r['status'])}")
            b_ok, b_false = st.columns(2)
            if b_ok.button("Подтвердить", key=f"ok_{r['id']}"):
                ss.storage.update_status(r["id"], "confirmed")
                st.rerun()
            if b_false.button("Ложное срабатывание", key=f"fp_{r['id']}"):
                ss.storage.update_status(r["id"], "false_positive")
                st.rerun()
            if r["evidence_path"]:
                img = cv2.imread(r["evidence_path"])
                if img is not None:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                             caption="Кадр-доказательство")
                else:
                    st.warning("Не удалось открыть кадр-доказательство")


journal_fragment(f_type, f_cam, f_zone, f_from, f_to)
