from datetime import datetime
from pathlib import Path
import hashlib
import json
import uuid

import cv2
import numpy as np
import streamlit as st

from processor import LinQBackend


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "data" / "image_results"


def draw_boxes(frame, detections):
    out = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = [
            int(value) for value in detection["bbox"]
        ]

        cv2.rectangle(
            out, (x1, y1), (x2, y2), (0, 255, 255), 2
        )

        text = (
            f"{detection['label']} "
            f"{detection['confidence']:.2f}"
        )

        cv2.putText(
            out,
            text,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    return out


def render_image_panel():
    st.header("Фотография → LinQ → результат")

    st.caption(
        "Только реальный LinQBackend. "
        "Демо-детектор здесь не используется. "
        "На одной фотографии не подтверждаем нарушение во времени."
    )

    uploaded = st.file_uploader(
        "Выберите фотографию",
        type=["jpg", "jpeg", "png"],
        key="linq_image_upload",
    )

    if uploaded is None:
        st.session_state.pop("linq_image_result", None)
        st.info("Загрузите фотографию для обработки.")
        return

    raw = uploaded.getvalue()
    image_id = hashlib.sha256(raw).hexdigest()

    frame = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        st.error("Не удалось прочитать изображение.")
        return

    previous = st.session_state.get("linq_image_result")

    if previous is not None and previous["image_id"] != image_id:
        st.session_state.pop("linq_image_result", None)

    st.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        caption=f"Исходное изображение: {uploaded.name}",
        width=650,
    )

    if st.button(
        "Обработать на LinQ",
        type="primary",
        key="run_linq_image",
    ):
        st.session_state.pop("linq_image_result", None)
        backend = None

        try:
            backend = LinQBackend()

            with st.spinner("Ожидание ответа LinQ..."):
                result = backend.process(frame, frame_id=0)

            if result.backend != "linq":
                raise RuntimeError(
                    "Обработчик вернул результат не от LinQ."
                )

            annotated = draw_boxes(frame, result.detections)

            st.session_state["linq_image_result"] = {
                "image_id": image_id,
                "source_name": uploaded.name,
                "original": frame.copy(),
                "annotated": annotated,
                "detections": result.detections,
                "latency_ms": result.latency_ms,
                "backend": result.backend,
                "created_at": datetime.now().astimezone().isoformat(),
                "saved_path": None,
            }

        except NotImplementedError as exc:
            st.warning(
                "Панель готова, но обработчик модели "
                "в LinQBackend ещё не реализован."
            )
            st.code(str(exc))

        except Exception as exc:
            st.error("Обработка на LinQ не завершилась.")
            st.exception(exc)

        finally:
            if backend is not None:
                try:
                    backend.client.close()
                except Exception as exc:
                    st.warning(
                        f"Не удалось корректно закрыть соединение: {exc}"
                    )

    record = st.session_state.get("linq_image_result")

    if record is None:
        return

    st.subheader("Результат")

    st.image(
        cv2.cvtColor(
            record["annotated"], cv2.COLOR_BGR2RGB
        ),
        caption="Рамки по результату LinQ",
    )

    left, right = st.columns(2)

    left.metric(
        "Задержка запроса",
        f"{record['latency_ms']:.1f} мс",
    )
    right.metric(
        "Обнаружено объектов",
        len(record["detections"]),
    )

    st.caption(
        "Задержка запроса — не чистое время вычислений ускорителя."
    )

    if record["detections"]:
        rows = [
            {
                "Класс": item["label"],
                "Уверенность": float(item["confidence"]),
                "Рамка": [
                    float(value) for value in item["bbox"]
                ],
            }
            for item in record["detections"]
        ]

        st.dataframe(rows, hide_index=True)
    else:
        st.info(
            "Объекты не обнаружены. "
            "Это не означает, что нарушений гарантированно нет."
        )

    if st.button(
        "Сохранить результат",
        key="save_linq_image",
    ):
        try:
            if record["saved_path"] is None:
                run_id = (
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                    + "_"
                    + uuid.uuid4().hex[:8]
                )

                folder = RESULTS_DIR / run_id
                folder.mkdir(parents=True)

                for name, image in (
                    ("original.jpg", record["original"]),
                    ("annotated.jpg", record["annotated"]),
                ):
                    if not cv2.imwrite(str(folder / name), image):
                        raise RuntimeError(
                            f"Не удалось сохранить {name}"
                        )

                report = {
                    "created_at": record["created_at"],
                    "source_name": record["source_name"],
                    "backend": record["backend"],
                    "request_latency_ms": float(
                        record["latency_ms"]
                    ),
                    "detections": [
                        {
                            "label": item["label"],
                            "confidence": float(
                                item["confidence"]
                            ),
                            "bbox": [
                                float(value)
                                for value in item["bbox"]
                            ],
                        }
                        for item in record["detections"]
                    ],
                }

                (folder / "result.json").write_text(
                    json.dumps(
                        report,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                record["saved_path"] = str(folder)

            st.success(
                f"Сохранено: {record['saved_path']}"
            )

        except Exception as exc:
            st.error("Не удалось сохранить результат.")
            st.exception(exc)