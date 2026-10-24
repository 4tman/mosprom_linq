"""
linq_client.py — клиент сервиса инференса LinQ (RabbitMQ / AMQP).

ТОЧНО соответствует API примера со стенда (example.ipynb):
  • задача: PyTask(tensors), а путь к .tpu / очередь ответа / task_id
    передаются В ЗАГОЛОВКАХ сообщения (TARGET_URI_ATTR_NAME и др.);
  • очередь ответов notebook_<id>: auto_delete + bind к exchange
    по routing_key = имени очереди;
  • креды guest/guest, как в примере.

Запуск с локалки: VPN → RabbitMQ 10.0.114.30 напрямую,
Jupyter не участвует (организаторы разрешили).

Перед использованием под вашу модель поменяйте в LinQConfig:
  • login            — ваш логин (уже вписан hackathon55190)
  • program          — имя .tpu-программы в /auto/cache/<логин>/
  • input_tensor_name / output_tensor_name — имена тензоров модели
  • input_shape / dtype — форма и тип входного тензора

Файл .tpu кладётся на стенде (один раз в Jupyter, как в примере):
    ! cp <программа> /auto/cache/<логин>/
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass


@dataclass
class LinQConfig:
    # --- подключение (со стенда, не меняем) ---
    host: str = "10.0.114.30"
    port: int = 5672
    exchange: str = "tpu.direct"
    input_queue: str = "input"
    username: str = "guest"
    password: str = "guest"

    # --- ваша программа на стенде ---
    login: str = "hackathon55190"
    program: str = "torch_resnet50_b1_o5_h1m.tpu"  # ← ваша .tpu

    # --- параметры модели (у вашей будут свои) ---
    input_tensor_name: str = "input.1"    # ← имя входа ВАШЕЙ модели
    output_tensor_name: str = "495"       # ← имя выхода ВАШЕЙ модели
    input_shape: tuple = (1, 3, 224, 224)
    dtype: str = "float32"

    # --- тайминги ---
    request_timeout_s: float = 60.0

    @property
    def program_path(self) -> str:
        return f"/auto/cache/{self.login}/{self.program}"


@dataclass
class LinQResponse:
    data: dict            # выходные тензоры: {имя: np.ndarray}
    metadata: dict        # метаданные ответа (task_id, status_code, …)
    worker: object        # информация о воркере (pid и т.п.)
    latency_ms: float      # задержка «отправка → ответ»
    status_ok: bool        # status_code == 0


class LinQClient:
    """Один клиент = одно соединение с RabbitMQ. В нашем прототипе задачи
    шлёт один цикл обработки кадров, поэтому потокобезопасность не нужна."""

    def __init__(self, config: LinQConfig | None = None):
        self.config = config or LinQConfig()
        self._conn = None
        self._chan = None

    # ------------------------------------------------------------------
    def _connect(self):
        import pika  # лениво: панель работает и без установленного pika
        if self._conn is not None and self._conn.is_open:
            return
        self._conn = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.config.host,
                port=self.config.port,
                virtual_host="/",
                credentials=pika.PlainCredentials(
                    self.config.username, self.config.password),
                socket_timeout=5,
                stack_timeout=15,
                connection_attempts=1,
                heartbeat=60,
            )
        )
        self._chan = self._conn.channel()

    def close(self):
        if self._conn is not None and self._conn.is_open:
            self._conn.close()
        self._conn = None
        self._chan = None

    # ------------------------------------------------------------------
    def infer(self, tensors: dict, timeout_s: float | None = None) -> LinQResponse:
        """Отправить входные тензоры, дождаться ответа.

        tensors — словарь {input_tensor_name: np.ndarray},
        например {"input.1": массив (1, 3, 224, 224) float32}.

        Формат сообщения — как в example.ipynb со стенда:
        тело = serialize(PyTask(tensors)), а путь к .tpu, очередь ответа
        и task_id едут в заголовках AMQP.
        """
        import pika
        from tpu_pycloud_format import (MESSAGE_ID_ATTR_NAME,
                                        RESPONSE_ROUTING_KEYS_ATTR_NAME,
                                        TARGET_URI_ATTR_NAME, PyTask,
                                        deserialize, serialize)

        self._connect()
        cfg = self.config
        timeout_s = timeout_s or cfg.request_timeout_s

        task_id = uuid.uuid4().hex
        reply_queue = f"notebook_{task_id[:8]}"

        # очередь ответов: как в примере — auto_delete + bind к exchange
        self._chan.queue_declare(queue=reply_queue, auto_delete=True)
        self._chan.queue_bind(exchange=cfg.exchange, queue=reply_queue,
                              routing_key=reply_queue)

        # тело задачи — только тензоры
        body = serialize(PyTask(tensors))

        t0 = time.monotonic()
        self._chan.basic_publish(
            exchange=cfg.exchange,
            routing_key=cfg.input_queue,
            body=body,
            properties=pika.BasicProperties(headers={
                TARGET_URI_ATTR_NAME: f"file://{cfg.program_path}?raw=False",
                RESPONSE_ROUTING_KEYS_ATTR_NAME: json.dumps([reply_queue]),
                MESSAGE_ID_ATTR_NAME: task_id,
            }),
        )

        # ждём СВОЙ ответ (по task_id из метаданных)
        for method, _props, raw in self._chan.consume(
                reply_queue, inactivity_timeout=timeout_s):
            if raw is None:
                self._chan.cancel()
                raise TimeoutError(
                    f"Ответ воркера не пришёл за {timeout_s} с "
                    f"(task_id={task_id})")
            self._chan.basic_ack(method.delivery_tag)
            data, metadata, worker = deserialize(raw)

            # task_id ищем в заголовках сообщения RabbitMQ.
            headers = _props.headers or {}
            resp_task_id = headers.get(MESSAGE_ID_ATTR_NAME)

            if resp_task_id is None:
                resp_task_id = _props.correlation_id

            if isinstance(resp_task_id, bytes):
                resp_task_id = resp_task_id.decode("utf-8")

            # Если идентификатор передан, проверяем соответствие.
            if resp_task_id is not None and str(resp_task_id) != task_id:
                continue

            self._chan.cancel()
            latency_ms = (time.monotonic() - t0) * 1000

            return LinQResponse(
                data=data,
                metadata=metadata,
                worker=worker,
                latency_ms=round(latency_ms, 1),
                status_ok=(worker.get("status_code") == 0),
            )

        raise TimeoutError(f"Ответ воркера не пришёл (task_id={task_id})")
