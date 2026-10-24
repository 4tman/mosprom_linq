"""
stand_check.py — проверка инференса на LinQ: запуск ЛОКАЛЬНО с ноутбука.

Подключение идёт ПРЯМО К RABBITMQ (10.0.114.30) через VPN — Jupyter
в этом файле не участвует. Организаторы подтвердили, что брокер
доступен с локальных машин.

Что проверяет:
  1. Соединение с RabbitMQ и один инференс программой из примера;
  2. Замер задержки на серии запросов: min / среднее / max;
  3. Распределение по воркерам (по pid в ответах).

ЗАПУСК С ЛОКАЛКИ:
  1. Включи VPN (маршрут до 10.0.114.30 идёт через tun0).
  2. pip install pika numpy
  3. Положи tpu_pycloud_format.py (скачанный со стенда) рядом с этим файлом.
  4. В linq_client.py впиши свой логин в LinQConfig.
  5. python stand_check.py

  .tpu-файл на ноутбуке НЕ нужен: путь /auto/cache/<логин>/ читает
  воркер на СВОЕЙ стороне (общий диск стенда). Если .tpu ещё не там —
  один раз выполни первую ячейку example.ipynb в Jupyter.

Этот скрипт — «проверка трубопровода», НЕ финальный инференс:
случайный тензор вместо картинки годится ровно для проверки связи.
"""
from __future__ import annotations

import time

import numpy as np

from linq_client import LinQClient, LinQConfig

# ----------------------------------------------------------------------
# 1. Настройка (логин уже вписан — проверь, что он ваш)
# ----------------------------------------------------------------------
LOGIN = "hackathon55190"

config = LinQConfig(
    login='guest',  # ваш логин в LinQ
    program="torch_resnet50_b1_o5_h1m.tpu",  # штатная программа примера
    input_tensor_name="input.1",
    output_tensor_name="495",
    input_shape=(1, 3, 224, 224),
)
client = LinQClient(config)
print(f"программа: {config.program_path}")

# ----------------------------------------------------------------------
# 2. Один запрос: связь есть?
# ----------------------------------------------------------------------
# случайный тензор формы входа модели — только для проверки связи
tensor = np.random.rand(*config.input_shape).astype(config.dtype)
tensors = {config.input_tensor_name: tensor}

try:
    response = client.infer(tensors)
except Exception as e:
    raise SystemExit(f"ЗАПРОС НЕ ПРОШЁЛ: {type(e).__name__}: {e}")

out = response.data.get(config.output_tensor_name)
print(f"status_ok: {response.status_ok}")
print(f"задержка: {response.latency_ms} мс")
print(f"воркер: {response.worker}")
print(f"форма выхода: {None if out is None else out.shape}")

assert response.status_ok, f"воркер вернул ошибку: {response.metadata}"
assert out is not None, f"выход {config.output_tensor_name} не найден в ответе"
print("СВЯЗЬ ОК: один инференс прошёл успешно")

# ----------------------------------------------------------------------
# 3. Серия из 10 запросов: замер задержки
# ----------------------------------------------------------------------
latencies = []
workers = []
for i in range(10):
    response = client.infer(tensors)
    assert response.status_ok, f"ошибка на запросе {i}: {response.metadata}"
    latencies.append(response.latency_ms)
    workers.append(str(response.worker))

print("\n--- задержки, мс (отправка → ответ) ---")
for i, lat in enumerate(latencies):
    print(f"  запрос {i + 1:>2}: {lat:8.1f}")
print(f"\nmin:    {min(latencies):8.1f} мс")
print(f"среднее: {sum(latencies) / len(latencies):8.1f} мс")
print(f"max:    {max(latencies):8.1f} мс")
print(f"воркеры в ответах: {sorted(set(workers))}")

# ----------------------------------------------------------------------
# 4. Итог для чекпоинта (скопируйте в отчёт)
# ----------------------------------------------------------------------
print("\n--- ИТОГ (для отчёта/чекпоинта) ---")
print(f"инференс на LinQ: работает")
print(f"задержка запроса (10 замеров): "
      f"min {min(latencies):.0f} / ср {sum(latencies)/len(latencies):.0f} / "
      f"max {max(latencies):.0f} мс")
print(f"время на кадр при непрерывной обработке — примерно как средняя "
      f"задержка; для потока 10 кадров/с этого {'достаточно' if sum(latencies)/len(latencies) < 100 else 'может быть мало'}")

client.close()
