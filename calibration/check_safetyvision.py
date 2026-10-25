from pathlib import Path
import numpy as np

import numpy as np
from PIL import Image, ImageOps

from linq_client import LinQClient, LinQConfig


ROOT = Path(__file__).resolve().parent

# Берём первую фотографию из папки калибровки.
images = sorted(
    path for path in (ROOT / "calibration").glob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    raise FileNotFoundError("Нет фотографий в папке calibration")

image_path = images[0]

# Точно такая же подготовка, как в скрипте сборки.
with Image.open(image_path) as image:
    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size
    scale = min(640 / width, 640 / height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR,
    )

    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    canvas.paste(
        resized,
        ((640 - new_width) // 2, (640 - new_height) // 2),
    )

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None].copy()

config = LinQConfig(
    login="hackathon55190",
    program="safetyvision_linq.tpu",
    input_tensor_name="images",
    output_tensor_name="output0",
    input_shape=(1, 3, 640, 640),
    dtype="float32",
    request_timeout_s=120,
)

client = LinQClient(config)

try:
    print("Изображение:", image_path, flush=True)
    print("Программа:", config.program_path, flush=True)
    print("Вход:", tensor.shape, tensor.dtype, flush=True)
    print("Отправка на LinQ...", flush=True)

    response = client.infer({"images": tensor})
    
    npz_path = "safetyvision_linq_output.npz"

    with np.load(npz_path, allow_pickle=False) as archive:
        outputs = {name: archive[name] for name in archive.files}

    for name, array in outputs.items():
        print(f"Выход {name}: shape={array.shape}, dtype={array.dtype}")

    # Массив модели по имени.
    output0 = outputs["output0"]

    print("Воркер:", response.worker, flush=True)
    print("Задержка запроса:", response.latency_ms, "мс", flush=True)

    if not response.status_ok:
        raise RuntimeError(
            f"Ошибка выполнения на стенде: {response.worker}"
        )

    print("Ключи выхода:", list(response.data), flush=True)

    if "output0" not in response.data:
        raise RuntimeError("В ответе отсутствует output0")

    output = np.asarray(response.data["output0"])

    print("Выход:", output.shape, output.dtype, flush=True)
    print("Все значения конечные:", bool(np.isfinite(output).all()))

    if output.size:
        print("Диапазон:", float(output.min()), float(output.max()))

    np.savez_compressed(
        ROOT / "safetyvision_linq_output.npz",
        output0=output,
    )

    print("ОТВЕТ LinQ ПОЛУЧЕН И СОХРАНЁН", flush=True)

finally:
    client.close()
