import base64
import io
import math
import mimetypes
from pathlib import Path
from typing import Optional, Tuple


MAX_RATIO = 200
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
SPATIAL_MERGE_SIZE = 32


def load_pil_image():
    try:
        module = __import__("PIL.Image", fromlist=["Image"])
        return module
    except Exception:
        return None


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def prepare_image_bytes_for_model(
    image_path: Path,
    model: str,
    spatial_merge_size: int = SPATIAL_MERGE_SIZE,
) -> bytes:
    binary = image_path.read_bytes()
    if "qwen" not in model.lower():
        return binary

    Image = load_pil_image()
    if Image is None:
        raise RuntimeError(
            "Model contains 'Qwen' but Pillow is not installed. Install pillow to resize images."
        )

    with Image.open(io.BytesIO(binary)) as image:
        image = image.convert("RGB")
        width, height = image.size
        target_height, target_width = smart_resize(
            height,
            width,
            factor=spatial_merge_size,
        )

        if (target_width, target_height) != (width, height):
            # print(
            #     "Resizing image from (%d, %d) to (%d, %d)"
            #     % (width, height, target_width, target_height)
            # )
            image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)

        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def image_to_base64_for_model(
    image_path: Path,
    model: str,
    spatial_merge_size: int = SPATIAL_MERGE_SIZE,
) -> str:
    binary = prepare_image_bytes_for_model(
        image_path=image_path,
        model=model,
        spatial_merge_size=spatial_merge_size,
    )
    return base64.b64encode(binary).decode("ascii")


def image_to_data_uri_for_model(
    image_path: Path,
    model: str,
    spatial_merge_size: int = SPATIAL_MERGE_SIZE,
) -> str:
    binary = prepare_image_bytes_for_model(
        image_path=image_path,
        model=model,
        spatial_merge_size=spatial_merge_size,
    )
    guessed_mime, _ = mimetypes.guess_type(str(image_path))
    mime = guessed_mime or "application/octet-stream"
    b64 = base64.b64encode(binary).decode("ascii")
    return f"data:{mime};base64,{b64}"