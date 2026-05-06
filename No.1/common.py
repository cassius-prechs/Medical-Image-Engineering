from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT_IMAGE = ROOT / "input.jpg"
OUTPUT_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class ImageInfo:
    file_name: str
    format_name: str
    width: int
    height: int
    channels: int
    bit_depth: int
    file_size_bytes: int


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_image(path: Path = INPUT_IMAGE) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def get_image_info(image: np.ndarray, path: Path = INPUT_IMAGE) -> ImageInfo:
    return ImageInfo(
        file_name=path.name,
        format_name=path.suffix.lstrip(".").upper() or "UNKNOWN",
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        channels=int(image.shape[2]) if image.ndim == 3 else 1,
        bit_depth=int(image.dtype.itemsize * 8),
        file_size_bytes=int(path.stat().st_size),
    )


def save(image: np.ndarray, filename: str) -> Path:
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    if image.ndim == 2:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), image)
    return path


def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def split_rgb(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blue, green, red = cv2.split(image)
    return red, green, blue


def add_salt_and_pepper_noise(image: np.ndarray, density: float = 0.05, salt_ratio: float = 0.5, seed: int = 42) -> np.ndarray:
    noisy = image.copy()
    height, width = noisy.shape[:2]
    total_pixels = width * height
    noisy_pixels = int(total_pixels * density)
    salt_pixels = int(noisy_pixels * salt_ratio)

    rng = random.Random(seed)
    selected = rng.sample(range(total_pixels), noisy_pixels)

    for index in selected[:salt_pixels]:
        x = index % width
        y = index // width
        noisy[y, x] = 255

    for index in selected[salt_pixels:]:
        x = index % width
        y = index // width
        noisy[y, x] = 0

    return noisy


def average_filter(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def weighted_average_filter(image: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]],
        dtype=np.float32,
    ) / 16.0
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    return cv2.medianBlur(image, size)



def info_lines(info: ImageInfo) -> list[str]:
    return [
        f"- file: {info.file_name}",
        f"- format: {info.format_name}",
        f"- dimensions: {info.width} x {info.height}",
        f"- channels: {info.channels}",
        f"- bit depth: {info.bit_depth} bit/channel",
        f"- file size: {info.file_size_bytes} bytes ({info.file_size_bytes / 1_000_000:.2f} MB)",
    ]
