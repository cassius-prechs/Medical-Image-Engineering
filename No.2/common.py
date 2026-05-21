from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_image(path: Path = INPUT_IMAGE) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def get_image_info(image: np.ndarray, file_name: str) -> ImageInfo:
    return ImageInfo(
        file_name=file_name,
        format_name=INPUT_IMAGE.suffix.lstrip(".").upper() or "UNKNOWN",
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        channels=int(image.shape[2]) if image.ndim == 3 else 1,
    )


def save(image: np.ndarray, filename: str) -> Path:
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    img = image
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.integer):
            img = img.astype(np.float64)

        if np.issubdtype(img.dtype, np.floating):
            minv = float(np.min(img))
            maxv = float(np.max(img))
            if minv >= 0.0:
                if maxv <= 1.0:
                    img_u8 = np.uint8(np.clip(img * 255.0, 0, 255))
                else:
                    img_u8 = np.uint8(np.clip(img, 0, 255))
            else:
                max_abs = max(abs(minv), abs(maxv))
                if max_abs == 0:
                    img_u8 = np.full(img.shape, 128, dtype=np.uint8)
                else:
                    scale = 127.0 / max_abs
                    img_u8 = np.uint8(np.clip(img * scale + 128.0, 0, 255))
        else:
            img_u8 = np.uint8(np.clip(img, 0, 255))
    else:
        img_u8 = img

    img_out = np.ascontiguousarray(img_u8)
    cv2.imwrite(str(path), img_out)
    return path


def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def write_text(filename: str, text: str) -> Path:
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path


def info_lines(info: ImageInfo) -> list[str]:
    return [
        f"- ファイル名: {info.file_name}",
        f"- フォーマット: {info.format_name}",
        f"- 画素数: {info.width} x {info.height}",
        f"- チャネル数: {info.channels}",
    ]


# 微分フィルタ関連の関数
def sobel_x(image: np.ndarray) -> np.ndarray:
    """
    Sobelオペレータ (X方向)
    """
    image = image.astype(np.float32)
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def sobel_y(image: np.ndarray) -> np.ndarray:
    """
    Sobelオペレータ (Y方向)
    """
    image = image.astype(np.float32)
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Sobelオペレータ (勾配の大きさ)
    """
    image = image.astype(np.float32)
    gx = sobel_x(image).astype(np.float32)
    gy = sobel_y(image).astype(np.float32)
    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude


def prewitt_x(image: np.ndarray) -> np.ndarray:
    """
    Prewittオペレータ (X方向)
    """
    image = image.astype(np.float32)
    kernel = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def prewitt_y(image: np.ndarray) -> np.ndarray:
    """
    Prewittオペレータ (Y方向)
    """
    image = image.astype(np.float32)
    kernel = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def prewitt_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Prewittオペレータ (勾配の大きさ)
    """
    image = image.astype(np.float32)
    gx = prewitt_x(image).astype(np.float32)
    gy = prewitt_y(image).astype(np.float32)
    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude


def laplacian(image: np.ndarray) -> np.ndarray:
    """
    Laplacianオペレータ
    """
    image = image.astype(np.float32)
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


# 鮮鋭化関連の関数
def sharpening(image: np.ndarray) -> np.ndarray:
    """
    鮮鋭化オペレータ
    """
    image = image.astype(np.float32)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    ガウシアンフィルタ
    """
    image = image.astype(np.float32)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


# FFT関連の関数
def fft_2d(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    2次元FFT
    
    Returns:
        (magnitude, phase): 周波数空間の振幅と位相
    """
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # FFT計算
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # 振幅と位相を計算
    magnitude = np.abs(f_shift)
    phase = np.angle(f_shift)
    
    # ログスケーリングして表示用に正規化
    magnitude_log = np.log(magnitude + 1)
    magnitude_display = np.uint8(255 * magnitude_log / np.max(magnitude_log))
    
    return magnitude_display, phase


def ideal_low_pass_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """
    理想低域フィルタ
    """
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # FFT
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # フィルタマスク作成
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    distance = np.sqrt(u**2 + v**2)
    
    # 理想低域フィルタ
    mask = (distance <= radius).astype(np.float32)
    f_filtered = f_shift * mask
    
    # 逆FFT
    f_ishift = np.fft.ifftshift(f_filtered)
    result = np.fft.ifft2(f_ishift)
    result = np.abs(result)
    
    return np.uint8(np.clip(result, 0, 255))


def ideal_high_pass_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """
    理想高域フィルタ
    """
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # FFT
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # フィルタマスク作成
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    distance = np.sqrt(u**2 + v**2)
    
    # 理想高域フィルタ
    mask = (distance > radius).astype(np.float32)
    f_filtered = f_shift * mask
    
    # 逆FFT
    f_ishift = np.fft.ifftshift(f_filtered)
    result = np.fft.ifft2(f_ishift)
    result = np.abs(result)
    
    return np.uint8(np.clip(result, 0, 255))
