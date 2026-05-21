from common import (
    grayscale,
    load_image,
    save,
    sobel_x,
    sobel_y,
    sobel_magnitude,
    prewitt_x,
    prewitt_y,
    prewitt_magnitude,
    laplacian,
)
import numpy as np


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    微分画像を0-255の範囲に正規化
    """
    image = image.astype(np.float32)
    image = np.abs(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255
    return np.uint8(np.clip(image, 0, 255))


def main() -> None:
    # 画像の読み込みと前処理
    image = load_image()
    gray = grayscale(image)
    
    # Sobel オペレータ
    sobel_gx = sobel_x(gray)
    sobel_gy = sobel_y(gray)
    sobel_mag = sobel_magnitude(gray)
    
    sobel_gx_display = normalize_for_display(sobel_gx)
    sobel_gy_display = normalize_for_display(sobel_gy)
    sobel_mag_display = normalize_for_display(sobel_mag)
    
    save(gray, "01_01_original.png")
    save(sobel_gx_display, "01_01_sobel_x.png")
    save(sobel_gy_display, "01_01_sobel_y.png")
    save(sobel_mag_display, "01_01_sobel_magnitude.png")
    
    # Prewitt オペレータ
    prewitt_gx = prewitt_x(gray)
    prewitt_gy = prewitt_y(gray)
    prewitt_mag = prewitt_magnitude(gray)
    
    prewitt_gx_display = normalize_for_display(prewitt_gx)
    prewitt_gy_display = normalize_for_display(prewitt_gy)
    prewitt_mag_display = normalize_for_display(prewitt_mag)
    
    save(prewitt_gx_display, "01_02_prewitt_x.png")
    save(prewitt_gy_display, "01_02_prewitt_y.png")
    save(prewitt_mag_display, "01_02_prewitt_magnitude.png")
    
    # Laplacian オペレータ
    laplacian_result = laplacian(gray)
    laplacian_display = normalize_for_display(laplacian_result)
    
    save(laplacian_display, "01_03_laplacian.png")
    
    print("課題1: 微分画像の作成 を出力しました")


if __name__ == "__main__":
    main()
