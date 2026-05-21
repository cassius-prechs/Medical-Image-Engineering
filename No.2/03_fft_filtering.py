from common import (
    grayscale,
    load_image,
    save,
    fft_2d,
    ideal_low_pass_filter,
)
import cv2
import numpy as np


def main() -> None:
    # 画像の読み込みと前処理
    image = load_image()
    
    # 画像リサイズ（256x256または512x512）
    # 処理効率のため256x256を使用
    image_resized = cv2.resize(image, (256, 256))
    gray = grayscale(image_resized)
    
    print(f"処理用画像サイズ: {gray.shape}")
    
    # 3.1 高速フーリエ変換 (FFT)
    print("FFT処理中...")
    magnitude_display, phase = fft_2d(gray)
    
    # 位相画像（追加課題）
    phase_display = np.uint8(255 * (phase + np.pi) / (2 * np.pi))
    save(phase_display, "03_01_fft_phase.png")

    save(gray, "03_01_original.png")
    save(magnitude_display, "03_01_fft_magnitude.png")
    
    # 3.2 周波数空間でのフィルタリング
    print("理想低域フィルタ処理中...")
    
    # 異なる半径での理想フィルタ処理
    radii = [5, 10, 50]
    filtered_images = {}
    
    for r in radii:
        filtered = ideal_low_pass_filter(gray, radius=r)
        filtered_images[r] = filtered
        save(filtered, f"03_02_lpf_r{r}.png")
        print(f"  R0={r} 処理完了")
    

    print("課題3: 2次元FFTによるフィルタリング を出力しました")


if __name__ == "__main__":
    main()
