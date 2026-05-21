from common import (
    grayscale,
    load_image,
    save,
    gaussian_blur,
    sharpening,
)
import numpy as np


def main() -> None:
    # 画像の読み込みと前処理
    image = load_image()
    gray = grayscale(image)
    
    # ぼけた画像を作成（ガウシアンフィルタを適用）
    blurred = gaussian_blur(gray, kernel_size=5, sigma=2.0)
    
    # 鮮鋭化処理
    sharpened = sharpening(blurred)
    
    # 処理結果を正規化
    sharpened = np.uint8(np.clip(sharpened, 0, 255))
    
    # 画像を保存
    save(gray, "02_original.png")
    save(blurred, "02_blurred.png")
    save(sharpened, "02_sharpened.png")
    
    print("課題2: 鮮鋭化オペレータ を出力しました")


if __name__ == "__main__":
    main()
