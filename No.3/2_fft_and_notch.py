import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# =========================
# load image (Pillow経由でGIFを安全に読み込む)
# =========================

def process_image(image_path):
    try:
        img = np.array(Image.open(image_path).convert("L"))
    except FileNotFoundError:
        print(f"エラー：ファイルが見つかりません。パスやファイル名を確認してください。 {image_path}")
        return
    if img is None or img.size == 0:
        print(f"エラー：画像の読み込みに失敗しました。 {image_path}")
        return

    print(f"画像読み込み成功！ サイズ: {img.shape}")

    # =========================
    # Pre-filter (reduce salt-and-pepper + mild blur)
    # =========================

    denoise = cv2.medianBlur(img, 5)
    denoise = cv2.GaussianBlur(denoise, (5, 5), 0)

    # =========================
    # FFT
    # =========================

    f = np.fft.fft2(denoise)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # =========================
    # notch filter (auto peak suppression)
    # =========================

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)

    # Suppress strongest periodic peaks outside the center region
    mag = np.abs(fshift)
    mag[crow-10:crow+10, ccol-10:ccol+10] = 0

    flat = mag.flatten()
    peak_count = 12
    peak_idx = np.argpartition(flat, -peak_count)[-peak_count:]
    peak_coords = np.column_stack(np.unravel_index(peak_idx, mag.shape))

    notch_r = 6
    for pr, pc in peak_coords:
        r0, r1 = max(pr - notch_r, 0), min(pr + notch_r + 1, rows)
        c0, c1 = max(pc - notch_r, 0), min(pc + notch_r + 1, cols)
        mask[r0:r1, c0:c1] = 0

    # mild low-pass to keep the large shape
    r = 60
    lp = np.zeros((rows, cols), np.uint8)
    lp[crow-r:crow+r, ccol-r:ccol+r] = 1
    mask = mask * lp

    fshift_filtered = fshift * mask

    # =========================
    # inverse FFT
    # =========================

    ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(ishift)
    img_back = np.abs(img_back)

    # =========================
    # threshold
    # =========================

    img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, binary = cv2.threshold(
        img_back_norm,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If background becomes white, invert so the chameleon is white
    white_ratio = np.mean(binary == 255)
    if white_ratio > 0.6:
        binary = cv2.bitwise_not(binary)

    # =========================
    # morphology + largest component
    # =========================

    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    result = np.zeros_like(binary)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        result[labels == largest] = 255
    else:
        result = binary.copy()

    # =========================
    # save output images to ./outputs directory
    # =========================

    os.makedirs('./outputs', exist_ok=True)

    base_name = os.path.basename(image_path)
    name_without_ext, ext = os.path.splitext(base_name)

    prefix = '2_fft_and_notch'
    plt.imsave(f'./outputs/{prefix}_original.png', img, cmap='gray')
    plt.imsave(f'./outputs/{prefix}_denoise.png', denoise, cmap='gray')
    plt.imsave(f'./outputs/{prefix}_spectrum.png', magnitude_spectrum, cmap='gray')
    plt.imsave(f'./outputs/{prefix}_binary.png', binary, cmap='gray')
    plt.imsave(f'./outputs/{prefix}_result.png', result, cmap='gray')

    # =========================
    # show
    # =========================

    plt.figure(figsize=(12,4))

    plt.subplot(141)
    plt.imshow(img, cmap='gray')
    plt.title("Original")

    plt.subplot(142)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Spectrum")

    plt.subplot(143)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary")

    plt.subplot(144)
    plt.imshow(result, cmap='gray')
    plt.title("Result")

    plt.show()

# Process Image No.2 only
process_image("カメレオン2-1.gif")