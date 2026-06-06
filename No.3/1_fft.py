import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# load image (Pillow経由でGIFを安全に読み込む)
# =========================
try:
    img = np.array(Image.open("カメレオン1-1.gif").convert("L"))
except FileNotFoundError:
    print("エラー：ファイルが見つかりません。パスやファイル名を確認してください。")
    exit()

if img is None or img.size == 0:
    print("エラー：画像の読み込みに失敗しました。")
    exit()

print(f"画像読み込み成功！ サイズ: {img.shape}")

# =========================
# pre-filter (reduce salt-and-pepper)
# =========================
denoise = cv2.medianBlur(img, 5)
denoise = cv2.GaussianBlur(denoise, (5, 5), 0)

# =========================
# FFT
# =========================
f = np.fft.fft2(denoise)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = rows//2, cols//2

# =========================
# low pass mask
# =========================
mask = np.zeros((rows, cols), np.uint8)
r = 55

mask[
    crow-r:crow+r,
    ccol-r:ccol+r
] = 1

fshift_filtered = fshift * mask

# inverse FFT
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

import os
os.makedirs('./outputs', exist_ok=True)

prefix = '1_fft'
plt.imsave(f'./outputs/{prefix}_spectrum.png', np.log(np.abs(fshift)+1), cmap='gray')
plt.imsave(f'./outputs/{prefix}_low_pass.png', img_back, cmap='gray')
plt.imsave(f'./outputs/{prefix}_binary.png', binary, cmap='gray')
plt.imsave(f'./outputs/{prefix}_result.png', result, cmap='gray')

plt.figure(figsize=(12,4))

plt.subplot(141)
plt.imshow(np.log(np.abs(fshift)+1), cmap='gray')
plt.title("Spectrum")

plt.subplot(142)
plt.imshow(img_back, cmap='gray')
plt.title("Low Pass")

plt.subplot(143)
plt.imshow(binary, cmap='gray')
plt.title("Binary")

plt.subplot(144)
plt.imshow(result, cmap='gray')
plt.title("Result")

plt.show()