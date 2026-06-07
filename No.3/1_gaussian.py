import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


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
# strong blur
# =========================
blur = cv2.GaussianBlur(
    img,
    (21,21),
    0
)

# =========================
# threshold
# =========================
_, binary = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# If background becomes white, invert so the chameleon is white
white_ratio = np.mean(binary == 255)
if white_ratio > 0.6:
    binary = cv2.bitwise_not(binary)

# =========================
# morphology
# =========================
kernel = np.ones((7, 7), np.uint8)

binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# largest component
num_labels, labels, stats, centroids = \
    cv2.connectedComponentsWithStats(binary)

result = np.zeros_like(binary)

if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result[labels == largest] = 255
else:
    result = binary.copy()

# smooth jagged edges
result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
result = cv2.medianBlur(result, 5)

# =========================
# save output images to ./outputs directory
# =========================

os.makedirs('./outputs', exist_ok=True)

prefix = '1_gaussian'
plt.imsave(f'./outputs/{prefix}_blur.png', blur, cmap='gray')
plt.imsave(f'./outputs/{prefix}_binary.png', binary, cmap='gray')
plt.imsave(f'./outputs/{prefix}_result.png', result, cmap='gray')

plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(blur, cmap='gray')
plt.title("Blur")

plt.subplot(132)
plt.imshow(binary, cmap='gray')
plt.title("Binary")

plt.subplot(133)
plt.imshow(result, cmap='gray')
plt.title("Result")

plt.show()