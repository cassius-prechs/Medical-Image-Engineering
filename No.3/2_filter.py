import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import generic_filter

# =========================
# load image (Pillow経由で安全に読み込む)
# =========================
try:
    img = np.array(Image.open("カメレオン2-1.gif").convert("L"))
except FileNotFoundError:
    print("エラー：ファイルが見つかりません。パスやファイル名を確認してください。")
    exit()

print(f"画像読み込み成功！ サイズ: {img.shape}")

# =========================
# 1. テクスチャ解析（ローカル標準偏差の計算）
# =========================
def local_std(X):
    return np.std(X)
std_img = generic_filter(img, local_std, size=5)

# 0-255に正規化
std_img_norm = cv2.normalize(std_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# =========================
# 2. 二値化（大津の二値化）
# =========================
_, binary = cv2.threshold(std_img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


binary = cv2.bitwise_not(binary)

# =========================
# 3. モルフォロジー変換と最大コンポーネント抽出
# =========================
# カメレオンの内部に残った細かい穴（黒いポツポツ）をしっかり埋めます
kernel_close = np.ones((11, 11), np.uint8)
binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

# 周囲に残った小さなノイズ（白いゴミ）を消します
kernel_open = np.ones((5, 5), np.uint8)
binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, kernel_open)

# 最大の白い塊（カメレオン本体）だけを残して完全に孤立したゴミを除去
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cleaned)

result = np.zeros_like(binary_cleaned)
if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    result[labels == largest] = 255
else:
    result = binary_cleaned.copy()

result = cv2.bitwise_not(result)

# =========================
# save output images to ./outputs directory
# =========================
os.makedirs('./outputs', exist_ok=True)
prefix = '2_filter'

plt.imsave(f'./outputs/{prefix}_std_texture.png', std_img_norm, cmap='gray')
plt.imsave(f'./outputs/{prefix}_binary.png', binary, cmap='gray')
plt.imsave(f'./outputs/{prefix}_result.png', result, cmap='gray')

# 結果の表示
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(std_img_norm, cmap='gray')
plt.title("Texture (Std Dev)")

plt.subplot(132)
plt.imshow(binary, cmap='gray')
plt.title("Inverted Binary")

plt.subplot(133)
plt.imshow(result, cmap='gray')
plt.title("Final Result (Chameleon is White)")

plt.show()