import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os

os.makedirs("./outputs", exist_ok=True)

# ── 画像読み込み（カラー→グレースケール変換）──────────────────────────
img1 = np.array(Image.open("./poc1.bmp").convert("L"), dtype=float)
img2 = np.array(Image.open("./poc2.bmp").convert("L"), dtype=float)

H, W = img1.shape  # 256 x 256

# ─────────────────────────────────────────────────────────────────────────────
# 1.1  FFT
# ─────────────────────────────────────────────────────────────────────────────
F = np.fft.fft2(img1)
G = np.fft.fft2(img2)

# 表示用にゼロ周波数を中央へ移動
F_shift = np.fft.fftshift(F)
G_shift = np.fft.fftshift(G)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle("1.1  FFT – Frequency Domain", fontsize=14)

axes[0, 0].imshow(img1, cmap="gray")
axes[0, 0].set_title("poc1.bmp (original)")
axes[0, 0].axis("off")

axes[0, 1].imshow(img2, cmap="gray")
axes[0, 1].set_title("poc2.bmp (shifted)")
axes[0, 1].axis("off")

axes[1, 0].imshow(np.log1p(np.abs(F_shift)), cmap="hot")
axes[1, 0].set_title("|F(u,v)|  (log scale)")
axes[1, 0].set_xlabel("u")
axes[1, 0].set_ylabel("v")

axes[1, 1].imshow(np.log1p(np.abs(G_shift)), cmap="hot")
axes[1, 1].set_title("|G(u,v)|  (log scale)")
axes[1, 1].set_xlabel("u")
axes[1, 1].set_ylabel("v")

plt.tight_layout()
plt.savefig("./outputs/fig_1_1_fft.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ./outputs/fig_1_1_fft.png")

# ─────────────────────────────────────────────────────────────────────────────
# 1.2  位相成分のみ抽出  F'(u,v) = F(u,v) / |F(u,v)|
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD = 1e-4

abs_F = np.abs(F)
abs_G = np.abs(G)

# 閾値以下の振幅成分は 0 にする
mask_F = abs_F > THRESHOLD
mask_G = abs_G > THRESHOLD

F_phase = np.where(mask_F, F / abs_F, 0.0 + 0.0j)
G_phase = np.where(mask_G, G / abs_G, 0.0 + 0.0j)

F_phase_shift = np.fft.fftshift(F_phase)
G_phase_shift = np.fft.fftshift(G_phase)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle("1.2  Phase-Only Components  |F'| = |G'| = 1", fontsize=14)

axes[0, 0].imshow(np.angle(F_phase_shift), cmap="hsv", vmin=-np.pi, vmax=np.pi)
axes[0, 0].set_title("Phase of F'(u,v)")
axes[0, 0].set_xlabel("u")
axes[0, 0].set_ylabel("v")

axes[0, 1].imshow(np.angle(G_phase_shift), cmap="hsv", vmin=-np.pi, vmax=np.pi)
axes[0, 1].set_title("Phase of G'(u,v)")
axes[0, 1].set_xlabel("u")
axes[0, 1].set_ylabel("v")

axes[1, 0].imshow(np.abs(F_phase_shift), cmap="gray", vmin=0, vmax=1)
axes[1, 0].set_title("|F'(u,v)|  (should be 0 or 1)")
axes[1, 0].set_xlabel("u")
axes[1, 0].set_ylabel("v")

axes[1, 1].imshow(np.abs(G_phase_shift), cmap="gray", vmin=0, vmax=1)
axes[1, 1].set_title("|G'(u,v)|  (should be 0 or 1)")
axes[1, 1].set_xlabel("u")
axes[1, 1].set_ylabel("v")

plt.tight_layout()
plt.savefig("./outputs/fig_1_2_phase.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ./outputs/fig_1_2_phase.png")

# ─────────────────────────────────────────────────────────────────────────────
# 1.3  積 R(u,v) = F'(u,v) * conj(G'(u,v))  →  IFFT  →  ピーク検出
# ─────────────────────────────────────────────────────────────────────────────
R = F_phase * np.conj(G_phase)
r = np.real(np.fft.ifft2(R))

# ─── ピーク位置 ───────────────────────────────────────────────────────────
peak_idx = np.argmax(r)
peak_y_raw, peak_x_raw = np.unravel_index(peak_idx, r.shape)

# fftshift して可視化用の相関面を作成（ゼロ中央）
r_shift = np.fft.fftshift(r)

# IFFT の結果はゼロ周波数が左上なので、折り返しを考慮して移動量を計算
# 正の移動量は [0, H/2), 負の移動量は [H/2, H) に折り返す
dx = -peak_x_raw if peak_x_raw < W // 2 else -peak_x_raw + W
dy = -peak_y_raw if peak_y_raw < H // 2 else -peak_y_raw + H

print(f"\n=== Phase-Only Correlation 結果 ===")
print(f"ピーク座標 (raw)  : x={peak_x_raw}, y={peak_y_raw}")
print(f"推定移動量        : Δx={dx} [pixel] (右方向正), Δy={dy} [pixel] (下方向正)")
print(f"※ poc2 は poc1 に対して 左右方向に {abs(dx)} px、上下方向に {abs(dy)} px移動")
print(f"   方向: {'右' if dx>0 else '左'}に{abs(dx)}px, {'下' if dy>0 else '上'}に{abs(dy)}px")

# 可視化（IFFT 結果）
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("1.3  IFFT of R(u,v) – Correlation Surface", fontsize=14)

im0 = axes[0].imshow(r, cmap="hot", origin="upper")
axes[0].set_title(f"r(x,y)  peak @ ({peak_x_raw}, {peak_y_raw})")
axes[0].plot(peak_x_raw, peak_y_raw, "c+", markersize=15, markeredgewidth=2)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0])

# fftshift 版（ゼロ中央）
cx, cy = W // 2, H // 2
peak_x_shift = peak_x_raw - W if peak_x_raw >= W // 2 else peak_x_raw
peak_y_shift = peak_y_raw - H if peak_y_raw >= H // 2 else peak_y_raw

im1 = axes[1].imshow(r_shift, cmap="hot", origin="upper",
                     extent=[-W//2, W//2, H//2, -H//2])
axes[1].set_title(f"r(x,y) fftshift  Δx={dx}, Δy={dy}")
axes[1].plot(peak_x_shift, peak_y_shift, "c+", markersize=15, markeredgewidth=2)
axes[1].set_xlabel("Δx [pixel]")
axes[1].set_ylabel("Δy [pixel]")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("./outputs/fig_1_3_poc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ./outputs/fig_1_3_poc.png")

print("\n全図の保存が完了しました。")
