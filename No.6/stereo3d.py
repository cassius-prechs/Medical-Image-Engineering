import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
import os

os.makedirs("./outputs", exist_ok=True)

# パラメータ
IMG_W, IMG_H = 400, 300
CX, CY = IMG_W / 2, IMG_H / 2       # 画像中心 (200, 150)
BASELINE_2B = 58.0                  # 基線長 2b [mm]
B = BASELINE_2B / 2.0               # b [mm]
F = 220.0                           # 焦点距離 [pixel]
MM_PER_PIXEL = 1.8e-2               # 1 pixel = 1.8e-2 mm

# 頂点のピクセル座標（画像左上が原点）
labels = [1, 2, 3, 4]
left_px = {
    1: (272, 224),
    2: (256, 286),
    3: (397, 151),
    4: (65, 53),
}
right_px = {
    1: (173, 217),
    2: (169, 282),
    3: (319, 147),
    4: (3, 47),
}

# (1) 画像中心 (200,150) を原点とした座標へ変換（x：右正, y：上正）
def to_centered(px):
    x, y = px
    return (x - CX, CY - y)

left_c = {k: to_centered(v) for k, v in left_px.items()}
right_c = {k: to_centered(v) for k, v in right_px.items()}

print("=== (1) 画像中心を原点とした座標 [pixel] ===")
print(f"{'頂点':<4}{'(xl, yl)':<16}{'(xr, yr)':<16}")
for k in labels:
    print(f"{k:<4}{str(left_c[k]):<16}{str(right_c[k]):<16}")

# ─────────────────────────────────────────────────────────────────────────
# (2) 基線中心を原点とした 3 次元座標 [mm]
#     x = b(xl+xr)/(xl-xr),  y = b(yl+yr)/(xl-xr),  z = 2bf/(xl-xr)
# ─────────────────────────────────────────────────────────────────────────
coords3d = {}
disparities = {}
for k in labels:
    xl, yl = left_c[k]
    xr, yr = right_c[k]
    d = xl - xr
    x = B * (xl + xr) / d
    y = B * (yl + yr) / d
    z = 2 * B * F / d
    coords3d[k] = np.array([x, y, z])
    disparities[k] = d

print("\n=== (2) 基線中心を原点とした 3 次元座標 [mm] ===")
print(f"{'頂点':<4}{'視差 d':<10}{'x [mm]':<12}{'y [mm]':<12}{'z [mm]':<12}")
for k in labels:
    x, y, z = coords3d[k]
    print(f"{k:<4}{disparities[k]:<10.1f}{x:<12.4f}{y:<12.4f}{z:<12.4f}")

print("\n=== 参考: 垂直視差 (yl - yr)（理想的な平行ステレオでは 0）===")
for k in labels:
    yl = left_c[k][1]
    yr = right_c[k][1]
    print(f"頂点{k}: yl - yr = {yl - yr:.1f} pixel")

# (3) 頂点1を基準とした3方向の辺ベクトル・長さ・なす角
origin = coords3d[1]
vecs = {k: coords3d[k] - origin for k in (2, 3, 4)}
lengths = {k: np.linalg.norm(v) for k, v in vecs.items()}

print("\n=== (3) 頂点1基準の辺ベクトルと長さ ===")
for k in (2, 3, 4):
    v = vecs[k]
    print(f"V1{k} = ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})  |V1{k}| = {lengths[k]:.4f} mm")

print(f"\n箱のサイズ ≈ {lengths[2]:.1f} mm : {lengths[3]:.1f} mm : {lengths[4]:.1f} mm")

pairs = [(2, 3), (2, 4), (3, 4)]
angles = {}
print("\n=== (3) ベクトルのなす角度 ===")
for i, j in pairs:
    cos_t = np.dot(vecs[i], vecs[j]) / (lengths[i] * lengths[j])
    theta = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
    angles[(i, j)] = theta
    print(f"∠(V1{i}, V1{j}) : cosθ = {cos_t:.4f}, θ = {theta:.2f} deg")

# 図1: 左右画像上に頂点をプロット
img_l = np.array(Image.open("./left.png"))
img_r = np.array(Image.open("./right.png"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, img, pts, title in (
    (axes[0], img_l, left_px, "left.png"),
    (axes[1], img_r, right_px, "right.png"),
):
    ax.imshow(img)
    ax.set_title(title)
    for k in labels:
        x, y = pts[k]
        ax.plot(x, y, "o", color="cyan", markersize=6, markeredgecolor="black")
        # ラベルが画像端で見切れないよう、端に近い頂点は逆方向にオフセットする
        dx = -8 if x > IMG_W - 40 else 8
        dy = 14 if y < 30 else -4
        ha = "right" if dx < 0 else "left"
        ax.annotate(str(k), (x, y), textcoords="offset points", xytext=(dx, dy),
                    color="blue", fontsize=12, fontweight="bold", ha=ha)
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")

plt.tight_layout()
plt.savefig("./outputs/fig_vertices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: ./outputs/fig_vertices.png")

# 図2: 再構成した3次元頂点と辺を描画
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

pts = np.array([coords3d[k] for k in labels])
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="red", s=40)
for k in labels:
    x, y, z = coords3d[k]
    ax.text(x, y, z, f"  {k}", fontsize=11)

for k in (2, 3, 4):
    p1, p2 = coords3d[1], coords3d[k]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "b-", linewidth=2)

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")
ax.set_title("Reconstructed 3D vertices (edges from vertex 1)")

plt.tight_layout()
plt.savefig("./outputs/fig_3d_reconstruction.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ./outputs/fig_3d_reconstruction.png")

print("\n完了しました。")
