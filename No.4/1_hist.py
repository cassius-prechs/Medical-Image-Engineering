"""
Task 4 Problem 1: Brightness Histogram
Convert 5 texture images to grayscale and create brightness histograms.
Input:  ./images/
Output: ./outputs/
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_DIR = "images"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGES = ["bamboo", "ground", "panch", "tile", "wall"]

fig, axes = plt.subplots(len(IMAGES), 2, figsize=(10, 4 * len(IMAGES)))

for i, name in enumerate(IMAGES):
    img_bgr = cv2.imread(f"{INPUT_DIR}/{name}.bmp")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(f"{OUTPUT_DIR}/{name}_gray.png", gray)

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))

    axes[i, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[i, 0].set_title(f"{name} (grayscale)")
    axes[i, 0].axis("off")

    axes[i, 1].bar(range(256), hist, width=1, color="steelblue", edgecolor="none")
    axes[i, 1].set_title(f"{name} histogram")
    axes[i, 1].set_xlabel("Intensity")
    axes[i, 1].set_ylabel("Frequency")
    axes[i, 1].set_xlim(0, 255)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/task1_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/task1_histograms.png")

# Save individual histogram plots
for name in IMAGES:
    img_bgr = cv2.imread(f"{INPUT_DIR}/{name}.bmp")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))

    fig2, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(256), hist, width=1, color="steelblue", edgecolor="none")
    ax.set_title(f"{name} histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 255)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Individual histogram images saved.")
