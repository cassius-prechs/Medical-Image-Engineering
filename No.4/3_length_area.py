"""
Task 4 Problem 3: Perimeter and Area of Objects
Compute perimeter and area for circle.bmp and chameleon3.bmp.
Input:  ./images/
Output: ./outputs/  (binary, contour, and overlay images)
"""

import cv2
import numpy as np
import os

INPUT_DIR = "images"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu thresholding; ensure object is white (foreground)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary == 255) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def largest_contour(binary: np.ndarray):
    """Return the contour with the largest area."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def process_image(filename: str, name: str):
    img_bgr = cv2.imread(f"{INPUT_DIR}/{filename}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # If already binary (few unique values), just threshold at 127
    if len(np.unique(gray)) <= 5:
        binary = (gray > 127).astype(np.uint8) * 255
        if np.sum(binary == 255) > binary.size * 0.5:
            binary = cv2.bitwise_not(binary)
    else:
        binary = binarize(gray)

    cv2.imwrite(f"{OUTPUT_DIR}/{name}_binary.png", binary)

    cnt = largest_contour(binary)
    area = int(cv2.contourArea(cnt))
    perimeter = cv2.arcLength(cnt, closed=True)

    # Contour-only image
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, [cnt], -1, 255, 1)
    cv2.imwrite(f"{OUTPUT_DIR}/{name}_contour.png", contour_img)

    # Overlay contour on grayscale
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 2)
    cv2.imwrite(f"{OUTPUT_DIR}/{name}_overlay.png", overlay)

    return area, perimeter


print("=" * 50)
print("Circle image (circle.bmp)")
area_circle, peri_circle = process_image("circle.bmp", "circle")
print(f"  Area      A = {area_circle} pixels")
print(f"  Perimeter L = {peri_circle:.2f} pixels")

print()
print("=" * 50)
print("Chameleon image (chameleon3.bmp)")
area_cham, peri_cham = process_image("chameleon3.bmp", "chameleon")
print(f"  Area      A = {area_cham} pixels")
print(f"  Perimeter L = {peri_cham:.2f} pixels")

print()
print("Saved to outputs/:")
print("  circle_binary.png, circle_contour.png, circle_overlay.png")
print("  chameleon_binary.png, chameleon_contour.png, chameleon_overlay.png")