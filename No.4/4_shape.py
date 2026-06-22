"""
Task 4 Problem 4: Shape Feature - Circularity
  Circularity = 4*pi*A / L^2
  Perfect circle -> Circularity = 1.0
Input:  ./images/
Output: printed to stdout
"""

import cv2
import numpy as np

INPUT_DIR = "images"


def binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary == 255) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def largest_contour(binary: np.ndarray):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def get_area_perimeter(filename: str):
    img_bgr = cv2.imread(f"{INPUT_DIR}/{filename}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if len(np.unique(gray)) <= 5:
        binary = (gray > 127).astype(np.uint8) * 255
        if np.sum(binary == 255) > binary.size * 0.5:
            binary = cv2.bitwise_not(binary)
    else:
        binary = binarize(gray)
    cnt = largest_contour(binary)
    area = int(cv2.contourArea(cnt))
    perimeter = cv2.arcLength(cnt, closed=True)
    return area, perimeter


def circularity(area: float, perimeter: float) -> float:
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter**2)


print("=" * 50)
print("Shape Feature: Circularity = 4*pi*A / L^2  (circle = 1.0)")
print("=" * 50)

area_c, peri_c = get_area_perimeter("circle.bmp")
circ_c = circularity(area_c, peri_c)
print(f"\nCircle (circle.bmp)")
print(f"  Area        A = {area_c} pixels")
print(f"  Perimeter   L = {peri_c:.2f} pixels")
print(f"  Circularity   = {circ_c:.4f}")

area_ch, peri_ch = get_area_perimeter("chameleon3.bmp")
circ_ch = circularity(area_ch, peri_ch)
print(f"\nChameleon (chameleon3.bmp)")
print(f"  Area        A = {area_ch} pixels")
print(f"  Perimeter   L = {peri_ch:.2f} pixels")
print(f"  Circularity   = {circ_ch:.4f}")