"""
Task 4 Problem 2: Texture Feature Quantities from Histogram
Input:  ./images/
Output: printed to stdout (LaTeX table also printed)

Using normalized histogram p(l) = h(l) / N:
  MEN = sum( l * p(l) )                         mean
  CNT = sum( l^2 * p(l) )                       contrast
  VAR = sum( (l-MEN)^2 * p(l) )                 variance
  SKW = (1/sigma^3) * sum( (l-MEN)^3 * p(l) )  skewness  (sigma = sqrt(VAR))
  KRT = (1/sigma^4) * sum( (l-MEN)^4 * p(l) )  kurtosis
  EGY = sum( p(l)^2 )                           energy
  EPY = -sum( p(l) * log(p(l)) )               entropy
"""

import cv2
import numpy as np

INPUT_DIR = "images"
IMAGES = ["bamboo", "ground", "panch", "tile", "wall"]

L = np.arange(256, dtype=np.float64)

print(f"{'Image':<10} {'MEN':>8} {'CNT':>10} {'VAR':>10} {'SKW':>8} {'KRT':>8} {'EGY':>10} {'EPY':>8}")
print("-" * 80)

results = {}
for name in IMAGES:
    img_bgr = cv2.imread(f"{INPUT_DIR}/{name}.bmp")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    N = gray.size
    p = hist.astype(np.float64) / N

    MEN = np.sum(L * p)
    CNT = np.sum(L**2 * p)
    VAR = np.sum((L - MEN) ** 2 * p)
    sigma = np.sqrt(VAR)
    SKW = np.sum((L - MEN) ** 3 * p) / (sigma**3) if sigma > 0 else 0.0
    KRT = np.sum((L - MEN) ** 4 * p) / (sigma**4) if sigma > 0 else 0.0

    # Exclude bins where p(l)=0 to avoid log(0)
    mask = p > 0
    EGY = np.sum(p[mask] ** 2)
    EPY = -np.sum(p[mask] * np.log(p[mask]))

    results[name] = dict(MEN=MEN, CNT=CNT, VAR=VAR, SKW=SKW, KRT=KRT, EGY=EGY, EPY=EPY)
    print(
        f"{name:<10}  {MEN:8.2f}  {CNT:10.2f}  {VAR:10.2f}  {SKW:8.4f}  {KRT:8.4f}  {EGY:10.6f}  {EPY:8.4f}"
    )

# LaTeX table output
print("\n--- LaTeX table ---")
print(r"\begin{tabular}{lrrrrrrr}")
print(r"\toprule")
print(r"Image & MEN & CNT & VAR & SKW & KRT & EGY & EPY \\")
print(r"\midrule")
for name in IMAGES:
    r = results[name]
    print(
        f"{name} & {r['MEN']:.2f} & {r['CNT']:.2f} & {r['VAR']:.2f} & "
        f"{r['SKW']:.4f} & {r['KRT']:.4f} & {r['EGY']:.6f} & {r['EPY']:.4f} \\\\"
    )
print(r"\bottomrule")
print(r"\end{tabular}")