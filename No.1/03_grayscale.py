from __future__ import annotations

from common import grayscale, load_image, save


def main() -> None:
    image = load_image()
    gray = grayscale(image)
    save(gray, "03_grayscale.png")

    print("grayscale conversion saved")

if __name__ == "__main__":
    main()
