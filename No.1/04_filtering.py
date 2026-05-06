from __future__ import annotations

from common import (
    add_salt_and_pepper_noise,
    average_filter,
    grayscale,
    load_image,
    median_filter,
    save,
    weighted_average_filter,
)


def main() -> None:
    image = load_image()
    gray = grayscale(image)
    noisy = add_salt_and_pepper_noise(gray, density=0.05, salt_ratio=0.5, seed=42)
    avg = average_filter(noisy)
    weighted = weighted_average_filter(noisy)
    median = median_filter(noisy, size=3)

    save(gray, "04_grayscale_for_filtering.png")
    save(noisy, "04_noisy.png")
    save(avg, "04_average_filter.png")
    save(weighted, "04_weighted_average_filter.png")
    save(median, "04_median_filter.png")

    print("filtered images saved")

if __name__ == "__main__":
    main()
