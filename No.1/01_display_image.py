from __future__ import annotations

from common import INPUT_IMAGE, get_image_info, info_lines, load_image, save


def main() -> None:
    image = load_image()
    info = get_image_info(image, INPUT_IMAGE)
    save(image, "01_original_copy.png")

    print("Original Image Information:")
    for line in info_lines(info):
        print(line)

if __name__ == "__main__":
    main()
