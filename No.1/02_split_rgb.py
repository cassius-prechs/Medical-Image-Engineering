from __future__ import annotations

from common import load_image, save, split_rgb


def main() -> None:
    image = load_image()
    red, green, blue = split_rgb(image)
    save(red, "02_red_channel.png")
    save(green, "02_green_channel.png")
    save(blue, "02_blue_channel.png")

    print("RGB channels saved")

if __name__ == "__main__":
    main()
