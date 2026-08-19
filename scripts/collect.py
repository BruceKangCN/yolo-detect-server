"""Utility script to collect images in different directories."""

import os
import shutil
from pathlib import Path

SRC_ROOT = Path(__file__).parent / "snapshot"
DST_DIR = Path(__file__).parent / "images"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    index = 0
    for dir in SRC_ROOT.iterdir():
        if not dir.is_dir():
            continue
        for file in dir.iterdir():
            if not file.is_file():
                continue
            if file.suffix not in IMAGE_EXTENSIONS:
                continue
            if not file.name.startswith("Color_"):
                continue
            shutil.copy(file, DST_DIR / f"{index:08d}{file.suffix}")
            index += 1


if __name__ == "__main__":
    main()
