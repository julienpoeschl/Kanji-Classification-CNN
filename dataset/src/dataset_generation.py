"""
Not ment to be imported!

Run this file to generate a dataset using the fonts (.ttf or .otf) in dataset/data/fonts/ and dataset/data/kanji_1000.txt.
The results will be written into dataset/data/raw_images/.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Raw image file naming convention
def get_file_name(kanji_idx: int, font_idx: int, kanji: str) -> str:
    return f"{kanji_idx:04d}_{font_idx:03d}_{kanji}.png"

# Dir and paths
DATASET_DIR = os.path.join("dataset", "data")
DATASET_FONTS_DIR = os.path.join(DATASET_DIR, "fonts")
RAW_IMAGES_DIR = os.path.join(DATASET_DIR, "raw_images")
KANJI_LIST_PATH = os.path.join(DATASET_DIR, "kanji_1000.txt")
FONT_PATHS = [os.path.join(dirpath,f) for (dirpath, _, filenames) in os.walk(DATASET_FONTS_DIR) for f in filenames]

# Image and font sizes
IMAGE_SIZE = 256
FONT_SIZE = 220


kanji_list = open(KANJI_LIST_PATH, encoding="utf-8").read().splitlines()

os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(DATASET_FONTS_DIR, exist_ok=True)

diagnostics : dict[str, int] = {}

for font_idx, path in enumerate(FONT_PATHS):
    font = ImageFont.truetype(path, FONT_SIZE)

    for kanji_idx, kanji in enumerate(kanji_list):

        img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), kanji, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        x = (IMAGE_SIZE - w) // 2
        y = (IMAGE_SIZE - h) // 2

        draw.text((x, y), kanji, fill=0, font=font)

        arr = np.array(img)
        if np.all(arr == 255):
            diagnostics.setdefault(path, 0)
            diagnostics[path] += 1
            continue


        filename = get_file_name(kanji_idx, font_idx, kanji)
        img.save(os.path.join(RAW_IMAGES_DIR, filename))


# Diagnostics printing
if len(FONT_PATHS) == 0:
    print("No fonts found. Skipped dataset generation.")

for k in diagnostics.keys():
    print(f"Skipped {diagnostics[k]} generations for {k}. Generated image was empty.")
