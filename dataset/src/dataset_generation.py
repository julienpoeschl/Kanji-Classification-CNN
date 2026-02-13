"""
Assumes the existence of dataset.data.ioamjm.ttf and .kanji_1000.txt
"""

from PIL import Image, ImageDraw, ImageFont
import os

DATASET_DIR = os.path.join("dataset", "data")
DATASET_FONTS_DIR = os.path.join(DATASET_DIR, "fonts")
FONT_PATH = os.path.join(DATASET_DIR, "ipamjm.ttf")
IMAGE_SIZE = 256
FONT_SIZE = 220
KANJI_LIST = open(os.path.join(DATASET_DIR, "kanji_1000.txt"), encoding="utf-8").read().splitlines()

os.makedirs(os.path.join(DATASET_DIR, "raw_images"), exist_ok=True)

font_paths = [os.path.join(dirpath,f) for (dirpath, _, filenames) in os.walk(DATASET_FONTS_DIR) for f in filenames]

for font_idx, path in enumerate(font_paths):
    font = ImageFont.truetype(path, FONT_SIZE)

    for kanji_idx, kanji in enumerate(KANJI_LIST):
        if font.getbbox(kanji) is None:
            print(f"{font} doesn't have glyph {kanji}.")
            continue

        img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), kanji, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        x = (IMAGE_SIZE - w) // 2
        y = (IMAGE_SIZE - h) // 2

        draw.text((x, y), kanji, fill=0, font=font)

        filename = f"{kanji_idx:04d}_{font_idx:03d}_{kanji}.png"
        img.save(os.path.join(DATASET_DIR, "raw_images", filename))


