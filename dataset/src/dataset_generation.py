"""
Assumes the existence of dataset.data.ioamjm.ttf and .kanji_1000.txt
"""

from PIL import Image, ImageDraw, ImageFont
import os

DATASET_DIR = os.path.join("dataset", "data")
FONT_PATH = os.path.join(DATASET_DIR, "ipamjm.ttf")
IMAGE_SIZE = 256
FONT_SIZE = 220
KANJI_LIST = open(os.path.join(DATASET_DIR, "kanji_1000.txt"), encoding="utf-8").read().splitlines()

os.makedirs(os.path.join(DATASET_DIR, "raw_images"), exist_ok=True)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

for idx, kanji in enumerate(KANJI_LIST):
    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), kanji, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = (IMAGE_SIZE - w) // 2
    y = (IMAGE_SIZE - h) // 2

    draw.text((x, y), kanji, fill=0, font=font)

    filename = f"{idx:04d}_{kanji}.png"
    img.save(os.path.join(DATASET_DIR, "raw_images", filename))


