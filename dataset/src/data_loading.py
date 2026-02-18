import json
from pathlib import Path
from PIL import Image
import numpy as np

from dataset.src.paths import KANJI_DATA_PATH, LABEL_MAP_PATH


def load_images_and_labels(image_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images and their labels.

    To correctly identify the label that is part of the images name, the name structure "kid_fid_kanji.png" (kid=kanji id, fid=font id) is assumed.
    
    :param image_dir: Directory of folder containing images (png).
    :type image_dir: Path
    :return: Images and their labels as numpy arrays.
    :rtype: tuple[ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]]]
    """
    images: list[np.ndarray] = []
    labels: list[str] = []
    for img_path in sorted(image_dir.glob("*.png")):
        img = Image.open(img_path)
        img = np.array(img)
        images.append(img)
        label = img_path.name.split("_")[2].split(".")[0]
        labels.append(label)
    return np.array(images), np.array(labels)


def load_kanji_data() -> dict:
    with open(KANJI_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_output_map() -> dict:
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)