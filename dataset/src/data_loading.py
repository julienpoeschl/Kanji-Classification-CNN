from pathlib import Path
from PIL import Image
import numpy as np


def load_images(image_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    for img_path in sorted(image_dir.glob("*.png")):
        img = Image.open(img_path)
        img = np.array(img)
        images.append(img)
        label = img_path.name.split("_")[2].split(".")[0]
        labels.append(label)
    return np.array(images), np.array(labels)