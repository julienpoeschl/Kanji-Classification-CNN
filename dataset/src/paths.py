import os

DATA_DIR = os.path.join("dataset", "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_output_map.json")

KANJI_DATA_PATH = os.path.join(DATA_DIR, "kanji_1000_extra.json")