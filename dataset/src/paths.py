import os

DATA_DIR = os.path.join("dataset", "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
PROCESSED_INPUTS_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_inputs.npz")
PROCESSED_OUTPUTS_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_outputs.npz")
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_output_map.json")

KANJI_DATA_PATH = os.path.join(DATA_DIR, "kanji_1000_extra.json")