import os

CHECKPOINT_DIR = os.path.join("model", "checkpoints")
BEST_MODEL_NAME = "best_model.pt"
FINAL_MODEL_NAME = "final_model.pt"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, FINAL_MODEL_NAME)