import numpy as np
import torch

from model.src.cnn_model import KanjiCNN


def evaluate(model: KanjiCNN, tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top3 = np.argsort(probs)[-3:][::-1]

        return top3, probs