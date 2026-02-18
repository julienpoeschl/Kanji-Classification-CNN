import torch
from dataset.src.data_preprocessing import DEFAULT_SETTINGS, DataPreprocessor
from dataset.src.data_loading import load_kanji_data as _load_kanji_data, load_output_map as _load_output_map
import numpy as np
from PIL.Image import Image
from model.src.cnn_model import create_model as _create_model, load_checkpoint as _load_checkpoint
from model.src.cuda import get_device as _get_device
from model.src.evaluation import evaluate as _evaluate

class KanjiController:

    def __init__(self) -> None:
        self._evaluate = _evaluate
        self.model, self.idx_to_class = self._load_best_model_and_idx_to_class(_create_model, _load_checkpoint, _get_device, _load_output_map)
        self.model.eval()
        self.preprocessor = self._get_preprocessor()

    def get_idx_to_class(self) -> dict:
        return self.idx_to_class

    def get_top_3_predictions(self, image: Image) -> tuple[np.ndarray, np.ndarray]:
        if image is None:
            return None
        inp = self.preprocessor.process_single(np.array(image))
        tensor = torch.from_numpy(inp)
        if tensor.ndim == 4 and tensor.shape[-1] == 1:
            tensor = tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return self._evaluate(self.model, tensor)


    def _load_best_model_and_idx_to_class(self, _create_model, _load_checkpoint, _get_device, _load_output_map) -> tuple:
        output_map = _load_output_map()
        idx_to_class = {v: k for k, v in output_map.items()}
        num_classes = len(output_map)
        device = _get_device()
        model = _create_model(num_classes=num_classes, device=device)
        checkpoint = _load_checkpoint()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, idx_to_class

    def _get_preprocessor(self) -> DataPreprocessor:
        return DataPreprocessor(DEFAULT_SETTINGS)

def get_kanji_data() -> dict:
    return _load_kanji_data()