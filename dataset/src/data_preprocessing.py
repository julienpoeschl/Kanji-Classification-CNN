import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal

from data_loading import load_images_and_labels


class PaddingMode(Enum):
    """Padding modes for image resizing."""
    CONSTANT = "constant"      # Pad with constant value (default: 0)
    EDGE = "edge"              # Pad with edge values (clamp)
    REFLECT = "reflect"        # Reflect at boundaries
    SYMMETRIC = "symmetric"    # Symmetric reflection at boundaries


class NormalizationMode(Enum):
    """Normalization modes for pixel values."""
    UNIT = "unit"              # Scale to [0, 1]
    CENTERED = "centered"      # Scale to [-1, 1]
    STANDARDIZE = "standardize"  # Zero mean, unit variance
    NONE = "none"              # No normalization


@dataclass
class PreprocessingSettings:
    """Configuration for data preprocessing transforms."""
    
    # Grayscale conversion
    grayscale_enabled: bool = True
    grayscale_weights: tuple[float, float, float] = (0.299, 0.587, 0.114)  # RGB to grayscale weights
    
    # Resizing settings
    resize_enabled: bool = True
    target_size: tuple[int, int] = (64, 64)  # (height, width)
    preserve_aspect_ratio: bool = True  # If True, resize then pad; if False, stretch
    padding_mode: PaddingMode = PaddingMode.CONSTANT
    padding_value: float = 0.0  # Used when padding_mode is CONSTANT
    center_content: bool = True  # Center the image within padding
    
    # Normalization settings
    normalization_mode: NormalizationMode = NormalizationMode.UNIT
    custom_mean: Optional[float] = None  # For standardization, if None compute from data
    custom_std: Optional[float] = None   # For standardization, if None compute from data
    
    # Inversion (useful when dataset has mixed white-on-black / black-on-white)
    invert_enabled: bool = False
    
    # Thresholding/binarization
    threshold_enabled: bool = False
    threshold_value: float = 0.5  # Threshold for binarization (after normalization to 0-1)
    
    # Smoothing/denoising
    smooth_enabled: bool = False
    smooth_sigma: float = 0.5  # Gaussian blur sigma
    
    # Channel dimension
    add_channel_dim: bool = True  # Add channel dimension for CNN (N, H, W) -> (N, H, W, 1)
    channel_first: bool = False   # If True: (N, 1, H, W), else (N, H, W, 1)
    
    # Output dtype
    output_dtype: Literal["float32", "float64", "uint8"] = "float32"


class DataPreprocessor:
    """
    Data preprocessing class for image datasets.
    
    Applies various preprocessing transforms:
    - Grayscale conversion
    - Resizing with aspect ratio preservation and padding
    - Normalization (unit, centered, standardization)
    - Inversion
    - Thresholding/binarization
    - Smoothing/denoising
    """

    def __init__(self, settings: Optional[PreprocessingSettings] = None) -> None:
        """
        Initialize the DataPreprocessor with preprocessing settings.
        
        :param settings: PreprocessingSettings dataclass with transform parameters.
                        If None, uses default settings.
        """
        self._settings = settings if settings is not None else PreprocessingSettings()
        self._computed_mean: Optional[float] = None
        self._computed_std: Optional[float] = None

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB/RGBA image to grayscale."""
        if image.ndim == 2:
            # Already grayscale
            return image
        
        if image.ndim == 3:
            if image.shape[-1] == 1:
                # Single channel, just squeeze
                return image.squeeze(axis=-1)
            elif image.shape[-1] == 3:
                # RGB
                weights = np.array(self._settings.grayscale_weights)
                return np.dot(image[..., :3], weights)
            elif image.shape[-1] == 4:
                # RGBA - ignore alpha
                weights = np.array(self._settings.grayscale_weights)
                return np.dot(image[..., :3], weights)
        
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")

    def _resize_single(self, image: np.ndarray) -> np.ndarray:
        """Resize a single image to target size with padding if needed."""
        target_h, target_w = self._settings.target_size
        h, w = image.shape[:2]
        
        if not self._settings.preserve_aspect_ratio:
            # Simple stretch resize using zoom
            zoom_h = target_h / h
            zoom_w = target_w / w
            zoomed = np.asarray(ndimage.zoom(image, (zoom_h, zoom_w), order=1))
            return zoomed
        
        # Calculate scale to fit within target while preserving aspect ratio
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize image
        resized: np.ndarray
        if scale != 1.0:
            zoom_factor = scale
            resized = np.asarray(ndimage.zoom(image, zoom_factor, order=1))
            # Ensure exact dimensions due to rounding
            resized = resized[:new_h, :new_w]
        else:
            resized = image
        
        # Create padded output
        result = np.full((target_h, target_w), self._settings.padding_value, dtype=image.dtype)
        
        # Calculate padding offsets
        if self._settings.center_content:
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
        else:
            pad_h = 0
            pad_w = 0
        
        # Handle edge cases where resized might be slightly off
        resized_h: int = resized.shape[0]
        resized_w: int = resized.shape[1]
        actual_h = min(new_h, resized_h, target_h - pad_h)
        actual_w = min(new_w, resized_w, target_w - pad_w)
        
        result[pad_h:pad_h + actual_h, pad_w:pad_w + actual_w] = resized[:actual_h, :actual_w]
        
        # Apply padding mode for edges if not constant
        if self._settings.padding_mode != PaddingMode.CONSTANT:
            result = self._apply_padding_mode(result, pad_h, pad_w, actual_h, actual_w)
        
        return result

    def _apply_padding_mode(self, image: np.ndarray, pad_h: int, pad_w: int, 
                            content_h: int, content_w: int) -> np.ndarray:
        """Apply non-constant padding modes to the padded regions."""
        target_h, target_w = image.shape[:2]
        
        if self._settings.padding_mode == PaddingMode.EDGE:
            # Fill top/bottom edges
            if pad_h > 0:
                image[:pad_h, pad_w:pad_w + content_w] = image[pad_h, pad_w:pad_w + content_w]
                image[pad_h + content_h:, pad_w:pad_w + content_w] = image[pad_h + content_h - 1, pad_w:pad_w + content_w]
            # Fill left/right edges
            if pad_w > 0:
                image[:, :pad_w] = image[:, pad_w:pad_w + 1]
                image[:, pad_w + content_w:] = image[:, pad_w + content_w - 1:pad_w + content_w]
        
        elif self._settings.padding_mode in (PaddingMode.REFLECT, PaddingMode.SYMMETRIC):
            # For reflect/symmetric, we'd need more complex logic
            # This is a simplified version that uses edge values
            mode = 'reflect' if self._settings.padding_mode == PaddingMode.REFLECT else 'symmetric'
            # Re-extract content and pad properly
            content = image[pad_h:pad_h + content_h, pad_w:pad_w + content_w]
            pad_top = pad_h
            pad_bottom = target_h - pad_h - content_h
            pad_left = pad_w
            pad_right = target_w - pad_w - content_w
            image = np.pad(content, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
        
        return image

    def _normalize(self, images: np.ndarray) -> np.ndarray:
        """Apply normalization to images."""
        mode = self._settings.normalization_mode
        
        if mode == NormalizationMode.NONE:
            return images
        
        # Ensure float type for normalization
        images = images.astype(np.float64)
        
        # First normalize to [0, 1] if values are in uint8 range
        if images.max() > 1.0:
            images = images / 255.0
        
        if mode == NormalizationMode.UNIT:
            # Already in [0, 1]
            return images
        
        elif mode == NormalizationMode.CENTERED:
            # Scale to [-1, 1]
            return images * 2.0 - 1.0
        
        elif mode == NormalizationMode.STANDARDIZE:
            # Zero mean, unit variance
            if self._settings.custom_mean is not None:
                mean = self._settings.custom_mean
            else:
                mean = np.mean(images)
                self._computed_mean = float(mean)
            
            if self._settings.custom_std is not None:
                std = self._settings.custom_std
            else:
                std = np.std(images)
                self._computed_std = float(std)
            
            if std < 1e-8:
                std = 1.0  # Avoid division by zero
            
            return (images - mean) / std
        
        return images

    def _invert(self, image: np.ndarray) -> np.ndarray:
        """Invert image values."""
        if image.max() <= 1.0:
            return 1.0 - image
        else:
            return 255 - image

    def _threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply binary thresholding."""
        threshold = self._settings.threshold_value
        if image.max() > 1.0:
            threshold = threshold * 255
        return (image > threshold).astype(image.dtype) * (1.0 if image.max() <= 1.0 else 255)

    def _smooth(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing."""
        return ndimage.gaussian_filter(image, sigma=self._settings.smooth_sigma)

    def _process_single(self, image: np.ndarray) -> np.ndarray:
        """Apply all enabled preprocessing to a single image."""
        result = image.copy()
        
        # Grayscale conversion (should be done first)
        if self._settings.grayscale_enabled:
            result = self._to_grayscale(result)
        
        # Smoothing (before resize to reduce aliasing)
        if self._settings.smooth_enabled:
            result = self._smooth(result)
        
        # Resize
        if self._settings.resize_enabled:
            result = self._resize_single(result)
        
        # Inversion
        if self._settings.invert_enabled:
            result = self._invert(result)
        
        # Thresholding
        if self._settings.threshold_enabled:
            result = self._threshold(result)
        
        return result

    def get_computed_stats(self) -> tuple[Optional[float], Optional[float]]:
        """
        Get computed mean and std from last process() call.
        Useful for applying same normalization to test data.
        
        :return: Tuple of (mean, std), or (None, None) if not computed
        """
        return self._computed_mean, self._computed_std

    def process(self, inputs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess a dataset of images.
        
        :param inputs: Array of images with shape (N, H, W), (N, H, W, C), or (N, C, H, W)
        :param labels: Optional corresponding labels array
        :return: Tuple of (processed_inputs, labels) - labels passed through unchanged
        """
        # Process each image
        processed = np.array([self._process_single(img) for img in inputs])
        
        # Apply normalization across entire dataset
        processed = self._normalize(processed)
        
        # Add channel dimension if requested
        if self._settings.add_channel_dim and processed.ndim == 3:
            if self._settings.channel_first:
                processed = processed[:, np.newaxis, :, :]  # (N, 1, H, W)
            else:
                processed = processed[..., np.newaxis]  # (N, H, W, 1)
        
        # Convert to output dtype
        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "uint8": np.uint8,
        }
        output_dtype = dtype_map[self._settings.output_dtype]
        
        if output_dtype == np.uint8:
            # Scale back to 0-255 for uint8
            if processed.max() <= 1.0 and processed.min() >= 0:
                processed = (processed * 255).astype(np.uint8)
            elif processed.min() >= -1.0 and processed.max() <= 1.0:
                processed = ((processed + 1) * 127.5).astype(np.uint8)
            else:
                # Standardized - clip and scale
                processed = np.clip(processed, -3, 3)
                processed = ((processed + 3) / 6 * 255).astype(np.uint8)
        else:
            processed = processed.astype(output_dtype)
        
        return processed, labels


if __name__ == "__main__":

    import os
    import random
    from pathlib import Path
    import json
    from PIL import Image

    data_preprocessor = DataPreprocessor(
        PreprocessingSettings(
            grayscale_enabled=True,
            resize_enabled=True,
            target_size=(64, 64),
            preserve_aspect_ratio=True,
            padding_mode=PaddingMode.CONSTANT,
            padding_value=1.0,          # white background after normalization
            normalization_mode=NormalizationMode.UNIT,
            invert_enabled=False,
            add_channel_dim=True,
            channel_first=False,        # TensorFlow-style (N, H, W, 1)
            output_dtype="float32"
        )
    )

    DATA_DIR = os.path.join("dataset", "data")
    RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")
    images, labels = load_images_and_labels(Path(RAW_IMAGES_DIR))
    
    unique_labels = list(dict.fromkeys(labels))
    output_map = {val: i for i, val in enumerate(unique_labels)}

    # Convert labels to numeric using output_map
    numeric_labels = np.array([output_map[label] for label in labels])

    processed_images, _ = data_preprocessor.process(images, labels)

    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    PROCESSED_INPUTS = os.path.join(PROCESSED_DATA_DIR, "processed_inputs.npz")
    PROCESSED_OUTPUTS = os.path.join(PROCESSED_DATA_DIR, "processed_outputs.npz")
    PROCESSED_OUTPUT_MAP = os.path.join(PROCESSED_DATA_DIR, "processed_output_map.json")

    np.savez(PROCESSED_INPUTS, arr=processed_images)
    np.savez(PROCESSED_OUTPUTS, arr=numeric_labels)

    with open(PROCESSED_OUTPUT_MAP, "w", encoding="utf-8") as f:
        json.dump(output_map, f, indent=4)

    # Save 5 random input images for preview
    PREVIEW_DIR = os.path.join(PROCESSED_DATA_DIR, "preview_images")
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    
    num_samples = min(5, len(processed_images))
    random_indices = random.sample(range(len(processed_images)), num_samples)
    
    for i, idx in enumerate(random_indices):
        img_array = processed_images[idx]
        # Remove channel dimension if present and convert to uint8
        if img_array.ndim == 3:
            img_array = img_array.squeeze()
        # Scale to 0-255 if normalized
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        label = labels[idx]
        img = Image.fromarray(img_array, mode='L')
        img.save(os.path.join(PREVIEW_DIR, f"sample_{i+1}_{label}.png"))
    
    print(f"Saved {num_samples} preview images to {PREVIEW_DIR}")

