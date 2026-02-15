import numpy as np
from scipy import ndimage
from dataclasses import dataclass
from typing import Optional


@dataclass
class AugmentationSettings:
    """Configuration for data augmentation transforms."""
    
    # Rotation settings
    rotation_enabled: bool = True
    rotation_range: float = 15.0  # Max degrees in either direction
    
    # Scale settings
    scale_enabled: bool = True
    scale_range: tuple[float, float] = (0.85, 1.15)  # Min and max scale factors
    
    # Noise settings
    noise_enabled: bool = True
    noise_stddev: float = 0.05  # Standard deviation of Gaussian noise (relative to [0,1] range)
    
    # Brightness/contrast settings (color adjustment for grayscale)
    brightness_enabled: bool = True
    brightness_range: tuple[float, float] = (-0.2, 0.2)  # Additive brightness adjustment
    contrast_enabled: bool = True
    contrast_range: tuple[float, float] = (0.8, 1.2)  # Multiplicative contrast adjustment
    
    # Translation/shift settings
    shift_enabled: bool = True
    shift_range: float = 0.1  # Max shift as fraction of image size
    
    # Elastic deformation (useful for handwritten characters)
    elastic_enabled: bool = False
    elastic_alpha: float = 20.0  # Intensity of deformation
    elastic_sigma: float = 3.0  # Smoothness of deformation

    # Inversion of colors
    inversion_enabled: bool = True
    inversion_probability: float = 0.5 # [0,1]
    
    # General settings
    augmentation_factor: int = 1  # Number of augmented copies per original image
    keep_originals: bool = True  # Whether to include original images in output


class DataAugmenter:
    """
    Data augmentation class for grayscale image datasets.
    
    Applies various transformations to expand training data:
    - Rotation
    - Scaling
    - Gaussian noise
    - Brightness/contrast adjustment
    - Translation/shifting
    - Elastic deformation
    """

    def __init__(self, settings: Optional[AugmentationSettings] = None) -> None:
        """
        Initialize the DataAugmenter with augmentation settings.
        
        :param settings: AugmentationSettings dataclass with transform parameters.
                        If None, uses default settings.
        """
        self._settings = settings if settings is not None else AugmentationSettings()
        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def _rotate(self, image: np.ndarray, cval: float) -> np.ndarray:
        """Apply random rotation to image."""
        angle = self._rng.uniform(-self._settings.rotation_range, self._settings.rotation_range)
        return ndimage.rotate(image, angle, reshape=False, mode='constant', cval=cval)

    def _scale(self, image: np.ndarray) -> np.ndarray:
        """Apply random scaling to image, maintaining original size."""
        scale = self._rng.uniform(*self._settings.scale_range)
        h, w = image.shape[:2]
        
        # Zoom the image
        zoomed = ndimage.zoom(image, (scale, scale), order=1)
        
        # Create output image of original size
        result: np.ndarray = np.zeros_like(image)

        if isinstance(zoomed, np.ndarray) and len(zoomed.shape) >= 2:
            zh: int = zoomed.shape[0]
            zw: int = zoomed.shape[1]
            
            if scale > 1:
                # Crop center
                start_h = (zh - h) // 2
                start_w = (zw - w) // 2
                result = zoomed[start_h:start_h + h, start_w:start_w + w]
            else:
                # Pad center
                start_h = (h - zh) // 2
                start_w = (w - zw) // 2
                result[start_h:start_h + zh, start_w:start_w + zw] = zoomed
            
        return result

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = self._rng.normal(0, self._settings.noise_stddev, image.shape)
        return np.clip(image + noise, 0, 1)
    
    def _invert(self, image : np.ndarray) -> np.ndarray:
        """Invert colors of image."""
        if self._rng.random() < self._settings.inversion_probability:
            image = 1 - image
        return image

    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness adjustment."""
        brightness = self._rng.uniform(*self._settings.brightness_range)
        return np.clip(image + brightness, 0, 1)

    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply random contrast adjustment."""
        contrast = self._rng.uniform(*self._settings.contrast_range)
        mean = np.mean(image)
        return np.clip((image - mean) * contrast + mean, 0, 1)

    def _shift(self, image: np.ndarray, cval: float) -> np.ndarray:
        """Apply random translation/shift to image."""
        h, w = image.shape[:2]
        max_shift_h = int(h * self._settings.shift_range)
        max_shift_w = int(w * self._settings.shift_range)
        
        shift_h = self._rng.integers(-max_shift_h, max_shift_h + 1)
        shift_w = self._rng.integers(-max_shift_w, max_shift_w + 1)
        
        return ndimage.shift(image, (shift_h, shift_w), mode='constant', cval=cval)

    def _elastic_deform(self, image: np.ndarray, cval: float) -> np.ndarray:
        """Apply elastic deformation to image."""
        h, w = image.shape[:2]
        alpha = self._settings.elastic_alpha
        sigma = self._settings.elastic_sigma
        
        # Random displacement fields
        dx = self._rng.uniform(-1, 1, (h, w)) * alpha
        dy = self._rng.uniform(-1, 1, (h, w)) * alpha
        
        # Smooth the displacement fields
        dx = ndimage.gaussian_filter(dx, sigma)
        dy = ndimage.gaussian_filter(dy, sigma)
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        coords = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]
        
        return ndimage.map_coordinates(image, coords, order=1, mode='constant', cval=cval)

    def augment_single(self, image: np.ndarray) -> np.ndarray:
        """Apply all enabled augmentations to a single image."""
        result = image.copy()
        
        # Normalize to [0, 1] if needed
        was_uint8 = result.dtype == np.uint8
        if was_uint8:
            result = result.astype(np.float32) / 255.0
        
        background = 0.0 if np.mean(result) > 0.5 else 1.0

        # Apply geometric transforms first
        if self._settings.rotation_enabled:
            result = self._rotate(result, background)
            
        if self._settings.scale_enabled:
            result = self._scale(result)
            
        if self._settings.shift_enabled:
            result = self._shift(result, background)
            
        if self._settings.elastic_enabled:
            result = self._elastic_deform(result, background)

        if self._settings.inversion_enabled:
            result = self._invert(result)
        
        # Apply intensity transforms
        if self._settings.contrast_enabled:
            result = self._adjust_contrast(result)
            
        if self._settings.brightness_enabled:
            result = self._adjust_brightness(result)
            
        if self._settings.noise_enabled:
            result = self._add_noise(result)
        
        # Convert back if needed
        if was_uint8:
            result = (result * 255).astype(np.uint8)
            
        return result

    def augment(self, inputs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset by generating transformed versions of input images.
        
        :param inputs: Array of grayscale images with shape (N, H, W) or (N, H, W, 1)
        :param labels: Corresponding labels array with shape (N,) or (N, num_classes)
        :return: Tuple of (augmented_inputs, augmented_labels) with expanded dataset
        """
        # Handle channel dimension
        squeeze_channel = False
        if inputs.ndim == 4 and inputs.shape[-1] == 1:
            inputs = inputs.squeeze(axis=-1)
            squeeze_channel = True
        
        n_samples = len(inputs)
        augmented_inputs = []
        augmented_labels = []
        
        # Optionally keep original samples
        if self._settings.keep_originals:
            augmented_inputs.append(inputs)
            augmented_labels.append(labels)
        
        # Generate augmented samples
        for _ in range(self._settings.augmentation_factor):
            batch_augmented = np.array([self.augment_single(img) for img in inputs])
            augmented_inputs.append(batch_augmented)
            augmented_labels.append(labels)
        
        # Combine all samples
        result_inputs = np.concatenate(augmented_inputs, axis=0)
        result_labels = np.concatenate(augmented_labels, axis=0)
        
        # Restore channel dimension if needed
        if squeeze_channel:
            result_inputs = result_inputs[..., np.newaxis]
        
        # Shuffle the dataset
        indices = self._rng.permutation(len(result_inputs))
        result_inputs = result_inputs[indices]
        result_labels = result_labels[indices]
        
        return result_inputs, result_labels
