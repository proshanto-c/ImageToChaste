"""
Microscopy image pre-processing module.

Implements Contrast Limited Adaptive Histogram Equalization (CLAHE),
Gaussian background subtraction for illumination correction, and intensity normalization.
"""

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image


def load_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Ensure the image is loaded into an RGB uint8 numpy array.
    """
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        if not Path(path_str).exists():
            raise FileNotFoundError(f"Image not found at: {path_str}")
        pil_img = Image.open(path_str)
        return np.array(pil_img.convert("RGB"))
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3 and image_input.shape[2] == 3:
            return image_input
        elif image_input.ndim == 3 and image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def preprocess_microscopy_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    clahe_clip_limit: float = 3.0,
    clahe_grid_size: Tuple[int, int] = (8, 8),
    gaussian_blur_kernel: Tuple[int, int] = (101, 101),
    invert: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Enhance microscopy images for robust SAM 2 segmentation.

    Args:
        image: Path to image, PIL Image, or numpy array.
        clahe_clip_limit: Threshold for contrast limiting in CLAHE.
        clahe_grid_size: Size of grid for histogram equalization (rows, cols).
        gaussian_blur_kernel: Kernel size for estimating background illumination (must be odd).
        invert: Whether to invert intensities (e.g. For brightfield vs fluorescence).
        normalize: Whether to scale pixel values across full 0-255 range.

    Returns:
        RGB uint8 numpy array with enhanced cell boundaries and flattened background.
    """
    rgb_arr = load_image(image)

    # Convert to single-channel grayscale for contrast adjustment
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

    if invert:
        gray = cv2.bitwise_not(gray)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    clahe_img = clahe.apply(gray)

    # Ensure kernel dimensions are odd
    k_w = gaussian_blur_kernel[0] if gaussian_blur_kernel[0] % 2 == 1 else gaussian_blur_kernel[0] + 1
    k_h = gaussian_blur_kernel[1] if gaussian_blur_kernel[1] % 2 == 1 else gaussian_blur_kernel[1] + 1

    # Subtract uneven background illumination
    background = cv2.GaussianBlur(clahe_img, (k_w, k_h), 0)
    flattened = cv2.subtract(clahe_img, background)

    if normalize:
        flattened = cv2.normalize(flattened, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Convert back to 3-channel RGB for SAM 2 input
    enhanced_rgb = cv2.cvtColor(flattened.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return enhanced_rgb
