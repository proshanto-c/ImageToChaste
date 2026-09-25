"""
Unit tests for microscopy pre-processing and CLAHE.
"""

import numpy as np
import pytest

from imagetochaste.segmentation.preprocessor import load_image, preprocess_microscopy_image


def test_load_image_array():
    # 2D grayscale
    gray = np.zeros((40, 40), dtype=np.uint8)
    rgb = load_image(gray)
    assert rgb.shape == (40, 40, 3)

    # 3D RGB
    orig_rgb = np.ones((40, 40, 3), dtype=np.uint8) * 128
    res = load_image(orig_rgb)
    assert res.shape == (40, 40, 3)


def test_load_image_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_file_path_123.png")


def test_preprocess_microscopy_image():
    # Create image with artificial uneven illumination gradient
    h, w = 120, 120
    x = np.linspace(50, 200, w)
    gradient = np.tile(x, (h, 1)).astype(np.uint8)
    rgb_grad = np.stack([gradient]*3, axis=-1)

    enhanced = preprocess_microscopy_image(rgb_grad, gaussian_blur_kernel=(31, 31))
    assert enhanced.shape == (h, w, 3)
    assert enhanced.dtype == np.uint8
