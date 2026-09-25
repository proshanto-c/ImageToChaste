"""
Pytest fixtures for ImageToChaste test suite.
"""

import numpy as np
import pytest


@pytest.fixture
def synthetic_circle_mask():
    """
    100x100 binary mask containing a centered circle of radius 25.
    """
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.uint8)
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - 50) ** 2 + (y - 50) ** 2
    mask[dist_sq <= 25 ** 2] = 1
    return mask


@pytest.fixture
def synthetic_adjacent_masks():
    """
    Two adjacent 100x100 square cell masks touching along a vertical boundary.
    """
    h, w = 100, 100
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)

    # Cell 1: x from 20 to 50, y from 30 to 70
    mask1[30:70, 20:50] = 1
    # Cell 2: x from 50 to 80, y from 30 to 70
    mask2[30:70, 50:80] = 1

    return [
        {"segmentation": mask1, "area": int(np.sum(mask1))},
        {"segmentation": mask2, "area": int(np.sum(mask2))},
    ]


@pytest.fixture
def sample_centroids():
    """
    A list of 6 2D points suitable for Voronoi meshing.
    """
    return [
        (10.0, 10.0),
        (30.0, 12.0),
        (20.0, 35.0),
        (45.0, 40.0),
        (15.0, 55.0),
        (40.0, 60.0),
    ]
