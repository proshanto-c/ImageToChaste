"""
Centroid computation for binary masks and 2D polygonal vertices.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def compute_centroid(binary_mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute the center of mass (Cx, Cy) for a 2D binary mask.

    Args:
        binary_mask: 2D numpy array where positive values represent the cell body.

    Returns:
        (x_centroid, y_centroid) as floating point coordinates, or None if empty.
    """
    y_coords, x_coords = np.where(binary_mask > 0)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None

    centroid_x = float(np.mean(x_coords))
    centroid_y = float(np.mean(y_coords))
    return centroid_x, centroid_y


def compute_centroids_from_masks(
    masks: List[Union[Dict[str, Any], np.ndarray]]
) -> List[Tuple[float, float]]:
    """
    Extract centroids for a list of SAM 2 mask dictionaries or binary arrays.
    """
    centroids: List[Tuple[float, float]] = []
    for m in masks:
        bin_mask = m["segmentation"] if isinstance(m, dict) else m
        c = compute_centroid(bin_mask)
        if c is not None:
            centroids.append(c)
    return centroids


def compute_polygon_centroid(vertices: np.ndarray) -> Tuple[float, float]:
    """
    Compute the analytical center of mass of a 2D polygon using Green's Theorem.

    Args:
        vertices: Array of shape (N, 2) defining ordered polygon vertices (x, y).

    Returns:
        (Cx, Cy) centroid coordinate.
    """
    pts = np.asarray(vertices, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError(f"Polygon must have at least 3 vertices, got {len(pts)}")

    # Ensure closed polygon loop for summation
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    x = pts[:, 0]
    y = pts[:, 1]

    # Cross term (x_i * y_{i+1} - x_{i+1} * y_i)
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_area = 0.5 * np.sum(cross)

    if np.isclose(signed_area, 0.0):
        # Degenerate collinear polygon: fallback to vertex mean
        return float(np.mean(pts[:-1, 0])), float(np.mean(pts[:-1, 1]))

    cx = (1.0 / (6.0 * signed_area)) * np.sum((x[:-1] + x[1:]) * cross)
    cy = (1.0 / (6.0 * signed_area)) * np.sum((y[:-1] + y[1:]) * cross)

    return float(cx), float(cy)
