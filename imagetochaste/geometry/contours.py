"""
Contour extraction and polygon boundary simplification for cell masks.
"""

from typing import Any, Dict, List, Union

import cv2
import numpy as np


def extract_contours(
    binary_mask: np.ndarray,
    epsilon_factor: float = 0.01,
    min_contour_area: float = 10.0,
) -> List[np.ndarray]:
    """
    Extract smoothed external polygonal boundaries from a binary mask.

    Uses cv2.findContours and Douglas-Peucker polygon simplification (approxPolyDP)
    to convert pixel-grid borders into compact geometric vertices.

    Args:
        binary_mask: 2D binary numpy array representing a single segmented cell.
        epsilon_factor: Multiplier for contour perimeter determining simplification tolerance.
            epsilon = epsilon_factor * arcLength. Smaller values retain more detail.
        min_contour_area: Minimum pixel area to consider as a valid cell contour.

    Returns:
        List of 2D numpy arrays with shape (V, 2) containing ordered (x, y) vertices.
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    raw_contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    polygons: List[np.ndarray] = []
    for c in raw_contours:
        area = cv2.contourArea(c)
        if area < min_contour_area:
            continue

        perimeter = cv2.arcLength(c, closed=True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(c, epsilon=epsilon, closed=True)

        # Reshape from (V, 1, 2) to (V, 2) with float coordinates
        pts = approx.reshape(-1, 2).astype(np.float64)
        if len(pts) >= 3:
            polygons.append(pts)

    return polygons


def extract_all_contours(
    masks: List[Union[Dict[str, Any], np.ndarray]],
    epsilon_factor: float = 0.01,
    min_contour_area: float = 10.0,
) -> List[np.ndarray]:
    """
    Extract the primary external boundary polygon for every mask in a list.

    Returns:
        List of (V, 2) coordinate arrays, one per segmented cell.
    """
    cell_polygons: List[np.ndarray] = []

    for m in masks:
        bin_mask = m["segmentation"] if isinstance(m, dict) else m
        polys = extract_contours(
            bin_mask,
            epsilon_factor=epsilon_factor,
            min_contour_area=min_contour_area,
        )
        if polys:
            # Pick the largest external contour for the cell
            largest_poly = max(polys, key=lambda p: cv2.contourArea(p.astype(np.float32)))
            cell_polygons.append(largest_poly)

    return cell_polygons
