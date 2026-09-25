"""
Visualization and rendering utilities for segmentation masks and prompts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create_mask_overlay(
    image: np.ndarray,
    masks: List[Union[Dict[str, Any], np.ndarray]],
    alpha: float = 0.5,
    draw_borders: bool = True,
    border_color: Tuple[int, int, int] = (255, 255, 255),
    border_thickness: int = 1,
) -> np.ndarray:
    """
    Render a composite RGB overlay image with colored masks and boundary outlines.

    Args:
        image: Original RGB uint8 image.
        masks: List of SAM 2 mask dicts or binary arrays.
        alpha: Opacity blending factor (0.0 to 1.0).
        draw_borders: Whether to outline mask perimeters.
        border_color: RGB tuple for perimeter line.
        border_thickness: Thickness of perimeter line in pixels.

    Returns:
        RGB uint8 numpy array with visualized masks.
    """
    overlay = image.copy().astype(np.float32)
    np.random.seed(42)

    for m in masks:
        bin_mask = m["segmentation"] if isinstance(m, dict) else m
        if not np.any(bin_mask):
            continue

        # Generate vibrant pseudo-random color for each cell
        color = np.random.randint(40, 240, size=3, dtype=np.uint8).astype(np.float32)
        idx = bin_mask > 0

        # Blend mask color onto base image
        overlay[idx] = (1.0 - alpha) * overlay[idx] + alpha * color

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if draw_borders:
        for m in masks:
            bin_mask = m["segmentation"] if isinstance(m, dict) else m
            contours, _ = cv2.findContours(
                bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(overlay, contours, -1, border_color, border_thickness)

    return overlay


def save_mask_visualization(
    image: np.ndarray,
    masks: List[Union[Dict[str, Any], np.ndarray]],
    output_path: Union[str, Path],
    title: Optional[str] = None,
) -> Path:
    """
    Save a high-resolution figure of the segmentation overlay to disk.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    overlay = create_mask_overlay(image, masks)
    ax.imshow(overlay)
    if title:
        ax.set_title(title, fontsize=14)
    ax.axis("off")

    fig.savefig(str(out_file), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_file
