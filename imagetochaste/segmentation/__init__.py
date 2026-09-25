"""
Segmentation module for SAM 2 adapter, pre-processing, filtering, and visualization.
"""

from imagetochaste.segmentation.filter import filter_masks_by_area
from imagetochaste.segmentation.preprocessor import load_image, preprocess_microscopy_image
from imagetochaste.segmentation.sam_adapter import SAMAdapter, resolve_device
from imagetochaste.segmentation.utils import create_mask_overlay, save_mask_visualization

__all__ = [
    "SAMAdapter",
    "resolve_device",
    "preprocess_microscopy_image",
    "load_image",
    "filter_masks_by_area",
    "create_mask_overlay",
    "save_mask_visualization",
]
