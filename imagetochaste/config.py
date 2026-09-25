"""
Configuration defaults and model checkpoint registries for ImageToChaste.
"""

from typing import Any, Dict

# Official Meta SAM 2 release checkpoint URLs
SAM2_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "config": "configs/sam2/sam2_hiera_t.yaml",
        "filename": "sam2_hiera_tiny.pt",
    },
    "small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "config": "configs/sam2/sam2_hiera_s.yaml",
        "filename": "sam2_hiera_small.pt",
    },
    "base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "config": "configs/sam2/sam2_hiera_b+.yaml",
        "filename": "sam2_hiera_base_plus.pt",
    },
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "config": "configs/sam2/sam2_hiera_l.yaml",
        "filename": "sam2_hiera_large.pt",
    },
    "sam2.1_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "filename": "sam2.1_hiera_large.pt",
    },
}

# Master's thesis calibrated default AMG parameters
DEFAULT_AMG_PARAMS: Dict[str, Any] = {
    "points_per_side": 32,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.75,
    "stability_score_thresh": 0.20,
    "crop_n_layers": 2,
    "crop_nms_thresh": 0.70,
    "crop_overlap_ratio": 512 / 1500,
    "min_mask_region_area": 10,
    "use_m2m": False,
    "multimask_output": True,
}

# Image pre-processing defaults
DEFAULT_PREPROCESS_PARAMS: Dict[str, Any] = {
    "clahe_clip_limit": 3.0,
    "clahe_grid_size": (8, 8),
    "gaussian_blur_kernel": (101, 101),
    "invert": False,
    "normalize": True,
}

# Two-stage area filtering thresholds
DEFAULT_FILTER_PARAMS: Dict[str, float] = {
    "stage1_std_multiplier": 3.0,
    "stage2_std_multiplier": 2.0,
}

# Presets calibrated for different microscopy modalities & cell densities
AMG_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "points_per_side": 32,
        "points_per_batch": 128,
        "pred_iou_thresh": 0.85,
        "stability_score_thresh": 0.35,
        "crop_n_layers": 1,
        "crop_nms_thresh": 0.70,
        "crop_overlap_ratio": 512 / 1500,
        "min_mask_region_area": 15,
        "use_m2m": False,
        "multimask_output": True,
    },
    "balanced": dict(DEFAULT_AMG_PARAMS),  # Oxford thesis calibrated default
    "sensitive": {
        "points_per_side": 32,
        "points_per_batch": 128,
        "pred_iou_thresh": 0.65,
        "stability_score_thresh": 0.12,
        "crop_n_layers": 2,
        "crop_nms_thresh": 0.70,
        "crop_overlap_ratio": 512 / 1500,
        "min_mask_region_area": 8,
        "use_m2m": False,
        "multimask_output": True,
    },
}

# Parameter search space from thesis amg_tuning.py
AMG_PARAM_SPACE: Dict[str, Any] = {
    "pred_iou_thresh": (0.50, 0.90),
    "stability_score_thresh": (0.10, 0.50),
    "crop_n_layers": [0, 1, 2],
    "crop_nms_thresh": (0.50, 0.90),
    "crop_overlap_ratio": (0.2, 0.5),
    "min_mask_region_area": (5, 25),
    "use_m2m": [False],
}

