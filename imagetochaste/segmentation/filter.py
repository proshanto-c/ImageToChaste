"""
Statistical mask filtering based on cell area distributions.
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np


def filter_masks_by_area(
    masks: List[Union[Dict[str, Any], np.ndarray]],
    stage1_std_multiplier: float = 3.0,
    stage2_std_multiplier: float = 2.0,
) -> Tuple[List[Union[Dict[str, Any], np.ndarray]], Dict[str, float]]:
    """
    Two-stage statistical outlier rejection on segmented cell areas.

    Removes over-segmented artifacts (e.g. background patches) and under-segmented debris.

    Stage 1: Removes masks larger than mean + (stage1_std_multiplier * std).
    Stage 2: Recomputes statistics and retains masks within mean +/- (stage2_std_multiplier * std).

    Args:
        masks: List of SAM 2 mask dictionaries (each with 'segmentation') or 2D binary numpy arrays.
        stage1_std_multiplier: Upper standard deviation threshold for stage 1.
        stage2_std_multiplier: Standard deviation threshold range for stage 2.

    Returns:
        Tuple of (filtered_masks, statistics_dict)
    """
    if not masks:
        return [], {"initial_count": 0, "final_count": 0, "mean_area": 0.0, "std_area": 0.0}

    # Extract area for each mask
    areas = []
    for m in masks:
        if isinstance(m, dict):
            if "area" in m:
                areas.append(float(m["area"]))
            else:
                areas.append(float(np.sum(m["segmentation"] > 0)))
        else:
            areas.append(float(np.sum(m > 0)))

    areas_arr = np.array(areas, dtype=np.float64)
    initial_count = len(masks)

    if initial_count < 3:
        # Not enough samples for statistical filtering
        return masks, {
            "initial_count": initial_count,
            "final_count": initial_count,
            "mean_area": float(np.mean(areas_arr)),
            "std_area": float(np.std(areas_arr)),
        }

    # Stage 1: Remove extreme large outliers
    mean1 = float(np.mean(areas_arr))
    std1 = float(np.std(areas_arr))
    stage1_cutoff = mean1 + stage1_std_multiplier * std1

    survivors_stage1 = [
        (m, a) for m, a in zip(masks, areas_arr) if a <= stage1_cutoff
    ]

    if not survivors_stage1:
        return masks, {
            "initial_count": initial_count,
            "final_count": initial_count,
            "mean_area": mean1,
            "std_area": std1,
        }

    # Stage 2: Recompute mean and std on trimmed population
    stage1_areas = np.array([a for _, a in survivors_stage1], dtype=np.float64)
    mean2 = float(np.mean(stage1_areas))
    std2 = float(np.std(stage1_areas))

    lower_bound = max(1.0, mean2 - stage2_std_multiplier * std2)
    upper_bound = mean2 + stage2_std_multiplier * std2

    filtered_masks = [
        m for m, a in survivors_stage1 if lower_bound <= a <= upper_bound
    ]

    stats = {
        "initial_count": initial_count,
        "final_count": len(filtered_masks),
        "mean_area": mean2,
        "std_area": std2,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }

    return filtered_masks, stats
