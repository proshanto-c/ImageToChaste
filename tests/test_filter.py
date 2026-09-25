"""
Unit tests for two-stage statistical area filtering.
"""

import numpy as np

from imagetochaste.segmentation.filter import filter_masks_by_area


def test_filter_masks_by_area_outlier_rejection():
    # 15 typical cells of area ~100
    typical_masks = [
        {"segmentation": np.ones((10, 10), dtype=np.uint8), "area": 100 + i}
        for i in range(15)
    ]
    # 1 extreme large artifact
    large_outlier = {
        "segmentation": np.ones((200, 200), dtype=np.uint8), "area": 40000
    }
    # 1 extreme small artifact
    small_outlier = {
        "segmentation": np.ones((1, 1), dtype=np.uint8), "area": 1
    }

    all_masks = typical_masks + [large_outlier, small_outlier]
    filtered, stats = filter_masks_by_area(all_masks)

    # Large outlier should definitely be eliminated
    assert len(filtered) < len(all_masks)
    assert not any(m["area"] == 40000 for m in filtered)
    assert stats["final_count"] == len(filtered)


def test_filter_masks_empty():
    filtered, stats = filter_masks_by_area([])
    assert filtered == []
    assert stats["final_count"] == 0
