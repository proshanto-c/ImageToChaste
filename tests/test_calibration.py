"""
Unit tests for calibration and deployment pipelines.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from imagetochaste.calibration import (
    calibrate,
    deploy,
    load_image_array,
    sample_random_params,
)


def test_sample_random_params():
    params = sample_random_params(seed=123)
    assert 0.50 <= params["pred_iou_thresh"] <= 0.90
    assert 0.10 <= params["stability_score_thresh"] <= 0.50
    assert params["crop_n_layers"] in [0, 1, 2]
    assert params["points_per_side"] == 32


def test_load_image_array(tmp_path: Path):
    # Test 2D grayscale to 3D RGB
    gray = np.zeros((50, 50), dtype=np.uint8)
    rgb = load_image_array(gray)
    assert rgb.shape == (50, 50, 3)

    # Test file path loading
    from PIL import Image
    pil_img = Image.fromarray(rgb)
    img_path = tmp_path / "test.png"
    pil_img.save(img_path)

    loaded = load_image_array(img_path)
    assert loaded.shape == (50, 50, 3)


def test_calibrate_pipeline(tmp_path: Path):
    # Create synthetic test image
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock SAMAdapter
    mock_adapter = MagicMock()

    # Create dummy mask results
    def dummy_generate(image, generator=None, filter_area=True):
        # Return 15 synthetic masks
        masks = [
            np.zeros((100, 100), dtype=bool) for _ in range(15)
        ]
        # Set central pixels
        for i, m in enumerate(masks):
            m[10 + i * 4 : 12 + i * 4, 10 + i * 4 : 12 + i * 4] = True
        return masks, {"initial_count": 15}

    mock_adapter.generate_masks.side_effect = dummy_generate
    mock_adapter.get_mask_generator.return_value = MagicMock()

    out_dir = tmp_path / "calibration_out"

    result = calibrate(
        images=[test_img],
        expected_cell_count=14,
        candidate_profiles=["conservative", "balanced"],
        adapter=mock_adapter,
        output_dir=out_dir,
    )

    assert "best_profile" in result
    assert "calibrated_params" in result
    assert result["mean_detected_count"] == 15.0
    assert (out_dir / "calibration.json").exists()

    calib_json = json.loads((out_dir / "calibration.json").read_text())
    assert calib_json["best_profile"] in ["conservative", "balanced"]


def test_deploy_pipeline(tmp_path: Path):
    # Create 2 synthetic frame files
    from PIL import Image
    frame1 = tmp_path / "frame_001.png"
    frame2 = tmp_path / "frame_002.png"

    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(frame1)
    Image.fromarray(arr).save(frame2)

    # Mock adapter
    mock_adapter = MagicMock()

    def dummy_generate(image, generator=None, filter_area=True):
        masks = []
        for x, y in [(20, 20), (20, 80), (80, 20), (80, 80), (50, 50)]:
            m = np.zeros((100, 100), dtype=bool)
            m[y - 2 : y + 2, x - 2 : x + 2] = True
            masks.append(m)
        return masks, {"initial_count": 5}

    mock_adapter.generate_masks.side_effect = dummy_generate
    mock_adapter.get_mask_generator.return_value = MagicMock()

    calib_config = {
        "best_profile": "balanced",
        "calibrated_params": {"stability_score_thresh": 0.20},
    }

    out_dir = tmp_path / "deployment_out"

    res = deploy(
        images=[frame1, frame2],
        calibration=calib_config,
        output_dir=out_dir,
        mode="voronoi",
        adapter=mock_adapter,
    )

    assert res["total_frames_processed"] == 2
    assert res["total_cells_detected"] == 10
    assert (out_dir / "deployment_summary.json").exists()

    # Verify frame 1 output files
    f1_dir = out_dir / "frame_001"
    assert f1_dir.exists()
    assert (f1_dir / "frame_001.node").exists()
    assert (f1_dir / "frame_001.cell").exists()
    assert (f1_dir / "frame_001.nodes").exists()
    assert (f1_dir / "frame_001.elements").exists()
    assert (f1_dir / "frame_001_overlay.png").exists()


def test_calibrate_legacy_adapter_signature(tmp_path: Path):
    """Verify calibrate seamlessly handles legacy adapters where generate_masks lacks generator."""
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_adapter = MagicMock()

    def legacy_generate(image, filter_area=True, preprocess=False, amg_params=None):
        # Strict signature: raises TypeError if called with generator=...
        masks = [np.zeros((100, 100), dtype=bool) for _ in range(10)]
        for i, m in enumerate(masks):
            m[10 + i * 4 : 12 + i * 4, 10 + i * 4 : 12 + i * 4] = True
        return masks, {"initial_count": 10}

    mock_adapter.generate_masks.side_effect = legacy_generate
    mock_adapter.get_mask_generator.return_value = MagicMock()

    out_dir = tmp_path / "legacy_calib"
    result = calibrate(
        images=[test_img],
        expected_cell_count=10,
        candidate_profiles=["balanced"],
        adapter=mock_adapter,
        output_dir=out_dir,
    )
    assert result["mean_detected_count"] == 10.0
    assert result["percent_error"] == 0.0

