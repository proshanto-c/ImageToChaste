"""
Calibration and deployment pipelines for ImageToChaste.

Allows researchers to calibrate SAM 2 segmentation parameters against reference
training frames and target cell counts, then deploy the locked parameters across
entire timelapse microscopy sequences with automated Chaste mesh generation.
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from imagetochaste.config import AMG_PARAM_SPACE, AMG_PRESETS, DEFAULT_AMG_PARAMS
from imagetochaste.exporters.chaste_formatter import export_chaste_nodes, export_chaste_vertex_mesh
from imagetochaste.geometry.centroids import compute_centroids_from_masks
from imagetochaste.geometry.mesh_builder import build_voronoi_mesh
from imagetochaste.segmentation.preprocessor import preprocess_microscopy_image
from imagetochaste.segmentation.sam_adapter import SAMAdapter
from imagetochaste.segmentation.utils import create_mask_overlay


def load_image_array(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Standardize various image input formats to a uint8 RGB numpy array.
    """
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return np.stack([image_input] * 3, axis=-1).astype(np.uint8)
        elif image_input.shape[2] == 4:
            return image_input[:, :, :3].astype(np.uint8)
        return image_input.astype(np.uint8)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"), dtype=np.uint8)
    else:
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)


def sample_random_params(seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Sample candidate hyperparameters from the calibrated thesis search space.
    """
    if seed is not None:
        random.seed(seed)

    base = dict(DEFAULT_AMG_PARAMS)
    base["pred_iou_thresh"] = round(random.uniform(*AMG_PARAM_SPACE["pred_iou_thresh"]), 2)
    base["stability_score_thresh"] = round(random.uniform(*AMG_PARAM_SPACE["stability_score_thresh"]), 2)
    base["crop_n_layers"] = random.choice(AMG_PARAM_SPACE["crop_n_layers"])
    base["crop_nms_thresh"] = round(random.uniform(*AMG_PARAM_SPACE["crop_nms_thresh"]), 2)
    base["crop_overlap_ratio"] = round(random.uniform(*AMG_PARAM_SPACE["crop_overlap_ratio"]), 2)
    base["min_mask_region_area"] = random.randint(*AMG_PARAM_SPACE["min_mask_region_area"])
    base["use_m2m"] = random.choice(AMG_PARAM_SPACE["use_m2m"])
    return base


def calibrate(
    images: Union[str, Path, np.ndarray, List[Union[str, Path, np.ndarray]]],
    expected_cell_count: Optional[int] = None,
    candidate_profiles: Optional[List[str]] = None,
    num_random_candidates: int = 0,
    adapter: Optional[SAMAdapter] = None,
    checkpoint: Optional[Union[str, Path]] = None,
    device: str = "auto",
    preprocess: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    auto_select: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Calibrate SAM 2 segmentation parameters on reference training image(s).

    Evaluates candidate parameter configurations (presets like conservative,
    balanced, sensitive, plus optional random sweeps), ranks them by difference
    from expected cell count, and produces diagnostic overlays.

    Args:
        images: Single image or list of training images.
        expected_cell_count: Optional approximate target cell count (e.g. 300).
        candidate_profiles: Names of AMG_PRESETS to evaluate (default: ['conservative', 'balanced', 'sensitive']).
        num_random_candidates: Number of randomized exploratory parameter sets to test.
        adapter: Optional existing SAMAdapter instance to reuse.
        checkpoint: Checkpoint path if creating new SAMAdapter.
        device: 'auto', 'cuda', 'mps', or 'cpu'.
        preprocess: Whether to apply CLAHE and background flattening.
        output_dir: Optional directory to save candidate overlays and calibration.json.
        auto_select: If True, automatically picks lowest error candidate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing best parameters, candidates evaluated, and performance metrics.
    """
    # 1. Standardize image list
    if not isinstance(images, list):
        image_list = [images]
    else:
        image_list = images

    if not image_list:
        raise ValueError("At least one image must be provided for calibration.")

    loaded_images = [load_image_array(img) for img in image_list]

    # Pre-process if requested
    if preprocess:
        working_images = [preprocess_microscopy_image(img, clahe_clip_limit=3.0) for img in loaded_images]
    else:
        working_images = loaded_images

    # 2. Build or obtain SAM adapter
    if adapter is None:
        if checkpoint is None:
            checkpoint = "checkpoints/sam2_hiera_tiny.pt"
            if not Path(checkpoint).exists():
                from imagetochaste.download_weights import download_checkpoint
                checkpoint = download_checkpoint(model_type="tiny", output_dir="checkpoints")
        adapter = SAMAdapter(checkpoint=checkpoint, device=device)

    # 3. Assemble candidate configurations
    if candidate_profiles is None:
        candidate_profiles = ["conservative", "balanced", "sensitive"]

    candidates: List[Dict[str, Any]] = []
    for name in candidate_profiles:
        if name in AMG_PRESETS:
            candidates.append({"name": name, "params": dict(AMG_PRESETS[name])})

    if num_random_candidates > 0:
        random.seed(seed)
        for i in range(num_random_candidates):
            candidates.append({
                "name": f"random_search_{i+1}",
                "params": sample_random_params(),
            })

    # Prepare output directory if requested
    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    # 4. Evaluate candidates across training images
    results = []
    for cand in candidates:
        cand_name = cand["name"]
        cand_params = cand["params"]

        generator = adapter.get_mask_generator(**cand_params)
        per_image_counts = []
        per_image_overlays = []

        for idx, (raw_img, work_img) in enumerate(zip(loaded_images, working_images)):
            masks, stats = adapter.generate_masks(work_img, generator=generator, filter_area=True)
            count = len(masks)
            per_image_counts.append(count)

            # Generate and optionally save overlay
            overlay = create_mask_overlay(raw_img, masks, alpha=0.5, draw_borders=True)
            per_image_overlays.append(overlay)

            if out_path:
                img_f = out_path / f"calibration_{cand_name}_img{idx}.png"
                Image.fromarray(overlay).save(img_f)

        mean_detected = float(np.mean(per_image_counts))

        if expected_cell_count is not None and expected_cell_count > 0:
            pct_error = float(abs(mean_detected - expected_cell_count) / expected_cell_count * 100.0)
        else:
            pct_error = 0.0

        results.append({
            "name": cand_name,
            "params": cand_params,
            "detected_counts": per_image_counts,
            "mean_detected_count": mean_detected,
            "expected_cell_count": expected_cell_count,
            "percent_error": round(pct_error, 2),
        })

    # 5. Rank and select best candidate
    if expected_cell_count is not None and expected_cell_count > 0:
        sorted_results = sorted(results, key=lambda x: x["percent_error"])
    else:
        # Default to balanced profile if no expected count given
        sorted_results = sorted(results, key=lambda x: 0 if x["name"] == "balanced" else 1)

    best = sorted_results[0]

    output_payload = {
        "best_profile": best["name"],
        "calibrated_params": best["params"],
        "expected_cell_count": expected_cell_count,
        "mean_detected_count": best["mean_detected_count"],
        "percent_error": best["percent_error"],
        "all_candidates": sorted_results,
    }

    if out_path:
        calib_file = out_path / "calibration.json"
        with open(calib_file, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        output_payload["calibration_file"] = str(calib_file)

    return output_payload


def deploy(
    images: Union[str, Path, List[Union[str, Path]]],
    calibration: Union[str, Path, Dict[str, Any]],
    output_dir: Union[str, Path] = "outputs/deployed",
    mode: str = "voronoi",
    adapter: Optional[SAMAdapter] = None,
    checkpoint: Optional[Union[str, Path]] = None,
    device: str = "auto",
    preprocess: bool = True,
    save_overlays: bool = True,
) -> Dict[str, Any]:
    """
    Deploy locked calibration parameters across a sequence of microscopy images.

    Executes full segmentation, two-stage morphological filtering, geometric extraction,
    and automatic Chaste simulation mesh export (.node, .cell, .nodes, .elements).

    Args:
        images: Directory path containing images, or list of image paths.
        calibration: Path to calibration.json or calibration dictionary from calibrate().
        output_dir: Destination folder for Chaste mesh outputs and summaries.
        mode: 'voronoi' (VertexMesh) or 'centroids' (NodesOnlyMesh).
        adapter: Optional pre-warmed SAMAdapter instance.
        checkpoint: Checkpoint path if initializing a new adapter.
        device: 'auto', 'cuda', 'mps', or 'cpu'.
        preprocess: Whether to apply CLAHE image pre-processing.
        save_overlays: Whether to render and save visualization overlay PNGs.

    Returns:
        Structured dictionary reporting per-frame results, statistics, and file locations.
    """
    # 1. Load calibration parameters
    if isinstance(calibration, (str, Path)):
        calib_path = Path(calibration)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
    else:
        calib_data = calibration

    calibrated_params = calib_data.get("calibrated_params", DEFAULT_AMG_PARAMS)

    # 2. Resolve image paths
    image_paths: List[Path] = []
    if isinstance(images, (str, Path)):
        path = Path(images)
        if path.is_dir():
            valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            image_paths = sorted([p for p in path.iterdir() if p.suffix.lower() in valid_exts])
        elif path.exists():
            image_paths = [path]
        else:
            raise FileNotFoundError(f"Images path not found: {path}")
    else:
        image_paths = [Path(p) for p in images]

    if not image_paths:
        raise ValueError(f"No microscopy images found in input: {images}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Build adapter and generator
    if adapter is None:
        if checkpoint is None:
            checkpoint = "checkpoints/sam2_hiera_tiny.pt"
            if not Path(checkpoint).exists():
                from imagetochaste.download_weights import download_checkpoint
                checkpoint = download_checkpoint(model_type="tiny", output_dir="checkpoints")
        adapter = SAMAdapter(checkpoint=checkpoint, device=device)

    generator = adapter.get_mask_generator(**calibrated_params)

    # 4. Deploy across image sequence
    frame_results = []
    total_cells = 0
    t0_batch = time.time()

    for idx, img_path in enumerate(image_paths):
        t0_frame = time.time()
        raw_img = load_image_array(img_path)
        h, w, _ = raw_img.shape

        if preprocess:
            working_img = preprocess_microscopy_image(raw_img, clahe_clip_limit=3.0)
        else:
            working_img = raw_img

        masks, stats = adapter.generate_masks(working_img, generator=generator, filter_area=True)
        cell_count = len(masks)
        total_cells += cell_count

        # Geometry extraction
        centroids = compute_centroids_from_masks(masks)
        stem = img_path.stem
        frame_out_dir = out_dir / stem
        frame_out_dir.mkdir(parents=True, exist_ok=True)

        # Export Chaste meshes
        exported_files = {}
        if mode == "centroids":
            nodes_f = export_chaste_nodes(centroids, frame_out_dir / f"{stem}.nodes")
            exported_files["nodes"] = str(nodes_f)
            num_nodes = len(centroids)
            num_elements = 0
        else:
            mesh = build_voronoi_mesh(centroids, bounding_box=(0, 0, w, h))
            nodes_f, elem_f = export_chaste_vertex_mesh(mesh, frame_out_dir / stem)
            exported_files["nodes"] = str(nodes_f)
            exported_files["elements"] = str(elem_f)
            exported_files["chaste_node_native"] = str(frame_out_dir / f"{stem}.node")
            exported_files["chaste_cell_native"] = str(frame_out_dir / f"{stem}.cell")
            num_nodes = mesh.num_nodes
            num_elements = mesh.num_elements

        # Save visual overlay
        if save_overlays:
            overlay = create_mask_overlay(raw_img, masks, alpha=0.5, draw_borders=True)
            overlay_file = frame_out_dir / f"{stem}_overlay.png"
            Image.fromarray(overlay).save(overlay_file)
            exported_files["overlay"] = str(overlay_file)

        elapsed = round(time.time() - t0_frame, 2)

        frame_results.append({
            "frame_index": idx,
            "filename": img_path.name,
            "detected_cells": cell_count,
            "nodes_count": num_nodes,
            "elements_count": num_elements,
            "processing_seconds": elapsed,
            "files": exported_files,
        })

    total_time = round(time.time() - t0_batch, 2)
    avg_cells = round(total_cells / len(image_paths), 1)

    summary_payload = {
        "total_frames_processed": len(image_paths),
        "total_cells_detected": total_cells,
        "average_cells_per_frame": avg_cells,
        "total_batch_time_seconds": total_time,
        "calibrated_profile_used": calib_data.get("best_profile", "custom"),
        "mode": mode,
        "frames": frame_results,
    }

    # Save summary report
    summary_file = out_dir / "deployment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    summary_payload["summary_file"] = str(summary_file)
    return summary_payload
