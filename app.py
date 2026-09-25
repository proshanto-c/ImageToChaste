"""
Gradio web application for ImageToChaste.
Runs seamlessly on CPU Basic (Free Tier) or Hugging Face Spaces with ZeroGPU.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gradio as gr
import numpy as np
import torch
from PIL import Image

from imagetochaste import (
    SAMAdapter,
    build_voronoi_mesh,
    export_chaste_nodes,
    export_chaste_vertex_mesh,
    preprocess_microscopy_image,
)

try:
    from imagetochaste import compute_centroids_from_masks
except ImportError:
    from imagetochaste.geometry.centroids import compute_centroids_from_masks

from imagetochaste.download_weights import download_checkpoint
from imagetochaste.segmentation.utils import create_mask_overlay

# Check hardware environment: automatically handle CPU Basic vs ZeroGPU
IS_CPU = not torch.cuda.is_available()

if not IS_CPU:
    try:
        import spaces
    except ImportError:
        class spaces:
            @staticmethod
            def GPU(func=None, **kwargs):
                if func is not None and callable(func):
                    return func
                return lambda f: f
else:
    # On CPU Basic, make spaces.GPU a strict no-op so no ZeroGPU quota is requested
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None and callable(func):
                return func
            return lambda f: f

# Global adapter instance
_ADAPTER = None


def get_adapter():
    global _ADAPTER
    if _ADAPTER is None:
        checkpoint = download_checkpoint(model_type="tiny", output_dir="checkpoints")
        dev = "cpu" if IS_CPU else "auto"
        _ADAPTER = SAMAdapter(checkpoint=checkpoint, device=dev)
    return _ADAPTER


def _extract_file_path(f: Any) -> Optional[str]:
    """Safely extract local file path from strings, Path, Gradio FileData, or TemporaryFileWrapper."""
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, Path):
        return str(f)
    if hasattr(f, "path") and isinstance(f.path, str):
        return f.path
    if hasattr(f, "name") and isinstance(f.name, str) and not isinstance(f, Path):
        return f.name
    if isinstance(f, dict):
        return f.get("path") or f.get("name")
    return str(f)


def load_image_array(image_input: Any) -> np.ndarray:
    """
    Standardize various image input formats to a uint8 RGB numpy array.
    Supports numpy arrays, PIL Images, paths, and Gradio Image/Editor dicts.
    """
    if image_input is None:
        raise ValueError("Image input is None.")
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return np.stack([image_input] * 3, axis=-1).astype(np.uint8)
        elif image_input.ndim == 3 and image_input.shape[2] == 4:
            return image_input[:, :, :3].astype(np.uint8)
        elif image_input.ndim == 3 and image_input.shape[2] == 1:
            return np.concatenate([image_input] * 3, axis=-1).astype(np.uint8)
        return image_input.astype(np.uint8)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    elif isinstance(image_input, dict):
        for k in ("composite", "image", "background", "path"):
            if k in image_input and image_input[k] is not None:
                return load_image_array(image_input[k])
        raise ValueError(f"Unrecognized image dictionary keys: {list(image_input.keys())}")
    elif hasattr(image_input, "path") and isinstance(image_input.path, str):
        return load_image_array(image_input.path)
    elif hasattr(image_input, "name") and isinstance(image_input.name, str) and not isinstance(image_input, Path):
        return load_image_array(image_input.name)
    else:
        path = Path(str(image_input))
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)


def safe_generate_masks(
    adapter: SAMAdapter,
    image_arr: np.ndarray,
    generator: Any = None,
    cand_params: Optional[Dict[str, Any]] = None,
):
    """
    Execute mask generation safely across different SAMAdapter versions.
    If generator keyword argument is rejected by older adapter, falls back to amg_params.
    """
    if generator is not None:
        try:
            return adapter.generate_masks(image_arr, generator=generator, filter_area=True)
        except TypeError:
            pass
    return adapter.generate_masks(image_arr, amg_params=cand_params, filter_area=True)


# AMG presets calibrated for microscopy (automatically balanced for CPU vs GPU)
POINTS_PER_SIDE = 24 if IS_CPU else 32
POINTS_PER_BATCH = 64 if IS_CPU else 128
CROP_LAYERS = 1 if IS_CPU else 2

AMG_PRESETS = {
    "conservative": {
        "points_per_side": POINTS_PER_SIDE,
        "points_per_batch": POINTS_PER_BATCH,
        "pred_iou_thresh": 0.85,
        "stability_score_thresh": 0.35,
        "crop_n_layers": 0 if IS_CPU else 1,
        "crop_nms_thresh": 0.70,
        "crop_overlap_ratio": 512 / 1500,
        "min_mask_region_area": 15,
        "use_m2m": False,
        "multimask_output": True,
    },
    "balanced": {
        "points_per_side": POINTS_PER_SIDE,
        "points_per_batch": POINTS_PER_BATCH,
        "pred_iou_thresh": 0.75,
        "stability_score_thresh": 0.20,
        "crop_n_layers": CROP_LAYERS,
        "crop_nms_thresh": 0.70,
        "crop_overlap_ratio": 512 / 1500,
        "min_mask_region_area": 10,
        "use_m2m": False,
        "multimask_output": True,
    },
    "sensitive": {
        "points_per_side": POINTS_PER_SIDE,
        "points_per_batch": POINTS_PER_BATCH,
        "pred_iou_thresh": 0.65,
        "stability_score_thresh": 0.12,
        "crop_n_layers": CROP_LAYERS,
        "crop_nms_thresh": 0.70,
        "crop_overlap_ratio": 512 / 1500,
        "min_mask_region_area": 8,
        "use_m2m": False,
        "multimask_output": True,
    },
}


def run_calibrate(
    images: Union[str, Path, np.ndarray, Image.Image, List[Any]],
    expected_cell_count: Optional[int] = None,
    candidate_profiles: Optional[List[str]] = None,
    adapter: Optional[SAMAdapter] = None,
    preprocess: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute calibration pipeline with safe fallback."""
    if not isinstance(images, list):
        image_list = [images]
    else:
        image_list = images

    if not image_list:
        raise ValueError("At least one image must be provided.")

    loaded_images = [load_image_array(img) for img in image_list]
    working_images = [
        preprocess_microscopy_image(img, clahe_clip_limit=3.0) if preprocess else img
        for img in loaded_images
    ]

    if adapter is None:
        adapter = get_adapter()

    if candidate_profiles is None:
        candidate_profiles = ["conservative", "balanced", "sensitive"]

    candidates = [
        {"name": name, "params": dict(AMG_PRESETS[name])}
        for name in candidate_profiles
        if name in AMG_PRESETS
    ]

    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    results = []
    for cand in candidates:
        cand_name = cand["name"]
        cand_params = cand["params"]
        generator = adapter.get_mask_generator(**cand_params)
        per_image_counts = []

        for idx, (raw_img, work_img) in enumerate(zip(loaded_images, working_images)):
            masks, stats = safe_generate_masks(
                adapter, work_img, generator=generator, cand_params=cand_params
            )
            count = len(masks)
            per_image_counts.append(count)

            overlay = create_mask_overlay(raw_img, masks, alpha=0.5, draw_borders=True)
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

    if expected_cell_count is not None and expected_cell_count > 0:
        sorted_results = sorted(results, key=lambda x: x["percent_error"])
    else:
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


def run_deploy(
    images: List[Path],
    calibration: Union[str, Path, Dict[str, Any]],
    output_dir: Union[str, Path] = "outputs/deployed",
    mode: str = "voronoi",
    adapter: Optional[SAMAdapter] = None,
    preprocess: bool = True,
) -> Dict[str, Any]:
    """Execute deploy pipeline with safe fallback."""
    if isinstance(calibration, (str, Path)):
        calib_path = Path(calibration)
        if calib_path.exists():
            with open(calib_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
        else:
            calib_data = {}
    elif isinstance(calibration, dict):
        calib_data = calibration
    else:
        calib_data = {}

    calibrated_params = calib_data.get("calibrated_params", AMG_PRESETS["balanced"])

    if adapter is None:
        adapter = get_adapter()

    generator = adapter.get_mask_generator(**calibrated_params)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_results = []
    total_cells = 0
    t0_batch = time.time()

    for idx, img_path in enumerate(images):
        t0_frame = time.time()
        raw_img = load_image_array(img_path)
        h, w, _ = raw_img.shape

        working_img = (
            preprocess_microscopy_image(raw_img, clahe_clip_limit=3.0) if preprocess else raw_img
        )
        masks, stats = safe_generate_masks(
            adapter, working_img, generator=generator, cand_params=calibrated_params
        )
        cell_count = len(masks)
        total_cells += cell_count

        centroids = compute_centroids_from_masks(masks)
        stem = img_path.stem
        frame_out_dir = out_dir / stem
        frame_out_dir.mkdir(parents=True, exist_ok=True)

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

        overlay = create_mask_overlay(raw_img, masks, alpha=0.5, draw_borders=True)
        overlay_file = frame_out_dir / f"{stem}_overlay.png"
        Image.fromarray(overlay).save(overlay_file)
        exported_files["overlay"] = str(overlay_file)

        frame_results.append({
            "frame_index": idx,
            "filename": img_path.name,
            "detected_cells": cell_count,
            "nodes_count": num_nodes,
            "elements_count": num_elements,
            "processing_seconds": round(time.time() - t0_frame, 2),
            "files": exported_files,
        })

    summary_payload = {
        "total_frames_processed": len(images),
        "total_cells_detected": total_cells,
        "average_cells_per_frame": round(total_cells / max(1, len(images)), 1),
        "total_batch_time_seconds": round(time.time() - t0_batch, 2),
        "calibrated_profile_used": calib_data.get("best_profile", "custom"),
        "mode": mode,
        "frames": frame_results,
    }

    summary_file = out_dir / "deployment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    summary_payload["summary_file"] = str(summary_file)
    return summary_payload


@spaces.GPU(duration=120)
def process_microscopy_scan(image: Any, preprocess: bool, mode: str):
    if image is None:
        return None, None, None, "Please upload or select a microscopy image."

    img_arr = load_image_array(image)
    h, w, _ = img_arr.shape

    # 1. Pre-process image if requested
    if preprocess:
        working_img = preprocess_microscopy_image(img_arr, clahe_clip_limit=3.0)
    else:
        working_img = img_arr

    # 2. Segment using SAM 2
    adapter = get_adapter()
    masks, stats = adapter.generate_masks(
        working_img,
        filter_area=True,
        amg_params=AMG_PRESETS["balanced"] if IS_CPU else None,
    )

    if not masks:
        return (
            None,
            None,
            None,
            "No cell masks detected. Try enabling pre-processing or adjusting thresholds.",
        )

    # 3. Create visual overlay
    overlay = create_mask_overlay(img_arr, masks, alpha=0.5, draw_borders=True)

    # 4. Geometric extraction
    centroids = compute_centroids_from_masks(masks)
    mesh = build_voronoi_mesh(centroids, bounding_box=(0, 0, w, h))

    # 5. Export Chaste simulation files
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    nodes_file = export_chaste_nodes(centroids, out_dir / "chaste_cells.nodes")
    node_f, elem_f = export_chaste_vertex_mesh(mesh, out_dir / "chaste_mesh")

    nodes_preview = "\n".join(nodes_file.read_text().splitlines()[:15])
    summary_text = (
        f"✅ Detected {len(masks)} cells (initial: {stats.get('initial_count', len(masks))}) | "
        f"Generated {mesh.num_nodes} junction nodes & {mesh.num_elements} polygonal elements."
    )

    if mode == "NodesOnlyMesh (Centroids)":
        return overlay, str(nodes_file), None, f"{summary_text}\n\nNodes preview:\n{nodes_preview}"
    else:
        return overlay, str(node_f), str(elem_f), f"{summary_text}\n\nNodes preview:\n{nodes_preview}"


@spaces.GPU(duration=120)
def run_calibration_ui(image: Any, expected_cells: Any, preprocess: bool):
    if image is None:
        return None, None, None, "Please upload a reference training frame.", None

    expected_cell_count = None
    if expected_cells is not None:
        try:
            val = int(float(expected_cells))
            if val > 0:
                expected_cell_count = val
        except (ValueError, TypeError):
            expected_cell_count = None

    adapter = get_adapter()
    out_dir = Path("outputs/calibration_ui")
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = run_calibrate(
        images=image,
        expected_cell_count=expected_cell_count,
        candidate_profiles=["conservative", "balanced", "sensitive"],
        adapter=adapter,
        preprocess=preprocess,
        output_dir=out_dir,
    )

    f_cons = out_dir / "calibration_conservative_img0.png"
    f_bal = out_dir / "calibration_balanced_img0.png"
    f_sens = out_dir / "calibration_sensitive_img0.png"

    img_cons = Image.open(f_cons) if f_cons.exists() else None
    img_bal = Image.open(f_bal) if f_bal.exists() else None
    img_sens = Image.open(f_sens) if f_sens.exists() else None

    target_str = str(expected_cell_count) if expected_cell_count is not None else "N/A"
    lines = [
        f"### 🎯 Calibration Report: Best Match = **{calib['best_profile'].upper()}**",
        "",
        "| Profile | Detected Cells | Target Count | Error % |",
        "|---|---|---|---|",
    ]
    for c in calib["all_candidates"]:
        lines.append(
            f"| **{c['name'].capitalize()}** | {int(c['mean_detected_count'])} | {target_str} | {c['percent_error']}% |"
        )

    lines.append(
        "\n*Download `calibration.json` below to deploy these parameters across batch timelapses.*"
    )
    calib_json = (
        str(out_dir / "calibration.json") if (out_dir / "calibration.json").exists() else None
    )

    return img_cons, img_bal, img_sens, "\n".join(lines), calib_json


@spaces.GPU(duration=120)
def run_deployment_ui(
    files: Any,
    calib_file: Any,
    mode: str,
    preprocess: bool,
):
    if not files:
        return None, "Please upload one or more timelapse frames."

    if not isinstance(files, (list, tuple)):
        files = [files]

    adapter = get_adapter()
    deploy_in = Path("outputs/deploy_ui_input")
    deploy_out = Path("outputs/deploy_ui_output")

    if deploy_in.exists():
        shutil.rmtree(deploy_in)
    if deploy_out.exists():
        shutil.rmtree(deploy_out)

    deploy_in.mkdir(parents=True, exist_ok=True)
    deploy_out.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for f in files:
        f_path = _extract_file_path(f)
        if f_path and Path(f_path).exists():
            dest = deploy_in / Path(f_path).name
            shutil.copy(f_path, dest)
            img_paths.append(dest)

    if not img_paths:
        return None, "No valid image files found in upload."

    calib_path = _extract_file_path(calib_file)
    config_source = (
        calib_path if (calib_path and Path(calib_path).exists()) else {"calibrated_params": {}}
    )

    mesh_mode = "centroids" if "Centroids" in mode else "voronoi"
    summary = run_deploy(
        images=img_paths,
        calibration=config_source,
        output_dir=deploy_out,
        mode=mesh_mode,
        adapter=adapter,
        preprocess=preprocess,
    )

    # Create zip bundle for download
    zip_path = shutil.make_archive("outputs/chaste_timelapse_meshes", "zip", deploy_out)

    report_lines = [
        "### 🚀 Batch Deployment Finished!",
        f"- **Frames Processed:** {summary['total_frames_processed']}",
        f"- **Total Cells Detected:** {summary['total_cells_detected']} (Avg: {summary['average_cells_per_frame']} cells/frame)",
        f"- **Total Execution Time:** {summary['total_batch_time_seconds']}s",
        "",
        "| Frame | Detected Cells | Nodes | Elements | Processing Time |",
        "|---|---|---|---|---|",
    ]
    for fr in summary["frames"]:
        report_lines.append(
            f"| {fr['filename']} | {fr['detected_cells']} | {fr['nodes_count']} | {fr['elements_count']} | {fr['processing_seconds']}s |"
        )

    return str(zip_path), "\n".join(report_lines)


# Build Multi-Tab Gradio Application
with gr.Blocks(title="ImageToChaste: Microscopy to Chaste C++ Meshes") as demo:
    gr.Markdown("# 🔬 ImageToChaste")
    mode_badge = (
        "🖥️ **Running on CPU Basic (Free Tier)** — SAM 2 'tiny' with CPU-balanced parameters."
        if IS_CPU
        else "⚡ **Running on ZeroGPU** — Accelerated with NVIDIA GPU."
    )
    gr.Markdown(
        f"Automated Cell Segmentation from Microscopy Scans via Meta's SAM 2 into Oxford Chaste C++ Simulation Meshes.\n\n{mode_badge}"
    )

    with gr.Tabs():
        # --- TAB 1: Single Frame Pipeline ---
        with gr.Tab("🖼️ Single Frame Pipeline"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Upload Microscopy Scan")
                    preprocess_chk = gr.Checkbox(
                        value=True, label="Apply CLAHE & Illumination Flattening"
                    )
                    mode_radio = gr.Radio(
                        ["NodesOnlyMesh (Centroids)", "VertexMesh (Polygons)"],
                        value="VertexMesh (Polygons)",
                        label="Chaste Simulation Format",
                    )
                    submit_btn = gr.Button("Generate Simulation Mesh", variant="primary")

                    example_f1 = "data/sample/drosophila_germband_f009.png"
                    example_f2 = "data/sample/sample_cells.png"
                    examples = []
                    if Path(example_f1).exists():
                        examples.append([example_f1, True, "VertexMesh (Polygons)"])
                    if Path(example_f2).exists():
                        examples.append([example_f2, True, "NodesOnlyMesh (Centroids)"])
                    if examples:
                        gr.Examples(
                            examples=examples, inputs=[input_img, preprocess_chk, mode_radio]
                        )

                with gr.Column():
                    output_img = gr.Image(type="numpy", label="Segmented Cell Overlay")
                    info_box = gr.Textbox(label="Status & Preview", lines=6)
                    download_nodes = gr.File(label="Download Chaste .node / .nodes File")
                    download_elements = gr.File(label="Download Chaste .cell / .elements File")

            submit_btn.click(
                fn=process_microscopy_scan,
                inputs=[input_img, preprocess_chk, mode_radio],
                outputs=[output_img, download_nodes, download_elements, info_box],
            )

        # --- TAB 2: Calibration Pipeline ---
        with gr.Tab("🎯 Calibration (Reference Frame Tuning)"):
            gr.Markdown(
                "Upload 1 representative training frame and enter your approximate expected cell count. "
                "ImageToChaste evaluates candidate profiles (Conservative, Balanced, Sensitive) and outputs the best configuration."
            )
            with gr.Row():
                with gr.Column():
                    calib_input = gr.Image(type="pil", label="Reference Training Frame")
                    expected_cells_input = gr.Number(
                        value=300, label="Approximate Target Cell Count"
                    )
                    calib_prep_chk = gr.Checkbox(value=True, label="Apply CLAHE Pre-processing")
                    calib_btn = gr.Button("Run Calibration Sweep", variant="primary")

                with gr.Column():
                    calib_summary = gr.Markdown()
                    download_calib_json = gr.File(label="Download calibration.json")

            with gr.Row():
                preview_cons = gr.Image(type="pil", label="1. Conservative Profile")
                preview_bal = gr.Image(type="pil", label="2. Balanced Profile (Thesis)")
                preview_sens = gr.Image(type="pil", label="3. Sensitive Profile")

            calib_btn.click(
                fn=run_calibration_ui,
                inputs=[calib_input, expected_cells_input, calib_prep_chk],
                outputs=[
                    preview_cons,
                    preview_bal,
                    preview_sens,
                    calib_summary,
                    download_calib_json,
                ],
            )

        # --- TAB 3: Batch Timelapse Deployment ---
        with gr.Tab("🚀 Batch Deployment (Timelapse Sequences)"):
            gr.Markdown(
                "Deploy calibrated SAM 2 parameters across full timelapse sequences to produce Chaste meshes for each movie frame."
            )
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(
                        file_count="multiple", label="Upload Timelapse Frames (PNG, TIF, JPG)"
                    )
                    batch_calib_file = gr.File(
                        label="Upload calibration.json (Optional: defaults to thesis parameters)"
                    )
                    batch_mode = gr.Radio(
                        ["VertexMesh (Polygons)", "NodesOnlyMesh (Centroids)"],
                        value="VertexMesh (Polygons)",
                        label="Chaste Mesh Mode",
                    )
                    batch_prep = gr.Checkbox(value=True, label="Apply CLAHE Pre-processing")
                    batch_btn = gr.Button("Deploy Across Batch", variant="primary")

                with gr.Column():
                    deploy_summary_md = gr.Markdown()
                    download_zip = gr.File(label="Download All Chaste Meshes (.zip)")

            batch_btn.click(
                fn=run_deployment_ui,
                inputs=[batch_files, batch_calib_file, batch_mode, batch_prep],
                outputs=[download_zip, deploy_summary_md],
            )


if __name__ == "__main__":
    demo.launch()
