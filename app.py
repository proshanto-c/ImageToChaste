"""
Gradio web application for ImageToChaste.
Can be run locally or deployed directly to Hugging Face Spaces with ZeroGPU.
"""

import shutil
from pathlib import Path
from typing import List, Optional

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None and callable(func):
                return func
            return lambda f: f

import gradio as gr
import numpy as np
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

try:
    from imagetochaste import calibrate, deploy
except ImportError:
    try:
        from imagetochaste.calibration import calibrate, deploy
    except ImportError:
        import subprocess
        import sys

        print("Updating imagetochaste in container...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--no-cache-dir",
                    "git+https://github.com/proshanto-c/ImageToChaste.git@v0.2.1",
                ],
                check=True,
            )
            from imagetochaste.calibration import calibrate, deploy
        except Exception as e:
            print(f"Auto-upgrade notice: {e}")
            calibrate = None
            deploy = None

from imagetochaste.download_weights import download_checkpoint
from imagetochaste.segmentation.utils import create_mask_overlay

# Global adapter instance
_ADAPTER = None


def get_adapter():
    global _ADAPTER
    if _ADAPTER is None:
        checkpoint = download_checkpoint(model_type="tiny", output_dir="checkpoints")
        _ADAPTER = SAMAdapter(checkpoint=checkpoint, device="auto")
    return _ADAPTER


@spaces.GPU(duration=120)
def process_microscopy_scan(image: Image.Image, preprocess: bool, mode: str):
    if image is None:
        return None, None, None, "Please upload or select a microscopy image."

    img_arr = np.array(image.convert("RGB"))
    h, w, _ = img_arr.shape

    # 1. Pre-process image if requested
    if preprocess:
        working_img = preprocess_microscopy_image(img_arr, clahe_clip_limit=3.0)
    else:
        working_img = img_arr

    # 2. Segment using SAM 2
    adapter = get_adapter()
    masks, stats = adapter.generate_masks(working_img, filter_area=True)

    if not masks:
        return None, None, None, "No cell masks detected. Try enabling pre-processing or adjusting thresholds."

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
def run_calibration_ui(image: Image.Image, expected_cells: Optional[int], preprocess: bool):
    if image is None:
        return None, None, None, "Please upload a reference training frame.", None

    if calibrate is None:
        return (
            None,
            None,
            None,
            "⚠️ The currently installed version of `imagetochaste` in this container is older than v0.2.0.\n"
            "Please click **Factory reboot** in your Space Settings (or update `requirements.txt`) to pull v0.2.0.",
            None,
        )

    adapter = get_adapter()
    out_dir = Path("outputs/calibration_ui")
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = calibrate(
        images=image,
        expected_cell_count=int(expected_cells) if expected_cells and expected_cells > 0 else None,
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

    lines = [
        f"### 🎯 Calibration Report: Best Match = **{calib['best_profile'].upper()}**",
        "",
        "| Profile | Detected Cells | Target Count | Error % |",
        "|---|---|---|---|",
    ]
    for c in calib["all_candidates"]:
        lines.append(
            f"| **{c['name'].capitalize()}** | {int(c['mean_detected_count'])} | {expected_cells or 'N/A'} | {c['percent_error']}% |"
        )

    lines.append("\n*Download `calibration.json` below to deploy these parameters across batch timelapses.*")
    calib_json = str(out_dir / "calibration.json")

    return img_cons, img_bal, img_sens, "\n".join(lines), calib_json


@spaces.GPU(duration=120)
def run_deployment_ui(
    files: List[gr.utils.NamedString],
    calib_file: Optional[gr.utils.NamedString],
    mode: str,
    preprocess: bool,
):
    if not files:
        return None, "Please upload one or more timelapse frames."

    if deploy is None:
        return (
            None,
            "⚠️ The currently installed version of `imagetochaste` in this container is older than v0.2.0.\n"
            "Please click **Factory reboot** in your Space Settings (or update `requirements.txt`) to pull v0.2.0.",
        )

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
        dest = deploy_in / Path(f.name).name
        shutil.copy(f.name, dest)
        img_paths.append(dest)

    config_source = calib_file.name if calib_file else {"calibrated_params": {}}

    mesh_mode = "centroids" if "Centroids" in mode else "voronoi"
    summary = deploy(
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
    gr.Markdown(
        "Automated Cell Segmentation from Microscopy Scans via Meta's SAM 2 into Oxford Chaste C++ Simulation Meshes."
    )

    with gr.Tabs():
        # --- TAB 1: Single Frame Pipeline ---
        with gr.Tab("🖼️ Single Frame Pipeline"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Upload Microscopy Scan")
                    preprocess_chk = gr.Checkbox(value=True, label="Apply CLAHE & Illumination Flattening")
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
                        gr.Examples(examples=examples, inputs=[input_img, preprocess_chk, mode_radio])

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
                    expected_cells_input = gr.Number(value=300, label="Approximate Target Cell Count")
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
                outputs=[preview_cons, preview_bal, preview_sens, calib_summary, download_calib_json],
            )

        # --- TAB 3: Batch Timelapse Deployment ---
        with gr.Tab("🚀 Batch Deployment (Timelapse Sequences)"):
            gr.Markdown(
                "Deploy calibrated SAM 2 parameters across full timelapse sequences to produce Chaste meshes for each movie frame."
            )
            with gr.Row():
                with gr.Column():
                    batch_files = gr.File(file_count="multiple", label="Upload Timelapse Frames (PNG, TIF, JPG)")
                    batch_calib_file = gr.File(label="Upload calibration.json (Optional: defaults to thesis parameters)")
                    batch_mode = gr.Radio(["VertexMesh (Polygons)", "NodesOnlyMesh (Centroids)"], value="VertexMesh (Polygons)", label="Chaste Mesh Mode")
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
