"""
Gradio web application for ImageToChaste.
Can be run locally or deployed directly to Hugging Face Spaces.
"""

from pathlib import Path

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
from imagetochaste.download_weights import download_checkpoint
from imagetochaste.segmentation.utils import create_mask_overlay

# Global adapter instance initialized lazily
_ADAPTER = None


def get_adapter():
    global _ADAPTER
    if _ADAPTER is None:
        checkpoint = download_checkpoint(model_type="tiny", output_dir="checkpoints")
        _ADAPTER = SAMAdapter(checkpoint=checkpoint, device="auto")
    return _ADAPTER


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


# Build Gradio UI
with gr.Blocks(title="ImageToChaste: Microscopy to Simulation Mesh") as demo:
    gr.Markdown("# 🔬 ImageToChaste")
    gr.Markdown(
        "Automated cell segmentation from microscopy scans via Meta's SAM 2 to C++ Chaste simulation meshes."
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Microscopy Scan (Brightfield / Fluorescence)")
            preprocess_chk = gr.Checkbox(value=True, label="Apply CLAHE & Illumination Flattening")
            mode_radio = gr.Radio(
                ["NodesOnlyMesh (Centroids)", "VertexMesh (Polygons)"],
                value="VertexMesh (Polygons)",
                label="Chaste Simulation Format",
            )
            submit_btn = gr.Button("Generate Simulation Mesh", variant="primary")

            # Sample gallery
            example_f1 = "data/sample/drosophila_germband_f009.png"
            example_f2 = "data/sample/sample_cells.png"
            examples = []
            if Path(example_f1).exists():
                examples.append([example_f1, True, "VertexMesh (Polygons)"])
            if Path(example_f2).exists():
                examples.append([example_f2, True, "NodesOnlyMesh (Centroids)"])

            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=[input_img, preprocess_chk, mode_radio],
                )

        with gr.Column():
            output_img = gr.Image(type="numpy", label="Segmented Cell Overlay")
            info_box = gr.Textbox(label="Status & Node Preview", lines=6)
            download_nodes = gr.File(label="Download Chaste .nodes File")
            download_elements = gr.File(label="Download Chaste .elements File")

    submit_btn.click(
        fn=process_microscopy_scan,
        inputs=[input_img, preprocess_chk, mode_radio],
        outputs=[output_img, download_nodes, download_elements, info_box],
    )


if __name__ == "__main__":
    demo.launch()
