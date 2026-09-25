"""
Command Line Interface for ImageToChaste.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from imagetochaste import __version__
from imagetochaste.download_weights import download_checkpoint
from imagetochaste.exporters.chaste_formatter import (
    export_centroids_json,
    export_chaste_nodes,
    export_chaste_vertex_mesh,
)
from imagetochaste.geometry.centroids import compute_centroids_from_masks
from imagetochaste.geometry.mesh_builder import build_voronoi_mesh
from imagetochaste.segmentation.preprocessor import load_image
from imagetochaste.segmentation.sam_adapter import SAMAdapter
from imagetochaste.segmentation.utils import save_mask_visualization


def run_pipeline(
    input_image: str,
    output_path: str,
    checkpoint: str = "checkpoints/sam2_hiera_large.pt",
    model_cfg: Optional[str] = None,
    device: str = "auto",
    mode: str = "centroids",
    preprocess: bool = False,
    filter_area: bool = True,
    visualize_path: Optional[str] = None,
):
    """
    Execute end-to-end ImageToChaste pipeline.
    """
    image_file = Path(input_image)
    if not image_file.exists():
        raise FileNotFoundError(f"Input image not found: {image_file}")

    print(f"=== ImageToChaste v{__version__} ===")
    print(f"Input image: {image_file}")
    print(f"Meshing mode: {mode}")
    print(f"Pre-processing: {preprocess}")

    # Load adapter and run segmentation
    adapter = SAMAdapter(checkpoint=checkpoint, model_cfg=model_cfg, device=device)
    raw_img = load_image(image_file)

    print("Running SAM 2 cell segmentation...")
    masks, stats = adapter.generate_masks(
        raw_img, filter_area=filter_area, preprocess=preprocess
    )
    print(f"Detected {len(masks)} cells (initial: {stats.get('initial_count', len(masks))}).")

    if not masks:
        print("Warning: No cell masks detected! Check pre-processing or thresholds.")
        return

    # Extract centroids
    centroids = compute_centroids_from_masks(masks)
    print(f"Computed {len(centroids)} cell centroids.")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if mode == "centroids":
        # Export Chaste NodesOnlyMesh (.nodes)
        nodes_file = export_chaste_nodes(centroids, out_p, header_style="simple")
        print(f"Exported Chaste NodesOnlyMesh: {nodes_file}")

        # Also write JSON companion for easy inspection
        json_file = out_p.with_suffix(".json")
        export_centroids_json(centroids, json_file)
        print(f"Exported centroid coordinates JSON: {json_file}")

    elif mode == "voronoi":
        # Export Chaste VertexMesh (.nodes and .elements)
        mesh = build_voronoi_mesh(centroids)
        n_file, e_file = export_chaste_vertex_mesh(mesh, out_p)
        print("Exported Chaste VertexMesh:")
        print(f"  Nodes:    {n_file} ({mesh.num_nodes} vertices)")
        print(f"  Elements: {e_file} ({mesh.num_elements} cells)")

    else:
        raise ValueError(f"Unknown meshing mode: {mode}. Choose 'centroids' or 'voronoi'.")

    # Optional visualization
    if visualize_path:
        vis_file = save_mask_visualization(raw_img, masks, visualize_path, title=f"Segmented Cells (n={len(masks)})")
        print(f"Saved segmentation visualization overlay: {vis_file}")

    print("Done! Pipeline completed successfully.")


def print_system_info():
    """
    Print diagnostics about Python environment, PyTorch, hardware, and SAM 2.
    """
    print(f"ImageToChaste Version: {__version__}")
    print(f"Python Version: {sys.version.split()[0]}")

    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Apple Silicon MPS Available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    try:
        from imagetochaste.segmentation.sam_adapter import import_sam2
        import_sam2()
        print("Meta SAM 2 Package: INSTALLED")
    except Exception as e:
        print(f"Meta SAM 2 Package: NOT INSTALLED ({e})")


def main():
    parser = argparse.ArgumentParser(
        prog="imagetochaste",
        description="Automated cell segmentation from microscopy scans to Chaste C++ simulation meshes.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run end-to-end segmentation and mesh export.")
    run_parser.add_argument("-i", "--input", required=True, help="Path to input microscopy scan (PNG/TIF/JPG).")
    run_parser.add_argument("-o", "--output", required=True, help="Destination path for Chaste mesh / nodes.")
    run_parser.add_argument("-c", "--checkpoint", default="checkpoints/sam2_hiera_large.pt", help="Path to SAM 2 weights (.pt).")
    run_parser.add_argument("--config", default=None, help="Path to SAM 2 model config YAML.")
    run_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Hardware device.")
    run_parser.add_argument("--mode", default="centroids", choices=["centroids", "voronoi"], help="Output format: centroids (NodesOnlyMesh) or voronoi (VertexMesh).")
    run_parser.add_argument("--preprocess", action="store_true", help="Apply CLAHE and illumination correction before segmentation.")
    run_parser.add_argument("--no-filter", action="store_true", help="Disable statistical area outlier filtering.")
    run_parser.add_argument("--visualize", default=None, help="Optional output path to save overlay image (e.g. out.png).")

    # 'calibrate' subcommand
    calib_parser = subparsers.add_parser("calibrate", help="Calibrate SAM 2 parameters using reference training frames and target cell count.")
    calib_parser.add_argument("-i", "--input", required=True, nargs="+", help="One or more reference training image paths.")
    calib_parser.add_argument("--expected-cells", type=int, default=None, help="Approximate expected cell count in the training image.")
    calib_parser.add_argument("--profiles", nargs="+", default=["conservative", "balanced", "sensitive"], help="Presets to evaluate.")
    calib_parser.add_argument("--random-runs", type=int, default=0, help="Number of random parameter exploration runs.")
    calib_parser.add_argument("-o", "--output-dir", default="outputs/calibration", help="Directory where calibration.json and previews will be saved.")
    calib_parser.add_argument("--checkpoint", default="checkpoints/sam2_hiera_large.pt", help="Path to SAM 2 weights.")
    calib_parser.add_argument("--device", default="auto", help="Compute device: auto, cuda, mps, cpu.")
    calib_parser.add_argument("--no-preprocess", action="store_true", help="Disable CLAHE preprocessing.")

    # 'deploy' subcommand
    deploy_parser = subparsers.add_parser("deploy", help="Deploy calibrated parameters across a sequence of timelapse microscopy frames.")
    deploy_parser.add_argument("-i", "--input", required=True, help="Directory containing timelapse images or image pattern.")
    deploy_parser.add_argument("-c", "--config", required=True, help="Path to calibration.json file from 'calibrate'.")
    deploy_parser.add_argument("-o", "--output-dir", default="outputs/deployed", help="Directory to save per-frame Chaste meshes.")
    deploy_parser.add_argument("--mode", default="voronoi", choices=["voronoi", "centroids"], help="Chaste mesh format: voronoi (VertexMesh) or centroids (NodesOnlyMesh).")
    deploy_parser.add_argument("--checkpoint", default="checkpoints/sam2_hiera_large.pt", help="Path to SAM 2 weights.")
    deploy_parser.add_argument("--device", default="auto", help="Compute device: auto, cuda, mps, cpu.")
    deploy_parser.add_argument("--no-preprocess", action="store_true", help="Disable CLAHE preprocessing.")

    # 'download-weights' subcommand
    dl_parser = subparsers.add_parser("download-weights", help="Download official Meta SAM 2 model checkpoints.")
    dl_parser.add_argument("--model", default="large", choices=["tiny", "small", "base_plus", "large", "sam2.1_large"], help="SAM 2 architecture variant.")
    dl_parser.add_argument("--output-dir", default="checkpoints", help="Directory where checkpoint is saved.")

    # 'export' subcommand
    export_parser = subparsers.add_parser("export", help="Convert existing centroid JSON to Chaste .nodes format.")
    export_parser.add_argument("-c", "--centroids", required=True, help="Path to JSON file with cell centroid list.")
    export_parser.add_argument("-o", "--output", required=True, help="Output .nodes file path.")
    export_parser.add_argument("--style", default="simple", choices=["simple", "extended"], help="Header style: simple or extended.")

    # 'info' subcommand
    subparsers.add_parser("info", help="Print system diagnostics and hardware acceleration info.")

    # Support shorthand: if --input is passed at root level, treat as run
    if len(sys.argv) > 1 and ("-i" in sys.argv or "--input" in sys.argv) and sys.argv[1] not in subparsers.choices:
        sys.argv.insert(1, "run")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(
            input_image=args.input,
            output_path=args.output,
            checkpoint=args.checkpoint,
            model_cfg=args.config,
            device=args.device,
            mode=args.mode,
            preprocess=args.preprocess,
            filter_area=not args.no_filter,
            visualize_path=args.visualize,
        )
    elif args.command == "calibrate":
        from imagetochaste.calibration import calibrate
        res = calibrate(
            images=args.input,
            expected_cell_count=args.expected_cells,
            candidate_profiles=args.profiles,
            num_random_candidates=args.random_runs,
            checkpoint=args.checkpoint,
            device=args.device,
            preprocess=not args.no_preprocess,
            output_dir=args.output_dir,
        )
        print(f"\n✅ Calibration complete! Best profile selected: '{res['best_profile']}'")
        if res.get("expected_cell_count"):
            print(f"Target count: {res['expected_cell_count']} | Detected count: {res['mean_detected_count']} (Error: {res['percent_error']}%)")
        print(f"Saved calibration profile to: {res.get('calibration_file', args.output_dir)}")
    elif args.command == "deploy":
        from imagetochaste.calibration import deploy
        res = deploy(
            images=args.input,
            calibration=args.config,
            output_dir=args.output_dir,
            mode=args.mode,
            checkpoint=args.checkpoint,
            device=args.device,
            preprocess=not args.no_preprocess,
        )
        print(f"\n✅ Batch deployment complete! Processed {res['total_frames_processed']} frames in {res['total_batch_time_seconds']}s.")
        print(f"Total cells segmented: {res['total_cells_detected']} (avg {res['average_cells_per_frame']} cells/frame).")
        print(f"Summary report written to: {res.get('summary_file')}")
    elif args.command == "download-weights":
        download_checkpoint(model_type=args.model, output_dir=args.output_dir)
    elif args.command == "export":
        with open(args.centroids, "r", encoding="utf-8") as f:
            data = json.load(f)
        coords = [(item["x"], item["y"]) for item in data]
        out_f = export_chaste_nodes(coords, args.output, header_style=args.style)
        print(f"Exported {len(coords)} nodes to: {out_f}")
    elif args.command == "info":
        print_system_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
