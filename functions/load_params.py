import os
import torch
import json
from sam.sam2.build_sam import build_sam2
from sam.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def load_generator_from_run(run_number: int, results_path: str = "outs/amg_tuning_test/results.json",
                            checkpoint: str = "sam/checkpoints/sam2.1_hiera_large.pt",
                            model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml") -> SAM2AutomaticMaskGenerator:
    """
    Load SAM2AutomaticMaskGenerator from a specific run number saved in results.json.

    Args:
        run_number (int): The run number to load parameters for.
        results_path (str): Path to the results JSON file.
        checkpoint (str): Path to the SAM2 checkpoint.
        model_cfg (str): Path to the model config YAML.

    Returns:
        SAM2AutomaticMaskGenerator: Configured generator ready to use.
    """
    # Load JSON
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Could not find results file at: {results_path}")
    with open(results_path, "r") as f:
        all_results = json.load(f)

    # Extract matching run
    run_data = next((r for r in all_results if r["run"] == run_number), None)
    if run_data is None:
        raise ValueError(f"Run number {run_number} not found in {results_path}.")

    params = run_data["params"]
    print(f"Loaded parameters for run {run_number}:\n{params}")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    # Build and return generator
    model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(model=model, **params)
