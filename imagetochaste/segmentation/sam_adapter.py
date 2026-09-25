"""
Adapter for Meta's Segment Anything Model 2 (SAM 2).

Provides decoupled loading, device management (CUDA / MPS / CPU),
and unified interfaces for automatic mask generation (AMG) and prompt-based inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from imagetochaste.config import DEFAULT_AMG_PARAMS, SAM2_CHECKPOINTS
from imagetochaste.segmentation.filter import filter_masks_by_area
from imagetochaste.segmentation.preprocessor import load_image, preprocess_microscopy_image


def resolve_device(requested_device: str = "auto"):
    """
    Select and configure PyTorch device with precision optimizations.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for SAM 2 segmentation. Install it via 'pip install torch torchvision'."
        ) from e

    req = requested_device.lower()
    if req == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(req)

    # Configure hardware accelerations
    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        pass  # MPS device selected

    return device


def import_sam2():
    """
    Attempt to import SAM 2 components from standard install or local path.
    """
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return build_sam2, SAM2AutomaticMaskGenerator, SAM2ImagePredictor
    except ImportError:
        pass

    try:
        from sam.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam.sam2.build_sam import build_sam2
        from sam.sam2.sam2_image_predictor import SAM2ImagePredictor
        return build_sam2, SAM2AutomaticMaskGenerator, SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "Segment Anything 2 (SAM 2) is not installed.\n"
            "Install it directly from Meta's repository:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git\n"
            "Or ensure the 'sam2' package is in your PYTHONPATH."
        ) from exc


class SAMAdapter:
    """
    Unified adapter for Meta's SAM 2 inference on microscopy scans.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path] = "checkpoints/sam2_hiera_large.pt",
        model_cfg: Optional[Union[str, Path]] = None,
        device: str = "auto",
        apply_postprocessing: bool = False,
    ):
        """
        Initialize the SAM 2 model adapter.

        Args:
            checkpoint: Path to model checkpoint file (.pt).
            model_cfg: Path to config YAML. If None, infers from checkpoint filename or defaults to large.
            device: 'auto', 'cuda', 'mps', or 'cpu'.
            apply_postprocessing: SAM 2 internal post-processing flag.
        """
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}\n"
                f"Download weights using: imagetochaste download-weights --model large"
            )

        self.device = resolve_device(device)
        self.model_cfg = self._resolve_model_cfg(model_cfg)
        self.apply_postprocessing = apply_postprocessing

        build_sam2, self._AMGClass, self._PredictorClass = import_sam2()

        print(f"Loading SAM 2 model [{self.checkpoint_path.name}] on device [{self.device}]...")
        self.model = build_sam2(
            str(self.model_cfg),
            str(self.checkpoint_path),
            device=self.device,
            apply_postprocessing=self.apply_postprocessing,
        )
        self._generator = None
        self._predictor = None

    def _resolve_model_cfg(self, model_cfg: Optional[Union[str, Path]]) -> str:
        if model_cfg is not None:
            return str(model_cfg)
        # Attempt to infer from checkpoint filename
        fname = self.checkpoint_path.name.lower()
        for variant, meta in SAM2_CHECKPOINTS.items():
            if variant in fname or meta["filename"].lower() in fname:
                return meta["config"]
        # Default fallback
        return "configs/sam2/sam2_hiera_l.yaml"

    def get_mask_generator(self, **kwargs) -> Any:
        """
        Get or instantiate the Automatic Mask Generator with custom parameters.
        """
        params = dict(DEFAULT_AMG_PARAMS)
        params.update(kwargs)
        return self._AMGClass(model=self.model, **params)

    def get_predictor(self) -> Any:
        """
        Get or instantiate the interactive SAM 2 Image Predictor.
        """
        if self._predictor is None:
            self._predictor = self._PredictorClass(self.model)
        return self._predictor

    def generate_masks(
        self,
        image: Union[str, Path, np.ndarray],
        filter_area: bool = True,
        preprocess: bool = False,
        amg_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Automatically segment all cells across an entire microscopy scan.

        Args:
            image: Image input (path or numpy array).
            filter_area: Whether to apply two-stage statistical area outlier rejection.
            preprocess: Whether to run CLAHE and illumination correction first.
            amg_params: Custom AMG hyperparameter dictionary.

        Returns:
            Tuple of (list_of_mask_dicts, metadata_dict)
        """
        import torch

        img_arr = load_image(image)
        if preprocess:
            img_arr = preprocess_microscopy_image(img_arr)

        generator = self.get_mask_generator(**(amg_params or {}))

        # Perform inference
        if self.device.type == "cuda":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                raw_masks = generator.generate(img_arr)
        else:
            with torch.inference_mode():
                raw_masks = generator.generate(img_arr)

        stats: Dict[str, Any] = {"raw_mask_count": len(raw_masks)}
        if filter_area:
            filtered_masks, filter_stats = filter_masks_by_area(raw_masks)
            stats.update(filter_stats)
            return filtered_masks, stats
        else:
            stats["final_count"] = len(raw_masks)
            return raw_masks, stats

    def predict_prompt(
        self,
        image: Union[str, Path, np.ndarray],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box_coords: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run prompt-guided segmentation using points or bounding boxes.

        Args:
            image: Image input.
            point_coords: Array of shape (N, 2) in (X, Y) pixel space.
            point_labels: Array of shape (N,) with 1 for positive, 0 for negative.
            box_coords: Array of shape (4,) in [X_min, Y_min, X_max, Y_max].
            multimask_output: Whether to return multiple mask ambiguities (3 masks).

        Returns:
            Tuple of (masks, scores, logits)
        """
        import torch

        img_arr = load_image(image)
        predictor = self.get_predictor()

        if self.device.type == "cuda":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(img_arr)
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_coords,
                    multimask_output=multimask_output,
                )
        else:
            with torch.inference_mode():
                predictor.set_image(img_arr)
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_coords,
                    multimask_output=multimask_output,
                )

        return masks, scores, logits
