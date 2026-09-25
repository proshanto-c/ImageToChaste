"""
ImageToChaste: Turn microscopy images into Chaste C++ simulation meshes
using Meta's Segment Anything Model 2 (SAM 2).
"""

__version__ = "0.1.1"

from imagetochaste.exporters.base import ChasteMesh, Element, Node
from imagetochaste.exporters.chaste_formatter import (
    export_chaste_nodes,
    export_chaste_vertex_mesh,
    format_elements_string,
    format_nodes_string,
)
from imagetochaste.geometry.centroids import (
    compute_centroid,
    compute_centroids_from_masks,
    compute_polygon_centroid,
)
from imagetochaste.geometry.contours import extract_all_contours, extract_contours
from imagetochaste.geometry.mesh_builder import (
    build_skeleton_junction_mesh,
    build_voronoi_mesh,
    extract_shared_edges,
)
from imagetochaste.segmentation.filter import filter_masks_by_area
from imagetochaste.segmentation.preprocessor import preprocess_microscopy_image
from imagetochaste.segmentation.sam_adapter import SAMAdapter

__all__ = [
    "__version__",
    "SAMAdapter",
    "preprocess_microscopy_image",
    "filter_masks_by_area",
    "extract_contours",
    "extract_all_contours",
    "compute_centroid",
    "compute_centroids_from_masks",
    "compute_polygon_centroid",
    "build_voronoi_mesh",
    "build_skeleton_junction_mesh",
    "extract_shared_edges",
    "Node",
    "Element",
    "ChasteMesh",
    "export_chaste_nodes",
    "export_chaste_vertex_mesh",
    "format_nodes_string",
    "format_elements_string",
]
