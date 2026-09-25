"""
Geometry processing module for contour extraction, centroids, validation, and meshing.
"""

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
from imagetochaste.geometry.validator import (
    is_self_intersecting,
    orient_polygon,
    polygon_signed_area,
    validate_polygon,
)

__all__ = [
    "compute_centroid",
    "compute_centroids_from_masks",
    "compute_polygon_centroid",
    "extract_contours",
    "extract_all_contours",
    "build_voronoi_mesh",
    "build_skeleton_junction_mesh",
    "extract_shared_edges",
    "polygon_signed_area",
    "orient_polygon",
    "is_self_intersecting",
    "validate_polygon",
]
