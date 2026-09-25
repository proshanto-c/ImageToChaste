"""
Unit tests for geometric operations, contour extraction, centroids, and meshing.
"""

import numpy as np
import pytest

from imagetochaste.geometry.centroids import (
    compute_centroid,
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


def test_compute_centroid_circle(synthetic_circle_mask):
    """
    Test centroid calculation on symmetric circular mask.
    Center should be near (50, 50).
    """
    cx, cy = compute_centroid(synthetic_circle_mask)
    assert 49.5 <= cx <= 50.5
    assert 49.5 <= cy <= 50.5


def test_compute_centroid_empty():
    """
    Empty mask should return None.
    """
    empty_mask = np.zeros((50, 50), dtype=np.uint8)
    assert compute_centroid(empty_mask) is None


def test_compute_polygon_centroid():
    """
    Test analytical centroid of square [0, 10] x [0, 10].
    Centroid must be precisely (5.0, 5.0).
    """
    square_pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    cx, cy = compute_polygon_centroid(square_pts)
    assert pytest.approx(cx) == 5.0
    assert pytest.approx(cy) == 5.0


def test_extract_contours_circle(synthetic_circle_mask):
    """
    Confirm contour extraction approximates circle into polygonal vertices.
    """
    polys = extract_contours(synthetic_circle_mask, epsilon_factor=0.01)
    assert len(polys) == 1
    poly = polys[0]
    assert len(poly) >= 6  # Circle polygon approximation has multiple vertices
    assert poly.shape[1] == 2  # (V, 2)


def test_extract_all_contours(synthetic_adjacent_masks):
    """
    Confirm extraction over multiple masks returns one polygon per cell.
    """
    polys = extract_all_contours(synthetic_adjacent_masks)
    assert len(polys) == 2
    for p in polys:
        assert len(p) >= 4  # Squares have at least 4 vertices


def test_polygon_orientation():
    """
    Confirm orient_polygon guarantees counter-clockwise vertex winding order.
    """
    # Clockwise square
    cw_square = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]])
    area_cw = polygon_signed_area(cw_square)
    assert area_cw < 0  # Negative indicates CW

    ccw_square = orient_polygon(cw_square, counter_clockwise=True)
    area_ccw = polygon_signed_area(ccw_square)
    assert area_ccw > 0  # Positive indicates CCW


def test_polygon_validation():
    """
    Test polygon topological validation on normal, self-intersecting, and degenerate polygons.
    """
    # Valid triangle
    valid_tri = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 5.0]])
    is_valid, reason = validate_polygon(valid_tri)
    assert is_valid

    # Self-intersecting figure-8 polygon
    fig8 = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    assert is_self_intersecting(fig8)
    is_valid, reason = validate_polygon(fig8)
    assert not is_valid
    assert "Self-intersecting" in reason

    # Degenerate collinear points
    collinear = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    is_valid, _ = validate_polygon(collinear)
    assert not is_valid


def test_build_voronoi_mesh(sample_centroids):
    """
    Test bounded Voronoi mesh builder on 2D centroids.
    """
    mesh = build_voronoi_mesh(sample_centroids, bounding_box=(0.0, 0.0, 60.0, 70.0))
    assert mesh.num_nodes > 0
    assert mesh.num_elements > 0

    # Ensure all elements have at least 3 nodes
    for elem in mesh.elements:
        assert elem.num_nodes >= 3
        # Check node IDs exist in mesh
        for nid in elem.node_ids:
            assert 0 <= nid < mesh.num_nodes


def test_build_skeleton_junction_mesh():
    """
    Test skeletonization and junction detection on synthetic 3-way branching shape.
    """
    # Create image with 3 adjacent blocks meeting at a central T-junction
    h, w = 60, 60
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    m3 = np.zeros((h, w), dtype=np.uint8)

    m1[5:28, 5:28] = 1
    m2[5:28, 32:55] = 1
    m3[32:55, 5:55] = 1

    masks = [{"segmentation": m1}, {"segmentation": m2}, {"segmentation": m3}]
    nodes, edges = build_skeleton_junction_mesh(masks, image_shape=(h, w))

    assert len(nodes) >= 1
    assert len(edges) >= 1


def test_extract_shared_edges(synthetic_adjacent_masks):
    """
    Confirm shared edge detection finds adjacency between touching cells.
    """
    edge_map, adjacency = extract_shared_edges(synthetic_adjacent_masks, dilation_radius=3)
    assert np.any(edge_map)
    assert (0, 1) in adjacency or (1, 0) in adjacency
