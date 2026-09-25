"""
Mesh generation algorithms: Bounded Voronoi tessellation, skeleton junction graphs,
and shared cell boundary extraction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np
from scipy.spatial import Voronoi
from skimage.morphology import dilation, skeletonize

from imagetochaste.exporters.base import ChasteMesh, Element, Node
from imagetochaste.geometry.validator import orient_polygon, validate_polygon


def build_voronoi_mesh(
    centroids: List[Tuple[float, float]],
    bounding_box: Optional[Tuple[float, float, float, float]] = None,
    snap_tolerance: float = 1e-4,
) -> ChasteMesh:
    """
    Construct a bounded 2D Voronoi mesh from cell centroids.

    Clips unbounded Voronoi rays to the bounding box so that perimeter cells
    have well-defined, closed polygonal elements suitable for Chaste VertexMesh.

    Args:
        centroids: List of (x, y) cell center coordinates.
        bounding_box: (xmin, ymin, xmax, ymax). If None, calculated from centroid extents.
        snap_tolerance: Distance threshold to deduplicate coincident junction nodes.

    Returns:
        ChasteMesh object containing uniquely indexed Nodes and CCW-oriented Elements.
    """
    pts = np.asarray(centroids, dtype=np.float64)
    if len(pts) < 4:
        raise ValueError(f"Voronoi meshing requires at least 4 centroids, got {len(pts)}")

    if bounding_box is None:
        min_x, min_y = np.min(pts, axis=0) - 20.0
        max_x, max_y = np.max(pts, axis=0) + 20.0
        bounding_box = (float(min_x), float(min_y), float(max_x), float(max_y))

    xmin, ymin, xmax, ymax = bounding_box

    # Mirror boundary points around bounding box to close infinite regions cleanly
    mirrored_pts = [pts]
    mirrored_pts.append(np.column_stack([2 * xmin - pts[:, 0], pts[:, 1]]))
    mirrored_pts.append(np.column_stack([2 * xmax - pts[:, 0], pts[:, 1]]))
    mirrored_pts.append(np.column_stack([pts[:, 0], 2 * ymin - pts[:, 1]]))
    mirrored_pts.append(np.column_stack([pts[:, 0], 2 * ymax - pts[:, 1]]))

    all_pts = np.vstack(mirrored_pts)
    vor = Voronoi(all_pts)

    # Dictionary to deduplicate nodes: rounded (x, y) -> Node
    node_registry: Dict[Tuple[float, float], int] = {}
    nodes: List[Node] = []
    elements: List[Element] = []

    def get_or_create_node(x: float, y: float) -> int:
        # Clip to bounding box
        cl_x = float(np.clip(x, xmin, xmax))
        cl_y = float(np.clip(y, ymin, ymax))
        key = (round(cl_x / snap_tolerance) * snap_tolerance, round(cl_y / snap_tolerance) * snap_tolerance)
        if key in node_registry:
            return node_registry[key]

        new_id = len(nodes)
        is_boundary = np.isclose(cl_x, xmin) or np.isclose(cl_x, xmax) or np.isclose(cl_y, ymin) or np.isclose(cl_y, ymax)
        node_registry[key] = new_id
        nodes.append(Node(node_id=new_id, x=cl_x, y=cl_y, is_boundary=bool(is_boundary)))
        return new_id

    # For each original cell centroid, find its Voronoi region
    for cell_idx in range(len(pts)):
        region_idx = vor.point_region[cell_idx]
        vertex_indices = vor.regions[region_idx]

        if not vertex_indices or -1 in vertex_indices:
            continue

        region_vertices = vor.vertices[vertex_indices]
        # Filter vertices within extended bounds
        valid_mask = (
            (region_vertices[:, 0] >= xmin - 10) &
            (region_vertices[:, 0] <= xmax + 10) &
            (region_vertices[:, 1] >= ymin - 10) &
            (region_vertices[:, 1] <= ymax + 10)
        )
        if np.sum(valid_mask) < 3:
            continue

        poly_pts = region_vertices[valid_mask]
        poly_pts = orient_polygon(poly_pts, counter_clockwise=True)
        is_valid, _ = validate_polygon(poly_pts, min_area=0.5)
        if not is_valid:
            continue

        element_node_ids = [get_or_create_node(vx, vy) for vx, vy in poly_pts]

        # Deduplicate consecutive identical node IDs in element loop
        dedup_node_ids = []
        for nid in element_node_ids:
            if not dedup_node_ids or nid != dedup_node_ids[-1]:
                dedup_node_ids.append(nid)
        if len(dedup_node_ids) > 1 and dedup_node_ids[0] == dedup_node_ids[-1]:
            dedup_node_ids.pop()

        if len(dedup_node_ids) >= 3:
            elements.append(Element(element_id=len(elements), node_ids=dedup_node_ids))

    return ChasteMesh(nodes=nodes, elements=elements)


def build_skeleton_junction_mesh(
    masks: List[Union[Dict[str, Any], np.ndarray]],
    image_shape: Tuple[int, int],
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    """
    Extract cell junction vertices and interconnected line segments using skeletonization.

    Adapts morphological boundary thinning and graph junction pruning.

    Args:
        masks: List of SAM 2 mask dicts or binary arrays.
        image_shape: (H, W) tuple of image dimensions.

    Returns:
        Tuple of (junction_coordinates, edge_endpoint_indices).
    """
    h, w = image_shape
    combined = np.zeros((h, w), dtype=np.uint8)

    for m in masks:
        bin_m = m["segmentation"] if isinstance(m, dict) else m
        combined = np.logical_or(combined, bin_m > 0).astype(np.uint8)

    # Invert so cell interstitial boundaries become foreground (1)
    boundary_binary = (combined == 0).astype(np.uint8)
    skeleton = skeletonize(boundary_binary).astype(np.uint8)

    # 3x3 convolution kernel to count neighboring pixels
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel) - skeleton

    junction_mask = np.logical_and(skeleton == 1, neighbor_count >= 3)
    junction_coords = np.column_stack(np.where(junction_mask))  # (y, x)

    # Build NetworkX graph from skeleton pixels
    G = nx.Graph()
    skel_coords = np.column_stack(np.where(skeleton > 0))

    for y, x in skel_coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_] == 1:
                    G.add_edge((y, x), (ny, nx_))

    junction_set = set((int(y), int(x)) for y, x in junction_coords)
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    poi = list(junction_set) + endpoints

    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    visited_edges = set()

    for start in poi:
        for neighbor in G.neighbors(start):
            edge_key = tuple(sorted([start, neighbor]))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            curr = neighbor
            prev = start
            while curr not in junction_set and G.degree[curr] == 2:
                next_nodes = [n for n in G.neighbors(curr) if n != prev]
                if not next_nodes:
                    break
                next_ = next_nodes[0]
                visited_edges.add(tuple(sorted([curr, next_])))
                prev, curr = curr, next_

            edges.append((start, curr))

    # Convert to (x, y) junction list and index pairs
    unique_nodes: List[Tuple[float, float]] = []
    node_to_idx: Dict[Tuple[int, int], int] = {}

    def get_idx(pt: Tuple[int, int]) -> int:
        if pt not in node_to_idx:
            node_to_idx[pt] = len(unique_nodes)
            unique_nodes.append((float(pt[1]), float(pt[0])))  # (x, y)
        return node_to_idx[pt]

    indexed_edges: List[Tuple[int, int]] = []
    for p1, p2 in edges:
        i1 = get_idx(p1)
        i2 = get_idx(p2)
        if i1 != i2:
            indexed_edges.append((i1, i2))

    return unique_nodes, indexed_edges


def _get_square_footprint(size: int):
    try:
        from skimage.morphology import footprint_rectangle
        return footprint_rectangle((size, size))
    except (ImportError, TypeError):
        from skimage.morphology import square
        return square(size)


def extract_shared_edges(
    masks: List[Union[Dict[str, Any], np.ndarray]],
    dilation_radius: int = 3,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Simulate cell contact for disjoint masks using morphological dilation
    and identify shared cell boundaries.

    Returns:
        Tuple of (shared_edge_map_boolean_image, list_of_adjacent_mask_index_pairs).
    """
    n_masks = len(masks)
    if n_masks == 0:
        return np.zeros((0, 0), dtype=bool), []

    first_mask = masks[0]["segmentation"] if isinstance(masks[0], dict) else masks[0]
    h, w = first_mask.shape

    dilated_masks = []
    struct_el = _get_square_footprint(dilation_radius)

    for m in masks:
        bin_m = m["segmentation"] if isinstance(m, dict) else m
        dilated_masks.append(dilation(bin_m > 0, struct_el))

    shared_edge_map = np.zeros((h, w), dtype=bool)
    adjacency: List[Tuple[int, int]] = []

    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            overlap = dilated_masks[i] & dilated_masks[j]
            if np.any(overlap):
                shared_edge_map |= overlap
                adjacency.append((i, j))

    return shared_edge_map, adjacency
