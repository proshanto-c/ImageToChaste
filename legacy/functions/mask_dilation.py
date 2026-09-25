from skimage.morphology import dilation, square
from skimage.segmentation import find_boundaries
import numpy as np
from skimage.measure import label
from sklearn.decomposition import PCA
from collections import defaultdict

def clean_and_connect_segments(segments, rounding=1):
    """
    Ensures that each endpoint belongs to at least two segments.
    Endpoints are snapped to a rounded grid for matching.
    
    Parameters:
    - segments: list of (i, j, x0, y0, x1, y1)
    - rounding: round coordinates to this many pixels for endpoint merging

    Returns:
    - cleaned_segments: list of (x0, y0, x1, y1) with shared endpoints
    - endpoint_counts: dictionary of rounded endpoints and their connection counts
    """
    endpoint_map = defaultdict(int)
    cleaned_segments = []

    def round_point(x, y):
        return (int(round(x / rounding) * rounding), int(round(y / rounding) * rounding))

    for i, j, x0, y0, x1, y1 in segments:
        p0 = round_point(x0, y0)
        p1 = round_point(x1, y1)
        if p0 == p1:
            continue  # skip degenerate
        endpoint_map[p0] += 1
        endpoint_map[p1] += 1
        cleaned_segments.append((*p0, *p1))

    # Filter: keep only segments whose both ends are used ≥ 2 times
    filtered_segments = [
        (x0, y0, x1, y1)
        for x0, y0, x1, y1 in cleaned_segments
        if endpoint_map[(x0, y0)] >= 2 and endpoint_map[(x1, y1)] >= 2
    ]

    return filtered_segments, endpoint_map

def extract_expanded_edge_segments_from_disjoint_masks(filtered_masks, dilation_radius=3):
    """
    Improved version: returns line segments that span the full shared boundary using PCA.
    Each segment is (i, j, x0, y0, x1, y1) where (x0, y0) and (x1, y1) are endpoints of the best-fit line.
    """
    n = len(filtered_masks)
    segments = []

    # Step 1: Dilate masks
    dilated_masks = [dilation(m['segmentation'], square(dilation_radius)) for m in filtered_masks]

    # Step 2: Compare each pair
    for i in range(n):
        for j in range(i + 1, n):
            overlap = dilated_masks[i] & filtered_masks[j]['segmentation']
            if not np.any(overlap):
                overlap = dilated_masks[j] & filtered_masks[i]['segmentation']
            if not np.any(overlap):
                continue

            coords = np.argwhere(overlap)
            if len(coords) < 2:
                continue  # Not enough to fit a line

            # PCA to find best-fit line
            pca = PCA(n_components=1)
            pca.fit(coords)
            direction = pca.components_[0]
            center = pca.mean_

            # Project points onto the line to find endpoints
            projections = coords @ direction
            min_proj, max_proj = projections.min(), projections.max()
            endpoint1 = center + direction * (min_proj - center @ direction)
            endpoint2 = center + direction * (max_proj - center @ direction)
            y0, x0 = endpoint1
            y1, x1 = endpoint2

            segments.append((i, j, x0, y0, x1, y1))

    return segments

def extract_shared_edge_segments_from_disjoint_masks(filtered_masks, dilation_radius=3):
    """
    Given a list of disjoint binary masks, returns a list of line segments for each shared edge.
    Each segment is (i, j, x0, y0, x1, y1) where:
      - i, j are indices of adjacent masks
      - (x0, y0) to (x1, y1) is a line segment approximating the shared boundary
    """
    n = len(filtered_masks)
    segments = []

    # Step 1: Dilate masks to simulate contact
    dilated_masks = [dilation(m['segmentation'], square(dilation_radius)) for m in filtered_masks]

    # Step 2: Compare each pair to find overlaps
    for i in range(n):
        for j in range(i + 1, n):
            overlap = dilated_masks[i] & filtered_masks[j]['segmentation']
            if not np.any(overlap):
                overlap = dilated_masks[j] & filtered_masks[i]['segmentation']
            if not np.any(overlap):
                continue  # No adjacency

            coords = np.argwhere(overlap)
            if len(coords) < 2:
                continue  # Too small to define a line

            # Use first and last overlapping points as endpoints of a line segment
            y0, x0 = coords[0]
            y1, x1 = coords[-1]
            segments.append((i, j, x0, y0, x1, y1))

    return segments

def extract_shared_edge_segments(filtered_masks, dilation_radius=3):
    """
    Returns a list of line segments for each shared edge between masks.
    Each segment is represented as (i, j, x0, y0, x1, y1), where:
    - (i, j) are the mask indices
    - (x0, y0) and (x1, y1) define the endpoints of the line segment
    """
    n = len(filtered_masks)
    segments = []
    dilated_masks = [dilation(m['segmentation'], square(dilation_radius)) for m in filtered_masks]

    for i in range(n):
        for j in range(i + 1, n):
            # Identify shared boundary pixels
            overlap = dilated_masks[i] & filtered_masks[j]['segmentation']
            if not np.any(overlap):
                overlap = dilated_masks[j] & filtered_masks[i]['segmentation']
            if not np.any(overlap):
                continue

            coords = np.argwhere(overlap)
            if len(coords) < 2:
                continue  # Not enough points to define a segment

            # Simplest approximation: use first and last point
            y0, x0 = coords[0]
            y1, x1 = coords[-1]
            segments.append((i, j, x0, y0, x1, y1))

    return segments


def extract_shared_edges_from_disjoint_masks(filtered_masks, dilation_radius=3):
    """
    For disjoint masks, simulate cell contact using dilation.
    Returns:
    - shared_edge_map: binary image where True indicates a shared boundary pixel
    - adjacency: list of (i, j) mask index pairs that share a boundary
    """
    n_masks = len(filtered_masks)
    mask_shape = filtered_masks[0]['segmentation'].shape
    H, W = mask_shape

    # Step 1: Create label volume to track which masks cover each pixel
    overlap_counter = np.zeros((H, W), dtype=np.uint8)
    pixel_owners = np.full((H, W), None, dtype=object)

    for i in range(n_masks):
        dilated = dilation(filtered_masks[i]['segmentation'], square(dilation_radius))
        overlap_counter[dilated > 0] += 1

        # Track ownership
        mask_pixels = np.argwhere(dilated > 0)
        for y, x in mask_pixels:
            if pixel_owners[y, x] is None:
                pixel_owners[y, x] = {i}
            else:
                pixel_owners[y, x].add(i)

    # Step 2: Shared edge map is where 2+ masks overlap
    shared_edge_map = overlap_counter >= 2

    # Step 3: Extract adjacency from pixel ownership
    adjacency_set = set()
    shared_pixels = np.argwhere(shared_edge_map)

    for y, x in shared_pixels:
        owners = list(pixel_owners[y, x])
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                a, b = sorted((owners[i], owners[j]))
                adjacency_set.add((a, b))

    adjacency = np.array(list(adjacency_set), dtype=int)
    return shared_edge_map, adjacency

def get_shared_edges_and_adjacency(filtered_masks, dilation_size=3):
    """
    Given a list of binary masks (filtered_masks), returns:
    - shared_edge_map: a binary image with True at pixels on shared boundaries between cells
    - adjacency_list: a list of (i, j) tuples representing adjacent cell indices
    """

    # Step 1: Build labeled image from instance masks
    H, W = filtered_masks[0]['segmentation'].shape
    label_image = np.zeros((H, W), dtype=np.int32)

    for idx, m in enumerate(filtered_masks):
        label_image[m['segmentation'] > 0] = idx + 1  # labels start at 1

    # Step 2: Pad label image to handle edge cases during boundary checks
    padded_label = np.pad(label_image, pad_width=1, mode='constant', constant_values=0)
    shared_edge_map = np.zeros_like(label_image, dtype=bool)
    adjacency_set = set()

    # Step 3: Find boundaries
    boundary_map = find_boundaries(label_image, mode='thick')

    # Step 4: For each boundary pixel, check if it touches multiple unique labels
    for y in range(1, H + 1):
        for x in range(1, W + 1):
            if not boundary_map[y - 1, x - 1]:
                continue
            neighborhood = padded_label[y - 1:y + 2, x - 1:x + 2]
            unique_labels = np.unique(neighborhood)
            unique_labels = unique_labels[unique_labels > 0]
            if len(unique_labels) >= 2:
                shared_edge_map[y - 1, x - 1] = True
                for i in range(len(unique_labels)):
                    for j in range(i + 1, len(unique_labels)):
                        a, b = sorted((unique_labels[i] - 1, unique_labels[j] - 1))
                        adjacency_set.add((a, b))

    adjacency_list = list(adjacency_set)
    return shared_edge_map, np.array(adjacency_list, dtype=int)

def dilate_masks(filtered_masks, dilation_size=3):
    n = len(filtered_masks)
    dilated_masks = [dilation(m['segmentation'], square(dilation_size)) for m in filtered_masks]
    edges = set()

    for i in range(n):
        for j in range(i + 1, n):
            if np.any(dilated_masks[i] & filtered_masks[j]['segmentation']) or \
               np.any(dilated_masks[j] & filtered_masks[i]['segmentation']):
                edges.add((i, j))
    return list(edges)

def get_shared_edge_map(filtered_masks):
    # Step 1: Create a labeled image
    H, W = filtered_masks[0]['segmentation'].shape
    label_image = np.zeros((H, W), dtype=np.int32)
    for i, m in enumerate(filtered_masks):
        label_image[m['segmentation'] > 0] = i + 1

    # Step 2: Find boundaries
    boundary_map = find_boundaries(label_image, mode='thick')

    # Step 3: For each boundary pixel, check if neighboring pixels belong to 2+ unique labels
    shared_edge_map = np.zeros_like(label_image, dtype=bool)
    padded = np.pad(label_image, 1)
    for y in range(1, H + 1):
        for x in range(1, W + 1):
            if not boundary_map[y - 1, x - 1]:
                continue
            local = padded[y-1:y+2, x-1:x+2]
            unique_labels = np.unique(local)
            unique_labels = unique_labels[unique_labels > 0]
            if len(unique_labels) >= 2:
                shared_edge_map[y - 1, x - 1] = True

    return shared_edge_map