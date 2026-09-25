"""
Polygon validation, winding orientation, and topological conditioning for Chaste meshes.
"""

from typing import Tuple

import numpy as np


def polygon_signed_area(vertices: np.ndarray) -> float:
    """
    Calculate the signed area of a 2D polygon using the Shoelace formula.
    Positive area indicates counter-clockwise (CCW) winding.
    """
    pts = np.asarray(vertices, dtype=np.float64)
    if len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    # Sum over consecutive edges
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def orient_polygon(vertices: np.ndarray, counter_clockwise: bool = True) -> np.ndarray:
    """
    Ensure polygon vertices follow the required winding orientation.
    Chaste vertex models require counter-clockwise (CCW) vertex ordering for positive element Jacobian.

    Args:
        vertices: Array of shape (N, 2) in (x, y) space.
        counter_clockwise: If True, enforce CCW ordering. If False, enforce clockwise.

    Returns:
        Array of shape (N, 2) with corrected vertex sequence.
    """
    pts = np.asarray(vertices, dtype=np.float64).copy()
    signed_area = polygon_signed_area(pts)

    is_ccw = signed_area > 0
    if (counter_clockwise and not is_ccw) or (not counter_clockwise and is_ccw):
        pts = np.flip(pts, axis=0)

    return pts


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    """
    Check if two 2D line segments (p1-p2 and q1-q2) strictly intersect.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    p1_a, p2_a = np.array(p1), np.array(p2)
    q1_a, q2_a = np.array(q1), np.array(q2)

    # Check straddling
    return (ccw(p1_a, q1_a, q2_a) != ccw(p2_a, q1_a, q2_a)) and (
        ccw(p1_a, p2_a, q1_a) != ccw(p1_a, p2_a, q2_a)
    )


def is_self_intersecting(vertices: np.ndarray) -> bool:
    """
    Determine whether a polygon's boundary edges intersect each other.
    """
    n = len(vertices)
    if n < 4:
        return False

    for i in range(n):
        p1 = tuple(vertices[i])
        p2 = tuple(vertices[(i + 1) % n])

        for j in range(i + 2, n):
            # Do not test adjacent edges sharing a vertex
            if (j + 1) % n == i:
                continue

            q1 = tuple(vertices[j])
            q2 = tuple(vertices[(j + 1) % n])

            if segments_intersect(p1, p2, q1, q2):
                return True

    return False


def validate_polygon(
    vertices: np.ndarray, min_area: float = 1.0
) -> Tuple[bool, str]:
    """
    Validate that a polygon is topologically sound for simulation meshing.

    Checks:
        1. At least 3 vertices.
        2. No NaN or Infinite coordinates.
        3. Non-zero area >= min_area.
        4. No self-intersecting boundary edges.

    Returns:
        Tuple of (is_valid, reason_string).
    """
    pts = np.asarray(vertices, dtype=np.float64)
    if len(pts) < 3:
        return False, f"Degenerate polygon: only {len(pts)} vertices."

    if not np.all(np.isfinite(pts)):
        return False, "Polygon contains non-finite (NaN or Inf) coordinates."

    if is_self_intersecting(pts):
        return False, "Self-intersecting polygon boundary detected."

    area = abs(polygon_signed_area(pts))
    if area < min_area:
        return False, f"Degenerate polygon area: {area:.3f} < {min_area}."

    return True, "Valid polygon"
