"""View-frustum extraction and AABB intersection tests.

The frustum is built from the 4×4 view-projection matrix and represented
as six half-planes.  An axis-aligned bounding box (AABB) is tested against
these planes to decide whether a spatial cluster is visible, partially
visible, or completely outside the frustum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

import numpy as np


class Visibility(IntEnum):
    """Result of a frustum–AABB intersection test."""

    OUTSIDE = 0
    INTERSECTING = 1
    INSIDE = 2


@dataclass
class FrustumPlane:
    """A single frustum half-plane in Hessian normal form (nx, ny, nz, d)."""

    normal: np.ndarray  # (3,) inward-pointing: nonnegative distance is inside
    d: float            # signed distance from origin


@dataclass
class Frustum:
    """Six-plane view frustum extracted from a view-projection matrix."""

    planes: List[FrustumPlane]  # [left, right, bottom, top, near, far]


def projection_matrix_from_fov(
    fov_y_deg: float,
    aspect: float = 16.0 / 9.0,
    near: float = 0.01,
    far: float = 100.0,
    fov_angles: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Build a perspective projection, preserving optional signed off-axis angles.

    Uses the OpenGL convention (column-major, right-handed, depth [-1, 1]).

    Args:
        fov_y_deg: Vertical field of view in degrees.
        aspect: Width / height ratio.
        near: Near clip distance.
        far: Far clip distance.

    Returns:
        4×4 projection matrix (row-major numpy array).
    """
    if fov_angles is not None:
        left, right, down, up = fov_tangents(fov_angles)
        if not 0 < near < far:
            raise ValueError("clip planes must satisfy 0 < near < far")
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = 2.0 / (right - left)
        proj[1, 1] = 2.0 / (up - down)
        proj[0, 2] = (right + left) / (right - left)
        proj[1, 2] = (up + down) / (up - down)
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2.0 * far * near / (near - far)
        proj[3, 2] = -1.0
        return proj
    fov_rad = np.radians(fov_y_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)

    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def fov_tangents(fov_angles: tuple[float, float, float, float]) -> np.ndarray:
    """Validate signed EyeNav/OpenXR angles ordered left, right, down, up."""
    angles = np.asarray(fov_angles, dtype=np.float64)
    if angles.shape != (4,) or not np.isfinite(angles).all():
        raise ValueError("FOV angles must contain four finite signed radians")
    if np.any(np.abs(angles) >= np.pi / 2) or angles[0] >= angles[1] or angles[2] >= angles[3]:
        raise ValueError("FOV angles must be ordered within (-pi/2, pi/2)")
    return np.tan(angles)


def camera_intrinsics_from_fov(
    width: int, height: int, fov_y_deg: float,
    fov_angles: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Build OpenCV intrinsics matching the OpenGL off-axis projection."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if fov_angles is None:
        fy = (height / 2.0) / np.tan(np.radians(fov_y_deg) / 2.0)
        fx, cx, cy = fy, width / 2.0, height / 2.0
    else:
        left, right, down, up = fov_tangents(fov_angles)
        fx, fy = width / (right - left), height / (up - down)
        cx, cy = -fx * left, fy * up
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def extract_frustum(view_proj: np.ndarray) -> Frustum:
    """Extract six frustum planes from a 4×4 view-projection matrix.

    Uses the Gribb-Hartmann method: each plane is a linear combination of
    two rows of the VP matrix.

    Args:
        view_proj: Combined View × Projection matrix (row-major 4×4).

    Returns:
        :class:`Frustum` with six normalised planes.
    """
    m = view_proj  # alias for brevity

    def _normalise(n: np.ndarray, d: float) -> FrustumPlane:
        length = np.linalg.norm(n)
        if length < 1e-12:
            length = 1e-12
        return FrustumPlane(normal=n / length, d=d / length)

    planes = [
        # Left:   row3 + row0
        _normalise(
            np.array([m[3, 0] + m[0, 0], m[3, 1] + m[0, 1], m[3, 2] + m[0, 2]]),
            m[3, 3] + m[0, 3],
        ),
        # Right:  row3 - row0
        _normalise(
            np.array([m[3, 0] - m[0, 0], m[3, 1] - m[0, 1], m[3, 2] - m[0, 2]]),
            m[3, 3] - m[0, 3],
        ),
        # Bottom: row3 + row1
        _normalise(
            np.array([m[3, 0] + m[1, 0], m[3, 1] + m[1, 1], m[3, 2] + m[1, 2]]),
            m[3, 3] + m[1, 3],
        ),
        # Top:    row3 - row1
        _normalise(
            np.array([m[3, 0] - m[1, 0], m[3, 1] - m[1, 1], m[3, 2] - m[1, 2]]),
            m[3, 3] - m[1, 3],
        ),
        # Near:   row3 + row2
        _normalise(
            np.array([m[3, 0] + m[2, 0], m[3, 1] + m[2, 1], m[3, 2] + m[2, 2]]),
            m[3, 3] + m[2, 3],
        ),
        # Far:    row3 - row2
        _normalise(
            np.array([m[3, 0] - m[2, 0], m[3, 1] - m[2, 1], m[3, 2] - m[2, 2]]),
            m[3, 3] - m[2, 3],
        ),
    ]
    return Frustum(planes=planes)


def check_aabb_frustum(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    frustum: Frustum,
) -> Visibility:
    """Test an AABB against a frustum.

    Uses the optimised positive-vertex / negative-vertex method for fast
    rejection.

    Args:
        bbox_min: (3,) lower corner of the AABB.
        bbox_max: (3,) upper corner of the AABB.
        frustum: View frustum to test against.

    Returns:
        :class:`Visibility` enum indicating the intersection status.
    """
    result = Visibility.INSIDE
    for plane in frustum.planes:
        n = plane.normal
        # Positive vertex: the corner of the AABB most along the plane normal
        p_vertex = np.where(n >= 0, bbox_max, bbox_min)
        # Negative vertex: opposite corner
        n_vertex = np.where(n >= 0, bbox_min, bbox_max)

        if np.dot(n, p_vertex) + plane.d < 0:
            return Visibility.OUTSIDE
        if np.dot(n, n_vertex) + plane.d < 0:
            result = Visibility.INTERSECTING

    return result


def aabb_center(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Compute the centre of an AABB."""
    return (bbox_min + bbox_max) / 2.0
