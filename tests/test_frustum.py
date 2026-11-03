"""Unit tests for frustum culling."""

from __future__ import annotations

import numpy as np
import pytest

from moq3dgs.viewport.frustum import (
    Visibility,
    extract_frustum,
    projection_matrix_from_fov,
    check_aabb_frustum,
)


def _make_view_proj(fov: float = 60.0) -> np.ndarray:
    """Build a VP matrix looking down -Z from origin."""
    proj = projection_matrix_from_fov(fov)
    # Identity view matrix = camera at origin looking down -Z
    view = np.eye(4, dtype=np.float64)
    return proj @ view


class TestFrustumExtraction:
    """Test frustum plane extraction."""

    def test_six_planes(self) -> None:
        vp = _make_view_proj()
        frustum = extract_frustum(vp)
        assert len(frustum.planes) == 6

    def test_planes_normalised(self) -> None:
        vp = _make_view_proj()
        frustum = extract_frustum(vp)
        for plane in frustum.planes:
            norm = np.linalg.norm(plane.normal)
            np.testing.assert_almost_equal(norm, 1.0, decimal=5)


class TestAABBFrustum:
    """Test AABB-frustum intersection."""

    def test_box_in_front(self) -> None:
        """A box directly in front of the camera should be INSIDE or INTERSECTING."""
        vp = _make_view_proj(90.0)
        frustum = extract_frustum(vp)
        result = check_aabb_frustum(
            np.array([-0.5, -0.5, -5.0]),
            np.array([0.5, 0.5, -2.0]),
            frustum,
        )
        assert result in (Visibility.INSIDE, Visibility.INTERSECTING)

    def test_box_behind(self) -> None:
        """A box behind the camera should be OUTSIDE."""
        vp = _make_view_proj(60.0)
        frustum = extract_frustum(vp)
        result = check_aabb_frustum(
            np.array([-1, -1, 10.0]),
            np.array([1, 1, 20.0]),
            frustum,
        )
        assert result == Visibility.OUTSIDE

    def test_box_far_to_side(self) -> None:
        """A box far to the right should be OUTSIDE."""
        vp = _make_view_proj(60.0)
        frustum = extract_frustum(vp)
        result = check_aabb_frustum(
            np.array([100, -1, -5]),
            np.array([200, 1, -2]),
            frustum,
        )
        assert result == Visibility.OUTSIDE


def test_translated_frustum_all_six_boundaries() -> None:
    view = np.eye(4)
    view[:3, 3] = [-3, 2, -4]
    frustum = extract_frustum(projection_matrix_from_fov(90, 1, 1, 10) @ view)
    camera = np.array([3., -2., 4.])
    def classify(offset: tuple[float, float, float]) -> Visibility:
        center = camera + np.array(offset)
        return check_aabb_frustum(center - 0.05, center + 0.05, frustum)
    assert classify((0, 0, -5)) == Visibility.INSIDE
    for center in [(-7, 0, -5), (7, 0, -5), (0, -7, -5),
                   (0, 7, -5), (0, 0, -0.5), (0, 0, -11)]:
        assert classify(center) == Visibility.OUTSIDE
