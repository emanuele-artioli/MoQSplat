"""Source-derived asymmetric OpenXR projection contracts, independent of CUDA."""

import math

import numpy as np
import pytest

from moq3dgs.models import ViewportUpdate
from moq3dgs.viewport.frustum import camera_intrinsics_from_fov, projection_matrix_from_fov

ANGLES = (-0.942, 0.698, -0.942, 0.733)


def test_offaxis_gl_corners_and_cv_intrinsics_agree():
    near, far, width, height = 0.2, 250.0, 640, 480
    left, right, down, up = np.tan(ANGLES)
    projection = projection_matrix_from_fov(95.969, width / height, near, far, ANGLES)
    intrinsics = camera_intrinsics_from_fov(width, height, 95.969, ANGLES)
    for x, nx in ((left, -1), (right, 1)):
        for y, ny in ((down, -1), (up, 1)):
            point_gl = np.array([near * x, near * y, -near, 1])
            clip = projection @ point_gl
            np.testing.assert_allclose(clip[:3] / clip[3], [nx, ny, -1], atol=1e-7)
            pixel_cv = intrinsics @ np.array([near * x, -near * y, near])
            np.testing.assert_allclose(
                pixel_cv[:2] / pixel_cv[2], [(nx + 1) * width / 2, (1 - ny) * height / 2], atol=3e-5
            )
    assert intrinsics[0, 2] != width / 2
    assert intrinsics[1, 2] != height / 2


def test_symmetric_callers_keep_the_same_projection():
    fov = 60.0
    projection = projection_matrix_from_fov(fov, 4 / 3)
    intrinsics = camera_intrinsics_from_fov(640, 480, fov)
    assert projection[0, 2] == projection[1, 2] == 0
    assert intrinsics[0, 2] == 320 and intrinsics[1, 2] == 240
    np.testing.assert_allclose(intrinsics[0, 0], intrinsics[1, 1])


@pytest.mark.parametrize(
    "angles",
    [
        (0.1, -0.1, -0.2, 0.2),
        (-0.2, 0.2, 0.2, -0.2),
        (-math.pi / 2, 0.2, -0.2, 0.2),
        (float("nan"), 0.2, -0.2, 0.2),
    ],
)
def test_invalid_offaxis_angles_rejected_by_model_and_projection(angles):
    with pytest.raises(ValueError):
        projection_matrix_from_fov(60, fov_angles=angles)
    with pytest.raises(ValueError):
        ViewportUpdate(
            client_id="test",
            timestamp_ms=0,
            camera_position=[0, 0, 0],
            view_matrix=np.eye(4).tolist(),
            fov=60,
            fov_angles=angles,
        )
