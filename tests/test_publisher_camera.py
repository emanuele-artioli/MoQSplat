"""CPU math tests for the original publisher camera helpers."""

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture
def camera():
    # Load the actual definitions without importing CUDA, YOLO or publisher dependencies.
    path = Path(__file__).resolve().parents[1] / "publisher" / "gsplat_clustering_vmaf.py"
    tree = ast.parse(path.read_text())
    names = {
        "Movement",
        "cam_axes_from_euler",
        "derive_views",
        "_movement_from_trajectory_frame",
        "_require_opengl_trajectory",
        "load_camera_json",
        "_cam_world_axes",
        "build_camera_looking_at",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    source = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    namespace = {"np": np, "Rsp": Rotation, "dataclass": dataclass, "json": json}
    exec(compile(ast.fix_missing_locations(source), str(path), "exec"), namespace)
    return namespace


def test_corrected_trajectory_view_and_offaxis_intrinsics(camera, tmp_path):
    rot = Rotation.from_euler("xyz", [12, 33, -10], degrees=True).as_matrix()
    pos = np.array([2.0, -3.0, 4.0])
    view = np.eye(4)
    view[:3, :3] = rot.T
    view[:3, 3] = -rot.T @ pos
    frame = {
        "view_matrix": view.tolist(),
        "camera_position": pos.tolist(),
        "fov": 80,
        "fov_angles": [-0.9, 0.6, -0.8, 0.7],
        "width": 800,
        "height": 600,
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"metadata": {"camera_basis": "OpenXR/OpenGL"}, "frames": [frame]}))
    base = camera["load_camera_json"](str(path))
    expected = np.diag([1.0, -1.0, -1.0, 1.0]) @ view
    np.testing.assert_allclose(base.viewmat_override, expected, atol=1e-6)
    left, right, down, up = np.tan(frame["fov_angles"])
    assert base.fx == pytest.approx(800 / (right - left))
    assert base.fy == pytest.approx(600 / (up - down))
    assert base.cx == pytest.approx(-base.fx * left)
    assert base.cy == pytest.approx(base.fy * up)
    views = camera["derive_views"](base, [(0, 0), (90, 0)], [0], view_distance=3)
    np.testing.assert_allclose(views[0].viewmat_override, expected, atol=1e-6)
    assert not np.allclose(views[1].viewmat_override, expected)
    for derived in views:
        np.testing.assert_allclose(
            derived.viewmat_override @ np.r_[[derived.x, derived.y, derived.z], 1],
            [0, 0, 0, 1],
            atol=1e-6,
        )
    centered = camera["build_camera_looking_at"](np.array([1.0, 2.0, 3.0]), base, 1.0)
    np.testing.assert_allclose(centered.viewmat_override[:3, :3], expected[:3, :3], atol=1e-6)


def test_reject_ambiguous_trajectory(camera, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"frames": [{"view_matrix": np.eye(4).tolist(), "fov": 60}]}))
    with pytest.raises(ValueError, match="OpenGL"):
        camera["load_camera_json"](str(path))


def test_legacy_camera_retains_existing_euler_path(camera, tmp_path):
    path = tmp_path / "legacy.json"
    frame = {
        "angle": 25,
        "elevation": -10,
        "x": 2,
        "y": 3,
        "z": 4,
        "fx": 400,
        "fy": 410,
        "cx": 320,
        "cy": 240,
        "width": 640,
        "height": 480,
    }
    path.write_text(json.dumps([frame]))
    base = camera["load_camera_json"](str(path))
    derived = camera["derive_views"](base, [(0, 0)], [0], view_distance=4)[0]
    assert derived.viewmat_override is None
    assert derived.angle == 25 and derived.elevation == -10
    np.testing.assert_allclose([derived.x, derived.y, derived.z], [2, 3, 4], atol=1e-6)
    assert derived.fx == 400 and derived.fy == 410


def test_invalid_offaxis_angles_are_rejected(camera):
    frame = {"view_matrix": np.eye(4).tolist(), "fov_angles": [-0.9, 0.6, float("nan"), 0.7]}
    with pytest.raises(ValueError, match="fov_angles"):
        camera["_movement_from_trajectory_frame"](frame)
