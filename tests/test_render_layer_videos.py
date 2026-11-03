"""Guards on the original whole-scene VMAF renderer."""

import ast
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "publisher" / "render_layer_videos.py"


@pytest.fixture
def helpers():
    tree = ast.parse(SCRIPT.read_text())
    names = {
        "_build_xyz_hash",
        "_mask_from_ply_file",
        "recover_object_full_masks",
        "load_ranked_source_indices",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    scope = {"np": np, "os": os, "hashlib": hashlib}
    exec(compile(ast.fix_missing_locations(module), str(SCRIPT), "exec"), scope)
    return scope


def test_duplicate_source_positions_fail_instead_of_silent_collision(helpers):
    class Model:
        xyz = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)

        def __len__(self):
            return 2

    with pytest.raises(ValueError, match="duplicate Gaussian XYZ"):
        helpers["_build_xyz_hash"](Model())


def test_missing_delta_fails_even_if_full_object_exists(helpers, tmp_path):
    class Model:
        xyz = np.array([[1, 2, 3]], dtype=np.float32)

        def __len__(self):
            return 1

    (tmp_path / "full.ply").write_bytes(b"not read")
    manifest = {
        "objects": [
            {
                "object_id": 0,
                "dir_relative_to_outdir": ".",
                "full_file": "full.ply",
                "layers": [{"file": "missing.ply"}],
            }
        ]
    }
    with pytest.raises(FileNotFoundError, match="missing DELTA"):
        helpers["recover_object_full_masks"](Model(), manifest, str(tmp_path))


def test_pipeline_requires_local_weights_only_for_segmentation(tmp_path):
    shell = SCRIPT.parent / "compute_rd_vmaf.sh"
    scene = tmp_path / "bicycle.ply"
    scene.write_text("fixture")
    env = os.environ.copy()
    env.pop("YOLO_MODEL", None)
    env.update(
        {
            "BICYCLE_PLY": str(scene),
            "BICYCLE_TRAJ": str(scene),
            "ROOM_PLY": str(tmp_path / "absent-room"),
            "ROOM_TRAJ": str(tmp_path / "absent-room-trace"),
            "SEG_SCRIPT": str(SCRIPT),
            "RENDER_SCRIPT": str(SCRIPT),
            "ORBIT_SCRIPT": str(SCRIPT),
            "FFMPEG_BIN": "/missing-ffmpeg",
        }
    )
    args = ["bash", str(shell), "--scene", "bicycle", "--work-dir", str(tmp_path / "work")]
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "YOLO_MODEL" in result.stderr
    result = subprocess.run(
        args + ["--skip-seg"], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0 and "ffmpeg" in result.stderr
    assert "YOLO_MODEL" not in result.stderr and "absent-room" not in result.stderr


def test_cuda_fails_without_gsplat_but_explicit_debug_fallback_works():
    path = ROOT / "publisher" / "gsplat_clustering_vmaf.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_one")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, node], type_ignores=[])
    scope = {"HAS_GSPLAT": False, "render_fallback": lambda *a, **kw: "debug"}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    with pytest.raises(RuntimeError, match="requires a working gsplat"):
        scope["render_one"](None, None, "cuda:0")
    assert scope["render_one"](None, None, "cuda:0", use_fallback=True) == "debug"


def test_source_indices_preserve_duplicate_positions_and_validate_hash(helpers, tmp_path):
    class Model:
        xyz = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)

        def __len__(self):
            return 2

    source = tmp_path / "source.ply"
    source.write_bytes(b"source")
    np.save(tmp_path / "indices.npy", np.array([1, 0], dtype=np.int64))
    (tmp_path / "layer.ply").write_bytes(b"delta")
    entry = {
        "object_id": 0,
        "dir_relative_to_outdir": ".",
        "ranked_source_indices": "indices.npy",
        "ranked_source_indices_sha256": hashlib.sha256(
            (tmp_path / "indices.npy").read_bytes()
        ).hexdigest(),
        "n_total": 2,
        "n_source_gaussians": 2,
        "layers": [{"file": "layer.ply", "size_bytes": 5}],
    }
    manifest = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "objects": [entry],
    }
    found = helpers["load_ranked_source_indices"](Model(), manifest, str(tmp_path), str(source))
    assert found[0].tolist() == [1, 0]
    entry["ranked_source_indices_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        helpers["load_ranked_source_indices"](Model(), manifest, str(tmp_path), str(source))


def test_matrix_orbit_first_frame_exact_and_centroid_preserved():
    import math

    path = ROOT / "publisher" / "make_cluster_orbit_trajectory.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "synthesise_orbit_trajectory"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"np": np, "math": math}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    view = np.eye(4)
    view[:3, 3] = [-1, -2, -3]
    camera = {
        "format": "opengl_matrix",
        "view_matrix": view.tolist(),
        "centroid": [0, 0, 0],
        "fov": 90,
        "fov_angles": [-0.7, 0.7, -0.7, 0.7],
        "width": 800,
        "height": 600,
    }
    frames = scope["synthesise_orbit_trajectory"](camera, 5, 30, 30)
    np.testing.assert_array_equal(frames[0]["view_matrix"], view)
    assert frames[0]["fov_angles"] == camera["fov_angles"]
    assert np.linalg.norm(frames[1]["camera_position"]) == pytest.approx(np.sqrt(14))
    assert not np.allclose(frames[1]["view_matrix"], view)
