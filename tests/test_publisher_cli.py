"""Exercise actual optional publisher imports, not extracted camera definitions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script",
    ["gsplat_clustering_vmaf.py", "make_cluster_orbit_trajectory.py", "render_layer_videos.py"],
)
def test_released_publisher_cli_imports(script, tmp_path):
    pytest.importorskip("cv2", reason="requires publisher extra")
    pytest.importorskip("ultralytics", reason="requires publisher extra")
    path = Path(__file__).resolve().parents[1] / "publisher" / script
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=tmp_path,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "YOLO_CONFIG_DIR": str(tmp_path / "yolo"),
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
        },
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
