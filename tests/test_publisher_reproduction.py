"""Guard tests for the offline publisher wrapper; no publisher execution."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_publisher_reproduction.py"


@pytest.fixture
def inputs(tmp_path):
    paths = {}
    for name in ("scene.ply", "camera.json", "weights.pt"):
        path = tmp_path / name
        path.write_text("input")
        paths[name] = path
    config = tmp_path / "config.json"
    config.write_text("[]")
    paths["config"] = config
    return paths


def invoke(tmp_path, inputs, *extra):
    argv = [
        sys.executable,
        str(SCRIPT),
        "--scene",
        str(inputs["scene.ply"]),
        "--camera-json",
        str(inputs["camera.json"]),
        "--weights",
        str(inputs["weights.pt"]),
        "--config",
        str(inputs["config"]),
        "--output",
        str(tmp_path / "output"),
        "--cache",
        str(tmp_path / "cache"),
        *extra,
    ]
    return subprocess.run(argv, text=True, capture_output=True, check=False)


def test_missing_weight_names_expected_path(tmp_path, inputs):
    missing = tmp_path / "missing.pt"
    result = invoke(tmp_path, inputs, "--weights", str(missing))
    assert result.returncode != 0
    assert str(missing) in result.stderr
    assert not (tmp_path / "output").exists()


def test_existing_output_is_preserved(tmp_path, inputs):
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "evidence"
    marker.write_text("untouched")
    result = invoke(tmp_path, inputs)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert marker.read_text() == "untouched"


def test_output_in_checkout_rejected(tmp_path, inputs):
    result = invoke(tmp_path, inputs, "--output", str(ROOT / "not-created-publisher-test"))
    assert result.returncode != 0
    assert "outside the checkout" in result.stderr
    assert not (ROOT / "not-created-publisher-test").exists()


@pytest.mark.parametrize(
    "token", ["--input", "--camera_json=x", "--yolo_model=x", "--outdir=x", "--device=x", "--inp=x"]
)
def test_reserved_argument_rejected(tmp_path, inputs, token):
    inputs["config"].write_text(json.dumps([token]))
    result = invoke(tmp_path, inputs)
    assert result.returncode != 0
    assert "reserved argument" in result.stderr
    assert not (tmp_path / "output").exists()
