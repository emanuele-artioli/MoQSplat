"""EyeNav pose and signed-FOV conversion regression tests."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from moq3dgs.viewport.trace import replay_trace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_trace.py"
spec = importlib.util.spec_from_file_location("convert_trace", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def write_csv(path, fov=(-0.942, 0.698, -0.942, 0.733)):
    fields = [
        "Timestamp",
        "ViewIndex",
        "PositionX",
        "PositionY",
        "PositionZ",
        "QuaternionX",
        "QuaternionY",
        "QuaternionZ",
        "QuaternionW",
        "FOV1",
        "FOV2",
        "FOV3",
        "FOV4",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for eye in (0, 1):
            writer.writerow([42, eye, 1 + eye, 2, 3, 0, 0, 0, 1, *fov])


def test_pose_transforms_raw_world_to_scene_world():
    pos = np.array([1.0, 2.0, 3.0])
    rotation = Rotation.from_euler("xyz", [0.3, -0.6, 0.8])
    view = np.array(converter.quaternion_to_view_matrix(pos, rotation.as_quat()))
    scene_point = converter.FLIP @ np.array([3.0, 5.0, -2.0])
    expected = rotation.as_matrix().T @ (np.array([3.0, 5.0, -2.0]) - pos)
    np.testing.assert_allclose(view @ np.r_[scene_point, 1.0], np.r_[expected, 1.0], atol=1e-7)
    np.testing.assert_allclose(view @ np.r_[converter.FLIP @ pos, 1.0], [0, 0, 0, 1], atol=1e-7)


@pytest.mark.parametrize("eye", [0, 1])
def test_fov_and_eye_selection(tmp_path, eye):
    source = tmp_path / "trace.csv"
    write_csv(source)
    output = tmp_path / "converted.json"
    converter.convert_csv_to_json(source, output, eye)
    data = json.loads(output.read_text())
    frame = data["frames"][0]
    assert data["metadata"]["selected_eye"] == eye
    assert frame["camera_position"] == [1 + eye, -2, -3]
    assert frame["fov"] == pytest.approx(95.97, abs=0.01)
    assert frame["fov"] != pytest.approx(np.degrees(0.698 + 0.733))
    assert frame["fov_angles"] == [-0.942, 0.698, -0.942, 0.733]
    assert list(replay_trace([frame]))[0].fov_angles == tuple(frame["fov_angles"])


def test_existing_output_never_overwritten(tmp_path):
    source, output = tmp_path / "trace.csv", tmp_path / "saved.json"
    write_csv(source)
    output.write_text("original")
    with pytest.raises(FileExistsError):
        converter.convert_csv_to_json(source, output)
    assert output.read_text() == "original"


@pytest.mark.parametrize(
    "angles",
    [
        (0.5, -0.5, -0.4, 0.4),
        (-0.5, 0.5, 0.4, -0.4),
        (-0.5, 0.5, float("nan"), 0.4),
        (-2, 0.5, -0.4, 0.4),
    ],
)
def test_invalid_fov_rejected(tmp_path, angles):
    source, output = tmp_path / "trace.csv", tmp_path / "new.json"
    write_csv(source, angles)
    with pytest.raises(ValueError, match="FOV"):
        converter.convert_csv_to_json(source, output)
    assert not output.exists()
