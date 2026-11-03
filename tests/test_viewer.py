"""Browser controls and spatial partition coverage without a CUDA dependency."""

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from moq3dgs.scene.partition import spatial_groups
from moq3dgs.viewer import FrameHub, PoseController, start_viewer


def trace():
    return [
        {
            "timestamp_ms": 0,
            "camera_position": [0, 0, 0],
            "view_matrix": np.eye(4).tolist(),
            "fov": 60,
        }
    ]


def test_camera_controls_preserve_rigid_transform_and_reject_nan():
    control = PoseController(trace())
    control.input({"forward": 1, "yaw": 0.2, "pitch": -0.1})
    pose = control.sample(100)
    view = np.array(pose["view_matrix"])
    np.testing.assert_allclose(view @ [*pose["camera_position"], 1], [0, 0, 0, 1], atol=1e-8)
    np.testing.assert_allclose(view[:3, :3] @ view[:3, :3].T, np.eye(3), atol=1e-8)
    assert pose["camera_position"] == [0.0, 0.0, -1.0]
    with pytest.raises(ValueError):
        control.input({"yaw": float("nan")})
    control.input({"trace": True})
    assert control.sample(100)["camera_position"] == [0, 0, 0]


def test_local_viewer_serves_frames_controls_and_budget():
    hub = FrameHub(PoseController(trace()))
    hub.publish(np.full((4, 4, 3), 90, dtype=np.uint8), {"received_layers": 1})
    server = start_viewer(hub, 0)
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert b"MoQSplat" in urllib.request.urlopen(url, timeout=2).read()
        status = json.load(urllib.request.urlopen(url + "/status", timeout=2))
        assert status["frames"] == 1
        for path, body in [("/input", {"right": 1}), ("/budget", {"mib": 256})]:
            request = urllib.request.Request(
                url + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
            )
            assert json.load(urllib.request.urlopen(request, timeout=2))["accepted"]
        assert hub.pending_budget == 256
        assert hub.controller.sample(0)["camera_position"] == [1.0, 0.0, 0.0]
        with urllib.request.urlopen(url + "/stream", timeout=2) as stream:
            assert stream.read(7) == b"--frame"
    finally:
        with hub.condition:
            hub.stopped = True
            hub.condition.notify_all()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("count,maximum", [(1, 5), (105, 8), (1000, 77)])
def test_spatial_groups_preserve_every_original_gaussian(count, maximum):
    means = np.random.default_rng(1).normal(size=(count, 3))
    groups = spatial_groups(means, maximum)
    assert all(0 < len(g) <= maximum for g in groups)
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(count))


def test_incomplete_black_view_holds_last_useful_image_with_explicit_status():
    hub = FrameHub(PoseController(trace()))
    useful = np.full((4, 4, 3), 90, dtype=np.uint8)
    hub.publish(useful, {"received_layers": 1, "total_layers": 3, "camera_position": [0, 0, 0]})
    previous = hub.jpeg
    result = hub.publish(
        np.zeros_like(useful),
        {
            "received_layers": 2,
            "total_layers": 3,
            "camera_position": [1, 0, 0],
            "nonblack_pixels": 0,
        },
    )
    assert hub.jpeg == previous
    assert result["display_held_previous"]
    assert result["displayed_camera_position"] == [0, 0, 0]
    assert hub.status["camera_position"] == [1, 0, 0]
    assert hub.status["nonblack_pixels"] == 0
    assert "Waiting for scene data" in hub.status["state"]
    assert result["viewer_held_frames"] == 1
    # A fully received empty view is legitimate; it must not retain a stale pose.
    result = hub.publish(np.zeros_like(useful), {"received_layers": 3, "total_layers": 3})
    assert not result["display_held_previous"]
    assert hub.jpeg != previous
    assert "state" not in hub.status
    hub.publish(useful, {"received_layers": 3, "total_layers": 3})
    assert hub.jpeg == previous


def test_no_useful_frame_is_not_reported_as_a_held_image():
    hub = FrameHub(PoseController(trace()))
    result = hub.publish(
        np.zeros((4, 4, 3), dtype=np.uint8),
        {
            "received_layers": 1,
            "total_layers": 3,
        },
    )
    assert not result["display_held_previous"]
    assert result["viewer_held_frames"] == 0
    assert "Waiting for enough scene data" in hub.status["state"]
