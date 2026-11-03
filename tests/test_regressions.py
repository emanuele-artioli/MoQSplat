"""Regression tests for SH loading and viewport subscriptions."""

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from moq3dgs.models import GroupInfo, SceneManifest, TrackInfo, ViewportUpdate
from moq3dgs.scene.loader import load_ply
from moq3dgs.transport.client import MoQClient


def test_standard_ply_sh_channel_order_and_truncation(tmp_path):
    fields = [
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    ]
    fields += [f"f_rest_{i}" for i in range(45)]
    data = np.zeros(1, dtype=[(name, "f4") for name in fields])
    for i in range(45):
        data[f"f_rest_{i}"] = i + 1
    path = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))
    for degree in (1, 3):
        result = load_ply(path, max_sh_degree=degree).sh_rest.numpy()[0]
        expected = np.arange(1, 46).reshape(3, 15)[:, : (degree + 1) ** 2 - 1].T
        np.testing.assert_array_equal(result, expected)


@pytest.mark.asyncio
async def test_subscriptions_progress_retry_and_deduplicate(tmp_path, monkeypatch):
    client = MoQClient(output_dir=tmp_path, device="cpu", render_enabled=False)
    bounds = dict(bbox_min=[-0.1, -0.1, -5.1], bbox_max=[0.1, 0.1, -4.9])
    group = GroupInfo(group_id="g", num_gaussians=1, **bounds)
    track = TrackInfo(track_id="t", groups=[group], **bounds)
    client.manifest = SceneManifest(broadcast_name="test", total_gaussians=1, tracks=[track])
    client._writer = object()
    sent = []

    async def send_json(message):
        if message["type"] == "subscribe":
            sent.append((message["data"]["subgroup_id"], message["data"]["quality_level"]))

    monkeypatch.setattr(client, "_send_json", send_json)
    update = ViewportUpdate(
        client_id="x",
        timestamp_ms=0,
        camera_position=[0, 0, 0],
        view_matrix=np.eye(4).tolist(),
        fov=90,
    )
    sequence = [(i, q) for q in (0, 1) for i in (0, 1, 2)]
    for sub, quality in sequence:
        await client.send_viewport_update(update)
        assert sent[-1] == (sub, quality)
    # No duplicates while requests are pending, including an absent subgroup.
    await client.send_viewport_update(update)
    assert sent == sequence
    for sub, quality in sequence:
        client.cache.put(
            dict(track_id="t", group_id="g", subgroup_id=sub, object_id=quality, num_gaussians=1)
        )
    await client.send_viewport_update(update)
    assert sent == sequence
    client.cache.evict("t", "g", 1, 0)
    await client.send_viewport_update(update)
    assert sent[-1] == (1, 0)
    client._pending_subscriptions[("t", "g", 1, 0)] -= client._subscription_retry_seconds + 1
    await client.send_viewport_update(update)
    assert sent[-2:] == [(1, 0), (1, 0)]
