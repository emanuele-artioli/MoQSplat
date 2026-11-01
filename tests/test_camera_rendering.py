"""Camera and rendering contracts independent of the original trace's pose."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from moq3dgs.render import rasterizer
from moq3dgs.viewport.trace import camera_forward_from_view_matrix


def test_world_forward_transforms_to_camera_negative_z():
    angle = np.deg2rad(63)
    view = np.eye(4)
    view[:3, :3] = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    view[:3, 3] = [2, -3, 5]
    forward = camera_forward_from_view_matrix(view.tolist())
    np.testing.assert_allclose(view[:3, :3] @ forward, [0, 0, -1], atol=1e-12)


def test_gsplat_receives_complete_sh_and_converted_camera(monkeypatch):
    observed = {}

    def render(**kwargs):
        observed.update(kwargs)
        return (torch.ones((1, 2, 3, 3)), torch.ones((1, 2, 3, 1)))

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rasterization=render))
    sh = torch.arange(48, dtype=torch.float32).reshape(1, 48)
    image = rasterizer._render_gsplat(
        torch.zeros((1, 3)),
        torch.zeros((1, 1)),
        sh,
        torch.zeros((1, 3)),
        torch.tensor([[1.0, 0, 0, 0]]),
        torch.eye(4),
        torch.eye(3),
        3,
        2,
        torch.zeros(3),
        "cpu",
    )
    assert observed["sh_degree"] == 3
    torch.testing.assert_close(observed["colors"], sh.reshape(1, 16, 3))
    torch.testing.assert_close(observed["viewmats"][0], torch.diag(torch.tensor([1.0, -1, -1, 1])))
    assert image.shape == (2, 3, 3)


def test_background_uses_returned_alpha_with_packed_renderer(monkeypatch):
    def render(**kwargs):
        assert "backgrounds" not in kwargs
        return (torch.zeros((1, 2, 3, 3)), torch.full((1, 2, 3, 1), 0.5))

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rasterization=render))
    image = rasterizer._render_gsplat(
        torch.zeros((1, 3)),
        torch.zeros((1, 1)),
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.tensor([[1.0, 0, 0, 0]]),
        torch.eye(4),
        torch.eye(3),
        3,
        2,
        torch.ones(3),
        "cpu",
    )
    assert torch.all(image == 127)


def test_cuda_failure_is_explicit_and_cannot_fall_back(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("native CUDA kernel failed")

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA request silently reached CPU fallback")

    monkeypatch.setattr(rasterizer, "_render_gsplat", broken)
    monkeypatch.setattr(rasterizer, "_render_cpu_fallback", forbidden)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        rasterizer.render_gaussians(
            torch.zeros((1, 3)),
            torch.zeros((1, 1)),
            torch.zeros((1, 3)),
            torch.zeros((1, 3)),
            torch.tensor([[1.0, 0, 0, 0]]),
            torch.eye(4),
            torch.eye(4),
            torch.zeros(3),
            bg_color=torch.zeros(3),
            device="cuda:0",
        )
