"""PLY scene loader for pre-trained 3D Gaussian Splatting models.

Reads the standard 3DGS .ply format produced by the original
Kerbl et al. training code and returns structured torch tensors for
downstream spatial partitioning and MoQ packetisation.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData

from moq3dgs.decorators import gpu_bound


@dataclass
class GaussianScene:
    """In-memory representation of a full 3DGS scene.

    All tensors live on the *CPU* after loading; callers move them to GPU
    as needed.  The SH coefficients are stored in two tiers:
      - ``sh_dc``  (N, 1, 3): zeroth-order (DC) colour — always present.
      - ``sh_rest`` (N, K, 3): higher-order bands (SH1-SH3) — may be empty
        for base-only exports.
    """

    means: torch.Tensor          # (N, 3)   float32
    opacities: torch.Tensor      # (N, 1)   float32  (logit space)
    scales: torch.Tensor         # (N, 3)   float32  (log space)
    rotations: torch.Tensor      # (N, 4)   float32  (quaternion wxyz)
    sh_dc: torch.Tensor          # (N, 1, 3)
    sh_rest: torch.Tensor        # (N, K, 3)   K = (max_sh_degree+1)^2 - 1
    num_gaussians: int

    @property
    def bbox_min(self) -> torch.Tensor:
        """Axis-aligned bounding box minimum corner."""
        return self.means.min(dim=0).values

    @property
    def bbox_max(self) -> torch.Tensor:
        """Axis-aligned bounding box maximum corner."""
        return self.means.max(dim=0).values


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid — used to convert opacity logits."""
    return 1.0 / (1.0 + np.exp(-x))


def load_ply(path: str | Path, max_sh_degree: int = 3) -> GaussianScene:
    """Load a 3DGS .ply file into a :class:`GaussianScene`.

    This mirrors the official 3DGS ``GaussianModel.load_ply`` logic so that
    any checkpoint exported by the original training code is compatible.

    Args:
        path: Filesystem path to the ``point_cloud.ply`` file.
        max_sh_degree: Maximum spherical-harmonics degree stored in the file.
            Standard 3DGS uses degree 3 → 16 SH coefficients per colour
            channel.

    Returns:
        A :class:`GaussianScene` with all attributes on CPU.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the PLY is missing required vertex properties.
    """
    ply_path = Path(path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]
    n = vertex.count

    # -- Positions (x, y, z) -------------------------------------------------
    means = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=-1
    ).astype(np.float32)

    # -- Opacities ------------------------------------------------------------
    opacities = vertex["opacity"].reshape(-1, 1).astype(np.float32)

    # -- Scales (log space) ---------------------------------------------------
    scales = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1
    ).astype(np.float32)

    # -- Rotations (quaternion) -----------------------------------------------
    rotations = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
        axis=-1,
    ).astype(np.float32)

    # -- Spherical harmonics --------------------------------------------------
    num_sh_coeffs = (max_sh_degree + 1) ** 2  # 16 for degree 3
    sh_dc = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
    ).astype(np.float32).reshape(n, 1, 3)

    # Standard 3DGS exports coefficients channel-major: R, then G, then B.
    names = vertex.data.dtype.names or ()
    indices = sorted(int(name[7:]) for name in names
                     if name.startswith("f_rest_") and name[7:].isdigit())
    if indices:
        if indices != list(range(len(indices))) or len(indices) % 3:
            raise ValueError("PLY f_rest properties must be contiguous and divisible by 3")
        stored_per_channel = len(indices) // 3
        keep = min(num_sh_coeffs - 1, stored_per_channel)
        sh_rest = np.stack([vertex[f"f_rest_{idx}"].astype(np.float32)
                            for idx in indices], axis=-1)
        sh_rest = sh_rest.reshape(n, 3, stored_per_channel)[:, :, :keep].transpose(0, 2, 1).copy()
    else:
        sh_rest = np.zeros((n, 0, 3), dtype=np.float32)

    return GaussianScene(
        means=torch.from_numpy(means),
        opacities=torch.from_numpy(opacities),
        scales=torch.from_numpy(scales),
        rotations=torch.from_numpy(rotations),
        sh_dc=torch.from_numpy(sh_dc),
        sh_rest=torch.from_numpy(sh_rest),
        num_gaussians=n,
    )


def find_ply_in_scene_dir(scene_dir: str | Path) -> Path:
    """Locate the ``point_cloud.ply`` inside a standard 3DGS scene directory.

    The official layout is ``<scene>/point_cloud/iteration_<N>/point_cloud.ply``.
    We pick the highest iteration available; if the path is already a .ply
    file we return it directly.

    Args:
        scene_dir: Path to the scene root, or directly to a .ply file.

    Returns:
        Resolved path to the .ply file.

    Raises:
        FileNotFoundError: If no .ply is found.
    """
    p = Path(scene_dir)
    if p.suffix == ".ply" and p.is_file():
        return p

    pc_dir = p / "point_cloud"
    if not pc_dir.is_dir():
        raise FileNotFoundError(
            f"No point_cloud/ directory found under {p}"
        )

    iteration_dirs = sorted(
        [d for d in pc_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")],
        key=lambda d: int(d.name.split("_")[-1]),
        reverse=True,
    )
    for d in iteration_dirs:
        candidate = d / "point_cloud.ply"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"No point_cloud.ply found under {pc_dir}")
