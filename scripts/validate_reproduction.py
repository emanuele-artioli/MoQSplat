#!/usr/bin/env python3
"""Validate real-scene CUDA rendering and wire roundtrip, without CPU fallback.

Input matrices are world-to-camera. OpenCV inputs are converted explicitly to
OpenGL by left-multiplication with diag(1, -1, -1, 1).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import moq3dgs.render.rasterizer as rasterizer
from moq3dgs.scene.loader import find_ply_in_scene_dir, load_ply
from moq3dgs.transport.protocol import decode_cluster, encode_cluster
from moq3dgs.viewport.frustum import projection_matrix_from_fov
from moq3dgs.viewport.trace import load_trace, replay_trace


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def save_summary(path: Path, summary: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-convention", choices=["opengl", "opencv"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    summary = {
        "valid": False,
        "command": sys.argv,
        "camera_convention": args.camera_convention,
        "frames_requested": args.frames,
        "frames": [],
    }
    save_summary(summary_path, summary)
    try:
        if args.frames < 1:
            raise ValueError("--frames must be positive")
        scene_path = find_ply_in_scene_dir(args.scene).resolve()
        trace_path = args.trace.resolve(strict=True)
        summary["inputs"] = {
            "scene": str(scene_path),
            "scene_sha256": sha256(scene_path),
            "trace": str(trace_path),
            "trace_sha256": sha256(trace_path),
        }
        summary["versions"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gsplat": package_version("gsplat"),
            "numpy": np.__version__,
            "pillow": package_version("Pillow"),
            "git_revision": command("git", "rev-parse", "HEAD"),
            "git_status": command("git", "status", "--short"),
            "nvidia_smi": command(
                "nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"
            ),
        }
        summary["code_sha256"] = {
            str(Path(module.__file__).resolve()): sha256(Path(module.__file__))
            for module in (rasterizer,)
        }
        summary["code_sha256"][str(Path(__file__).resolve())] = sha256(Path(__file__))
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("A working CUDA device is required")
        torch.cuda.set_device(device)
        summary["device_type"] = device.type
        props = torch.cuda.get_device_properties(device)
        summary["gpu"] = {
            "device": str(device),
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "uuid": str(getattr(props, "uuid", "unavailable")),
        }
        save_summary(summary_path, summary)
        scene = load_ply(scene_path)
        summary["gaussian_count"] = scene.num_gaussians
        if scene.num_gaussians == 0:
            raise ValueError("Scene contains no Gaussians")
        sh = torch.cat((scene.sh_dc, scene.sh_rest), dim=1).reshape(scene.num_gaussians, -1)
        n = min(scene.num_gaussians, 16384)
        attributes = {
            "means": scene.means[:n],
            "opacities": scene.opacities[:n],
            "sh_coeffs": sh[:n],
            "scales": scene.scales[:n],
            "rotations": scene.rotations[:n],
        }
        wire = encode_cluster("validation", "shard", 0, 1, n, **attributes)
        decoded = decode_cluster(wire)
        if (
            decoded["track_id"],
            decoded["group_id"],
            decoded["subgroup_id"],
            decoded["object_id"],
            decoded["num_gaussians"],
        ) != ("validation", "shard", 0, 1, n):
            raise AssertionError("wire metadata mismatch")
        for name, tensor in attributes.items():
            if decoded[name].dtype != torch.float32 or not torch.equal(tensor, decoded[name]):
                raise AssertionError(f"wire roundtrip mismatch: {name}")
        summary["wire_roundtrip_exact"] = True
        summary["wire"] = {
            "first_n_gaussians": n,
            "bytes": len(wire),
            "sha256": hashlib.sha256(wire).hexdigest(),
            "attribute_bytes": {k: v.numel() * v.element_size() for k, v in attributes.items()},
        }
        frames = load_trace(trace_path)
        if args.frames > len(frames):
            raise ValueError(f"Requested {args.frames} frames but trace has {len(frames)}")
        summary["trace_frame_count"] = len(frames)
        summary["first_trace_entry"] = frames[0]
        save_summary(summary_path, summary)

        def no_fallback(*_args, **_kwargs):
            raise RuntimeError("CPU fallback invoked")

        rasterizer._render_cpu_fallback = no_fallback
        conversion = (
            np.diag([1.0, -1.0, -1.0, 1.0]) if args.camera_convention == "opencv" else np.eye(4)
        )
        # Diagnostics use a bounded deterministic prefix, while rendering uses all splats.
        sample = scene.means[: min(scene.num_gaussians, 16384)].numpy().astype(np.float64)
        for index, update in enumerate(replay_trace(frames[: args.frames])):
            view = conversion @ np.asarray(update.view_matrix, dtype=np.float64)
            proj = projection_matrix_from_fov(
                update.fov, aspect=320 / 240, fov_angles=update.fov_angles,
            )
            camera = sample @ view[:3, :3].T + view[:3, 3]
            clip = np.column_stack((sample, np.ones(len(sample)))) @ (proj @ view).T
            visible = (clip[:, 3] > 0) & np.all(np.abs(clip[:, :3]) <= clip[:, 3:4], axis=1)
            camera_position = np.linalg.inv(view)[:3, 3]
            image = rasterizer.render_gaussians(
                means=scene.means,
                opacities=scene.opacities,
                sh_coeffs=sh,
                scales=scene.scales,
                rotations=scene.rotations,
                view_matrix=torch.tensor(view, dtype=torch.float32),
                proj_matrix=torch.tensor(proj, dtype=torch.float32),
                camera_pos=torch.tensor(camera_position, dtype=torch.float32),
                image_width=320,
                image_height=240,
                fov_y_deg=update.fov,
                fov_angles=update.fov_angles,
                device=str(device),
            )
            if image.dtype != torch.uint8 or tuple(image.shape) != (240, 320, 3):
                raise AssertionError(
                    f"Unexpected image dtype or shape: {image.dtype}, {tuple(image.shape)}"
                )
            if (
                not torch.isfinite(image).all()
                or image.max().item() == 0
                or image.min().item() == image.max().item()
            ):
                raise AssertionError(f"Frame {index} is nonfinite, black, or constant")
            output = args.output / f"frame_{index:04d}.png"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
            Image.fromarray(image.cpu().numpy()).save(output)
            summary["frames"].append(
                {
                    "index": index,
                    "timestamp_ms": update.timestamp_ms,
                    "fov": update.fov,
                    "path": str(output),
                    "sha256": sha256(output),
                    "min": int(image.min().item()),
                    "max": int(image.max().item()),
                    "mean": float(image.float().mean().item()),
                    "std": float(image.float().std().item()),
                    "sample_count": len(sample),
                    "sample_visible": int(visible.sum()),
                    "sample_z_negative_fraction": float((camera[:, 2] < 0).mean()),
                    "sample_z_positive_fraction": float((camera[:, 2] > 0).mean()),
                    "sample_z_quantiles": np.quantile(
                        camera[:, 2], [0, 0.25, 0.5, 0.75, 1]
                    ).tolist(),
                }
            )
            save_summary(summary_path, summary)
        summary["frames_rendered"] = len(summary["frames"])
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        summary["valid"] = True
        save_summary(summary_path, summary)
    except Exception as exc:
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_summary(summary_path, summary)
        raise


if __name__ == "__main__":
    main()
