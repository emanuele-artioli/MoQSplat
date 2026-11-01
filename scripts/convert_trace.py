#!/usr/bin/env python3
"""Convert raw EyeNavGS OpenXR CSV poses to scene-world OpenGL view matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
FLIP = np.diag([1.0, -1.0, -1.0])


def quaternion_to_view_matrix(pos, quat_xyzw):
    """Map scene-world coordinates into the raw OpenXR camera basis."""
    position = np.asarray(pos, dtype=float)
    quat = np.asarray(quat_xyzw, dtype=float)
    if not np.isfinite(position).all() or not np.isfinite(quat).all():
        raise ValueError("Position and quaternion must be finite")
    rotation = Rotation.from_quat(quat).as_matrix()
    view = np.eye(4)
    view[:3, :3] = rotation.T @ FLIP
    view[:3, 3] = -rotation.T @ position
    return view.tolist()


def convert_csv_to_json(csv_path, json_path, eye=0):
    source = Path(csv_path).resolve(strict=True)
    output_arg = Path(json_path)
    if output_arg.exists() or output_arg.is_symlink():
        raise FileExistsError(f"Output already exists: {output_arg}")
    output = output_arg.resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError(f"Output must be outside the checkout: {output}")
    if eye not in (0, 1):
        raise ValueError("eye must be 0 or 1")
    frames = []
    with source.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["ViewIndex"]) != eye:
                continue
            pos = np.array([float(row[f"Position{x}"]) for x in "XYZ"])
            quat = [float(row[f"Quaternion{x}"]) for x in "XYZW"]
            angles = np.array([float(row[f"FOV{i}"]) for i in range(1, 5)])
            left, right, down, up = angles
            if (
                not np.isfinite(angles).all()
                or np.any(np.abs(angles) >= np.pi / 2)
                or left >= right
                or down >= up
            ):
                raise ValueError(f"Invalid signed OpenXR FOV angles: {angles.tolist()}")
            frames.append(
                {
                    "timestamp_ms": int(row["Timestamp"]),
                    "camera_position": (FLIP @ pos).tolist(),
                    "view_matrix": quaternion_to_view_matrix(pos, quat),
                    "fov": float(np.degrees(up - down)),
                    "fov_angles": angles.tolist(),
                }
            )
    if not frames:
        raise ValueError(f"No frames found for eye {eye}")
    metadata = {
        "source_csv": str(source),
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "eyenav_software_commit": "88cbd793e806e1a9ae23fa7d5b93d3b3b3ac78d2",
        "eyenav_dataset_commit": "7f264fda0223cbc74e191f115ef9fd3b580d8cf4",
        "quaternion_order": "xyzw",
        "camera_basis": "OpenXR/OpenGL: +X right, +Y up, -Z forward",
        "world_basis": "scene world = diag(1,-1,-1) @ recorded OpenXR world",
        "selected_eye": eye,
        "fov_angle_order": ["left", "right", "down", "up"],
        "fov_angles_units": "radians, signed",
        "asymmetric_projection": "per-frame signed angles preserved in fov_angles",
    }
    with output.open("x") as f:
        json.dump({"metadata": metadata, "frames": frames}, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Converted {len(frames)} frames to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_json")
    parser.add_argument("--eye", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    convert_csv_to_json(args.input_csv, args.output_json, args.eye)


if __name__ == "__main__":
    main()
