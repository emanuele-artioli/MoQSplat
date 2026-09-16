"""
make_cluster_orbit_trajectory.py
================================

Reads the segmentation output for one scene and writes a JSON trajectory file
describing a SMOOTH 150-frame orbit around a chosen cluster. The FIRST frame
is identical to the cluster's preview camera (the same viewpoint the paper
figure was rendered from) and the remaining 149 frames gently orbit around
the cluster centroid.

The output JSON uses the LEGACY format -- a list of frames each with
{angle, elevation, x, y, z, fx, fy, cx, cy, width, height} -- so it can be
consumed by `load_trajectory_as_movements()` (which auto-detects the format)
and by any tool that already reads user1_bicycle.json / user1_room.json.

Used by:
    render_layer_videos.py --camera_json <generated.json> ...

CRITICAL: this script does NOT re-run YOLO or recompute clusters. It just
reads the catalog/manifest the segmentation pipeline already wrote, and
synthesises the orbit using the cluster's pre-computed `preview_camera`.

USAGE
-----
    python make_cluster_orbit_trajectory.py \
        --manifest out_bicycle/manifest.json \
        --out_json out_bicycle/cluster_orbit_traj.json \
        --duration_s 5 --fps 30 \
        --target_class bicycle           # optional; auto-pick if absent
        --orbit_amplitude_deg 15         # peak ± azimuth swing (default 15)

TARGET CLUSTER SELECTION
------------------------
- If --target_class is given, picks the FIRST cluster that contains an
  object of that class. This is the right choice for paper figures where
  the user wants "the cluster containing the bicycle" or "the cluster
  containing the couch".
- Otherwise, picks the cluster with the most member objects (singletons
  last), breaking ties by total gaussian count. This favours visually rich
  multi-object clusters.

ORBIT CAMERA MOTION
-------------------
The reference camera has azimuth A0. The orbit sweeps to A0 + amplitude
over the first half, then back through A0 to A0 - amplitude in the second
half, then back to A0. We use a cosine schedule so the motion is smooth at
the boundary (zero angular velocity at start, midpoint, end) -- this avoids
jitter on the first/last few VMAF frames and produces an aesthetically
pleasing video.

The camera distance and elevation are kept FIXED to the reference values,
so the cluster stays well-framed throughout. Position is recomputed each
frame so the camera continues to look at the cluster centroid.
"""

import argparse
import json
import math
import os
import sys
from typing import Optional, List, Dict, Any
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
os.environ["MAX_JOBS"] = "1"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsplat_object_segmentation import cam_axes_from_euler   # noqa: E402


def _pick_target_cluster(catalog: dict,
                          target_class: Optional[str] = None) -> dict:
    """Pick the cluster that the orbit should circle. See module docstring."""
    clusters = catalog.get("clusters", [])
    if not clusters:
        raise RuntimeError("clusters_catalog has zero clusters; nothing to orbit")

    # Filter out the 'other' background cluster (label contains 'other' and
    # has no object_ids) -- it's huge and would not make a useful orbit.
    eligible = [c for c in clusters
                if c.get("object_ids") and c.get("label", "") != "other"]
    if not eligible:
        # all clusters are 'other'; fall back to the largest cluster anyway
        eligible = clusters

    if target_class:
        tc_lower = target_class.lower()
        matched = [c for c in eligible
                    if any(cls.lower() == tc_lower
                            for cls in c.get("object_classes", []))]
        if matched:
            # Prefer the cluster with the most members of target_class, then
            # most total gaussians.
            matched.sort(key=lambda c: (
                -sum(1 for cls in c["object_classes"]
                     if cls.lower() == tc_lower),
                -c.get("n_gaussians", 0)),)
            print(f"[orbit] --target_class='{target_class}' -> "
                  f"cluster {matched[0]['cluster_id']} "
                  f"[{matched[0]['label']}] with "
                  f"classes={matched[0]['object_classes']}")
            return matched[0]
        print(f"[orbit] WARN: --target_class='{target_class}' not found "
              f"in any cluster; falling back to auto-selection")

    # Auto-pick: most member objects, then most gaussians.
    eligible.sort(key=lambda c: (-c.get("n_objects", 0),
                                  -c.get("n_gaussians", 0)))
    print(f"[orbit] auto-picked cluster {eligible[0]['cluster_id']} "
          f"[{eligible[0]['label']}] with {eligible[0]['n_objects']} "
          f"objects and {eligible[0]['n_gaussians']:,} gaussians "
          f"(classes={eligible[0]['object_classes']})")
    return eligible[0]


def _orbit_position(centroid: np.ndarray, distance: float,
                     azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Position the camera at `distance` from `centroid` along the inverted
    forward axis of a camera with the given azimuth/elevation. Identical to
    the math in `build_camera_looking_at` so the first frame reproduces the
    exact same viewpoint."""
    _, _, fwd = cam_axes_from_euler(azimuth_deg, elevation_deg)
    return centroid - distance * fwd


def synthesise_orbit_trajectory(preview_cam: dict,
                                 n_frames: int,
                                 amplitude_deg: float,
                                 fps: int) -> List[Dict[str, Any]]:
    """Build a list of legacy-format trajectory frames whose first entry
    matches `preview_cam` exactly. Frames orbit around the cluster centroid
    with a cosine schedule peaking at +amplitude_deg and -amplitude_deg
    (then back to 0).

    Cosine schedule, normalised to t in [0, 1]:
        delta_az(t) = amplitude_deg * sin(2 * pi * t)
    Gives:
        delta_az(0) = 0  (exact match with preview)
        delta_az(0.25) = +amplitude_deg
        delta_az(0.5) = 0
        delta_az(0.75) = -amplitude_deg
        delta_az(1) = 0
    Smooth at t=0 (derivative = 2*pi*amp*cos(0) = nonzero -- but small for
    small amplitude). For paper-grade smoothness we use a HANN-WEIGHTED sin
    so the angular velocity also goes to zero at the boundaries:
        delta_az(t) = amplitude_deg * sin(2*pi*t) * 0.5 * (1 - cos(2*pi*t))
    That's smoothly zero AND has zero velocity at t=0 and t=1, eliminating
    any "snap" at the start of the video.
    """
    if n_frames < 2:
        raise ValueError(f"need at least 2 frames, got {n_frames}")

    centroid = np.asarray(preview_cam["centroid"], dtype=np.float64)
    base_az  = float(preview_cam["angle"])
    base_el  = float(preview_cam["elevation"])

    # Recover the original camera-to-centroid distance from the stored
    # position. This guarantees the same framing as the preview, even if
    # someone later changes fit_fraction.
    pos = np.array([preview_cam["x"], preview_cam["y"], preview_cam["z"]],
                    dtype=np.float64)
    distance = float(np.linalg.norm(centroid - pos))
    if distance < 1e-6:
        raise RuntimeError("preview camera is AT the centroid; cannot orbit")
    print(f"[orbit] centroid={centroid.tolist()}  distance={distance:.3f}  "
          f"base_az={base_az:.2f}  base_el={base_el:.2f}  amp={amplitude_deg}")

    frame_ms = int(round(1000.0 / fps))
    frames: List[Dict[str, Any]] = []
    for i in range(n_frames):
        t = i / (n_frames - 1)             # 0..1 inclusive
        hann_window = 0.5 * (1.0 - math.cos(2 * math.pi * t))
        delta_az = amplitude_deg * math.sin(2 * math.pi * t) * hann_window

        az = base_az + delta_az
        el = base_el
        new_pos = _orbit_position(centroid, distance, az, el)

        frames.append({
            "tMs":         i * frame_ms,
            "durationMs":  frame_ms,
            "angle":       float(az),
            "elevation":   float(el),
            "x":           float(new_pos[0]),
            "y":           float(new_pos[1]),
            "z":           float(new_pos[2]),
            "fx":          float(preview_cam["fx"]),
            "fy":          float(preview_cam["fy"]),
            "cx":          float(preview_cam["cx"]),
            "cy":          float(preview_cam["cy"]),
            "width":       int(preview_cam["width"]),
            "height":      int(preview_cam["height"]),
            "profile":     0,
            # Bookkeeping (ignored by load_trajectory_as_movements but useful
            # for inspection -- e.g. checking that frame 0 didn't drift).
            "_orbit_delta_az_deg": float(delta_az),
        })

    # Sanity assertion: frame 0 should be byte-identical to the preview camera
    # (up to the new tMs/_orbit_delta_az_deg fields).
    f0 = frames[0]
    assert abs(f0["angle"]     - base_az) < 1e-9, "frame 0 az drift"
    assert abs(f0["elevation"] - base_el) < 1e-9, "frame 0 el drift"
    assert abs(f0["x"] - preview_cam["x"]) < 1e-5, "frame 0 x drift"
    assert abs(f0["y"] - preview_cam["y"]) < 1e-5, "frame 0 y drift"
    assert abs(f0["z"] - preview_cam["z"]) < 1e-5, "frame 0 z drift"
    print(f"[orbit] {n_frames} frames synthesised, frame 0 == preview camera")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="top-level manifest.json from segmentation run "
                         "(must have clustering.enabled=true)")
    ap.add_argument("--out_json", required=True,
                    help="where to write the synthesised trajectory JSON")
    ap.add_argument("--duration_s", type=float, default=5.0,
                    help="video duration in seconds (default 5)")
    ap.add_argument("--fps", type=int, default=30,
                    help="frames per second (default 30)")
    ap.add_argument("--target_class", default=None,
                    help="prefer the cluster containing this class name "
                         "(e.g. 'bicycle' or 'couch'). If absent or no "
                         "match, auto-picks the largest cluster.")
    ap.add_argument("--orbit_amplitude_deg", type=float, default=15.0,
                    help="peak +/- azimuth sweep in degrees (default 15)")
    ap.add_argument("--catalog", default=None,
                    help="path to clusters_catalog.json. Default: looks for "
                         "it in the same dir as --manifest.")
    args = ap.parse_args()

    n_frames = int(round(args.duration_s * args.fps))
    print(f"[orbit] target: {args.duration_s}s @ {args.fps} fps = "
          f"{n_frames} frames")

    catalog_path = (args.catalog or
                    os.path.join(os.path.dirname(args.manifest),
                                  "clusters_catalog.json"))
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"clusters_catalog.json not found at {catalog_path}. "
            f"Run gsplat_object_segmentation.py with --clustering first.")
    with open(catalog_path) as f:
        catalog = json.load(f)
    print(f"[orbit] loaded catalog from {catalog_path} "
          f"({catalog['n_clusters']} clusters)")

    target = _pick_target_cluster(catalog, args.target_class)
    if not target.get("preview_camera"):
        raise RuntimeError(
            f"cluster {target['cluster_id']} has no 'preview_camera' field. "
            f"Re-run segmentation with this updated version of "
            f"gsplat_object_segmentation.py (it now persists the camera "
            f"into the catalog).")

    frames = synthesise_orbit_trajectory(
        preview_cam=target["preview_camera"],
        n_frames=n_frames,
        amplitude_deg=args.orbit_amplitude_deg,
        fps=args.fps,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".",
                 exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(frames, f, indent=2)
    print(f"[orbit] wrote {len(frames)} frames -> {args.out_json}")
    print(f"[orbit] target cluster: id={target['cluster_id']} "
          f"label={target['label']} classes={target['object_classes']}")


if __name__ == "__main__":
    main()
