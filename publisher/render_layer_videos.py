"""
render_layer_videos.py
======================

For VMAF rate-distortion evaluation: given an already-segmented scene (i.e.
gsplat_object_segmentation.py has been run and its outputs are on disk),
render the SAME camera trajectory at each progressive layer level, producing
PNG frame sequences that can be encoded to MP4 and fed to libvmaf.

CRITICAL: this script uses ONLY the trajectory frames from --camera_json.
It does NOT use derive_views(), lateral_offsets, or fov_scale -- those are
detection-time camera manipulations that have nothing to do with what the
client actually sees. Here we render exactly the viewpoints the user moved
their head through.

Output layout:
    <out_dir>/
        layer_0_base/        frame_000000.png, frame_000001.png, ...
        layer_1_enh1/        ...
        layer_2_enh2/        ...
        layer_3_enh3_full/   ...  <- last level (alias for the "full" level
                                     used as VMAF reference; 100% gaussians)
        reference/           ...  <- the full ORIGINAL .ply rendered through
                                     the trajectory with no mask. Same as the
                                     last layer in expectation; rendered
                                     separately for sanity.
        rates.json                <- per-level cumulative bytes (rate axis)

Mask reconstruction:
    For each object, we re-read its saved full_n*.ply to recover the boolean
    set of gaussian indices belonging to it (matched by float32 xyz bytes --
    the segmentation pipeline writes them without quantization so the bit
    patterns are preserved). Then within each object we deterministically
    re-partition into layers using the SAME importance ranking the pipeline
    used (recorded in manifest["rank_method"]), taking the top
    `cumulative_gaussians` gaussians for cumulative level k.

    This is byte-equivalent to loading every layer .ply back from disk and
    OR'ing them, but it's faster (one ply read per object instead of K) and
    avoids float32 equality edge cases between layers.

Usage:
    python render_layer_videos.py \
        --input bicycle.ply \
        --camera_json user1_bicycle.json \
        --manifest out_bicycle/manifest.json \
        --out_dir vmaf_input/bicycle \
        [--max_frames 60]      # render at most N frames (uniformly subsampled)
        [--device cuda]        # cuda or cpu
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
os.environ["MAX_JOBS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsplat_object_segmentation import (   # noqa: E402
    GaussianModel,
    load_gaussian_ply,
    load_trajectory_as_movements,
    render_one,
    _gaussian_importance,
    build_other_mask,
)


# ----------------------------------------------------------------------------
#  Recover per-object full masks from disk
# ----------------------------------------------------------------------------

def _build_xyz_hash(model: GaussianModel) -> Dict[bytes, int]:
    """One-shot hash table: 12-byte xyz tuple -> index into model. Used to
    look up which gaussian in the original model corresponds to each row
    of a saved per-object .ply. O(N) memory, ~N dict entries."""
    print(f"[mask-recon] building xyz hash for {len(model):,} gaussians...")
    xyz_bytes = model.xyz.astype(np.float32).tobytes()
    h = {}
    for i in range(len(model)):
        h[xyz_bytes[i*12:(i+1)*12]] = i
    return h


def _mask_from_ply_file(ply_path: str, model: GaussianModel,
                        xyz_to_idx: Dict[bytes, int]) -> np.ndarray:
    """Read a per-object .ply and return a boolean mask over the FULL model
    of length N. Matching is by exact xyz float32 byte equality."""
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    xs = np.asarray(v['x'], dtype=np.float32)
    ys = np.asarray(v['y'], dtype=np.float32)
    zs = np.asarray(v['z'], dtype=np.float32)
    out = np.zeros(len(model), dtype=bool)
    n_matched = 0
    # Use stacked bytes for vectorisation
    xyz_stack = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    raw = xyz_stack.tobytes()
    for i in range(len(xs)):
        key = raw[i*12:(i+1)*12]
        idx = xyz_to_idx.get(key)
        if idx is not None:
            out[idx] = True
            n_matched += 1
    print(f"[mask-recon]   {os.path.basename(ply_path)}: "
          f"matched {n_matched:,}/{len(xs):,}")
    return out


def recover_object_full_masks(model: GaussianModel, manifest: dict,
                              outdir: str) -> Dict[int, np.ndarray]:
    """For every object in manifest["objects"], reconstruct its full-mask
    boolean array (length N). Reads each object's full_n*.ply (or falls back
    to concatenating its layer files if the full file isn't present).

    Returns dict {object_id -> mask}. The 'other' (background) entry uses
    object_id = -1 by the segmentation script's convention.
    """
    xyz_to_idx = _build_xyz_hash(model)
    object_masks: Dict[int, np.ndarray] = {}
    for entry in manifest["objects"]:
        oid = entry["object_id"]
        sub = entry["dir_relative_to_outdir"]
        full_fname = entry.get("full_file")
        if full_fname:
            full_path = os.path.join(outdir, sub, full_fname)
            if os.path.exists(full_path):
                m = _mask_from_ply_file(full_path, model, xyz_to_idx)
                object_masks[oid] = m
                continue
        # Fallback: union of per-layer plys.
        print(f"[mask-recon] object_id={oid}: no full ply, reading layers...")
        m = np.zeros(len(model), dtype=bool)
        for L in entry["layers"]:
            p = os.path.join(outdir, sub, L["file"])
            if not os.path.exists(p):
                print(f"[mask-recon] WARN: missing {p}; skipping")
                continue
            m |= _mask_from_ply_file(p, model, xyz_to_idx)
        object_masks[oid] = m
    return object_masks


def cumulative_mask_for_object(model: GaussianModel,
                               full_mask: np.ndarray,
                               cumulative_gaussian_count: int,
                               rank_method: str) -> np.ndarray:
    """Recover the boolean mask of an object's top-K importance gaussians,
    where K = cumulative_gaussian_count and the ranking matches what the
    segmentation pipeline used when writing the layer files."""
    if cumulative_gaussian_count <= 0 or full_mask.sum() == 0:
        return np.zeros(len(model), dtype=bool)
    scores = _gaussian_importance(model, full_mask, rank_method=rank_method)
    obj_indices = np.where(full_mask)[0]
    order = np.argsort(-scores)   # descending by importance (matches pipeline)
    sorted_indices = obj_indices[order]
    k = min(cumulative_gaussian_count, len(sorted_indices))
    out = np.zeros(len(model), dtype=bool)
    out[sorted_indices[:k]] = True
    return out


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Render trajectory frames at each progressive layer level "
                    "for VMAF RD evaluation.")
    ap.add_argument("--input", required=True, help="original scene .ply")
    ap.add_argument("--camera_json", required=True,
                    help="CLIENT trajectory JSON (not the detection-time "
                         "synthetic views)")
    ap.add_argument("--manifest", required=True,
                    help="top-level manifest.json produced by "
                         "gsplat_object_segmentation.py")
    ap.add_argument("--out_dir", required=True,
                    help="where to write per-layer PNG frame sequences")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="render at most this many trajectory frames "
                         "(uniformly subsampled). Default: all frames.")
    ap.add_argument("--device", default=None,
                    help="cuda or cpu (auto-detected if not set)")
    ap.add_argument("--use_fallback_rasterizer", action="store_true",
                    help="force NumPy CPU rasterizer (debug only).")
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    print(f"[render-layers] device={args.device}")

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    print(f"[render-layers] manifest dir: {manifest_dir}")

    model = load_gaussian_ply(args.input)

    with open(args.manifest) as f:
        manifest = json.load(f)
    layer_pcts  = manifest["layer_percentages"]
    rank_method = manifest["rank_method"]
    n_layers    = len(layer_pcts)
    print(f"[render-layers] {n_layers} layers @ {layer_pcts}%, "
          f"rank='{rank_method}'")
    print(f"[render-layers] {len(manifest['objects'])} object/other entries")

    # Trajectory (NO detection-time manipulation: load directly from JSON).
    traj_mvs = load_trajectory_as_movements(args.camera_json,
                                             max_frames=args.max_frames)
    n_frames = len(traj_mvs)
    if n_frames == 0:
        raise RuntimeError(f"trajectory {args.camera_json} has no frames")
    print(f"[render-layers] trajectory: {n_frames} frames "
          f"(max_frames={args.max_frames})")
    print(f"[render-layers] viewport: {traj_mvs[0].width}x{traj_mvs[0].height}")

    # 1) Recover per-object full masks from saved .plys.
    print("[render-layers] reconstructing per-object full masks from disk...")
    obj_full_masks = recover_object_full_masks(model, manifest, manifest_dir)
    # 'other' may or may not be present in manifest; if not, derive it.
    if -1 not in obj_full_masks:
        non_other = {k: v for k, v in obj_full_masks.items() if k != -1}
        obj_full_masks[-1] = build_other_mask(non_other, len(model))
        print(f"[render-layers] derived other_mask: "
              f"{int(obj_full_masks[-1].sum()):,} gaussians")

    # 2) Precompute cumulative scene mask at each layer level.
    #    Scene_mask[k] = union over all objects and 'other' of their top-K
    #    importance gaussians at level k.
    print("[render-layers] precomputing per-level combined scene masks...")
    combined_masks_by_level = []
    cumulative_bytes_by_level = []
    n_gaussians_by_level = []
    for k in range(n_layers):
        combined = np.zeros(len(model), dtype=bool)
        for entry in manifest["objects"]:
            oid = entry["object_id"]
            full_mask = obj_full_masks.get(oid)
            if full_mask is None or full_mask.sum() == 0:
                continue
            if k < len(entry["layers"]):
                cum_n = entry["layers"][k]["cumulative_gaussians"]
            else:
                # this object has fewer layers than this level -> use all
                cum_n = entry["layers"][-1]["cumulative_gaussians"]
            combined |= cumulative_mask_for_object(model, full_mask, cum_n,
                                                     rank_method)
        combined_masks_by_level.append(combined)

        # Rate axis: cumulative bytes across the whole scene at this level.
        agg = manifest.get("per_level_aggregate", [])
        cum_bytes = (int(agg[k]["cumulative_bytes_scene"])
                     if k < len(agg) else -1)
        cumulative_bytes_by_level.append(cum_bytes)
        n_gaussians_by_level.append(int(combined.sum()))

        lname = "base" if k == 0 else f"enh{k}"
        print(f"  level {k} ({lname}): combined_gaussians={int(combined.sum()):,}  "
              f"cum_bytes_scene={cum_bytes}  "
              f"cum_pct={layer_pcts[k]}%")

    # 3) Render each level through the full trajectory.
    os.makedirs(args.out_dir, exist_ok=True)
    rates_per_level = {}
    level_dir_names = []
    for k in range(n_layers):
        lname = "base" if k == 0 else f"enh{k}"
        if k == n_layers - 1:
            lname += "_full"          # tag last level as the reference alias
        level_dir_name = f"layer_{k}_{lname}"
        level_dir_names.append(level_dir_name)
        level_dir = os.path.join(args.out_dir, level_dir_name)
        os.makedirs(level_dir, exist_ok=True)

        rates_per_level[level_dir_name] = {
            "level":             k,
            "name":              lname,
            "cumulative_pct":    layer_pcts[k],
            "cumulative_bytes":  cumulative_bytes_by_level[k],
            "cumulative_kb":     round(cumulative_bytes_by_level[k] / 1024.0, 2)
                                  if cumulative_bytes_by_level[k] > 0 else -1,
            "n_gaussians":       n_gaussians_by_level[k],
            "n_frames":          n_frames,
            "is_reference":      (k == n_layers - 1),
        }

        print(f"\n[render-layers] level {k} ({lname}): "
              f"rendering {n_frames} frames...")
        mask_k = combined_masks_by_level[k]
        for fi, mv in enumerate(tqdm(traj_mvs, desc=f"L{k}")):
            img = render_one(model, mv, args.device,
                              use_fallback=args.use_fallback_rasterizer,
                              subset_mask=mask_k)
            # render_one returns RGB; cv2 writes BGR.
            cv2.imwrite(os.path.join(level_dir, f"frame_{fi:06d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 4) Also render the TRUE reference: the full ORIGINAL .ply (no mask) at
    #    each trajectory frame. In a well-formed manifest, this is visually
    #    identical to the last layer (since union of all layers == full),
    #    and VMAF(last_layer, reference) should be ~100. Keep this rendered
    #    separately so bash can use it as the reference for libvmaf.
    ref_dir_name = "reference"
    ref_dir = os.path.join(args.out_dir, ref_dir_name)
    os.makedirs(ref_dir, exist_ok=True)
    print(f"\n[render-layers] REFERENCE: rendering full original model "
          f"(no mask)...")
    for fi, mv in enumerate(tqdm(traj_mvs, desc="ref")):
        img = render_one(model, mv, args.device,
                          use_fallback=args.use_fallback_rasterizer,
                          subset_mask=None)
        cv2.imwrite(os.path.join(ref_dir, f"frame_{fi:06d}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 5) Save the rates manifest.
    rates_path = os.path.join(args.out_dir, "rates.json")
    with open(rates_path, "w") as f:
        json.dump({
            "scene_ply":         args.input,
            "camera_json":       args.camera_json,
            "n_frames":          n_frames,
            "viewport": {"width": traj_mvs[0].width,
                         "height": traj_mvs[0].height},
            "rank_method":       rank_method,
            "layer_percentages": layer_pcts,
            "level_dirs":        level_dir_names,
            "reference_dir":     ref_dir_name,
            "per_level":         rates_per_level,
        }, f, indent=2)
    print(f"\n[render-layers] rates -> {rates_path}")
    print("[done]")


if __name__ == "__main__":
    main()