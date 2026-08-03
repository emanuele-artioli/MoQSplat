"""
Gaussian Splat -> Multi-view rendering -> Object detection -> Per-object splat extraction
==========================================================================================

What this does
--------------
1. Loads a 3DGS .ply scene.
2. Loads a base camera from a JSON file (matching the user's existing format).
3. Builds N cameras around the scene by combining:
     - rotation (azimuth & elevation offsets, "view preset")
     - translation along the camera's right-axis ("lateral_offsets") so that
       elongated scenes -- like a bicycle on a path -- get coverage along the
       path, not just rotated views of one spot.
4. Renders every camera with `gsplat.rasterization` (the same CUDA kernel the
   user's working server uses; identical args).
5. Runs a 2D detector on each render. Two backends supported:
     - "yolo"        : standard YOLO + COCO labels (80 fixed classes;
                       no trees / grass / generic stuff).
     - "yoloworld"   : open-vocabulary YOLO-World; you pass `--text_prompts`
                       like "bicycle,bench,tree,grass,path" and it detects
                       whatever you ask. *Recommended for park / outdoor
                       scenes where you want trees, grass, paths, etc.*
6. Assigns 3D Gaussians to detected objects via multi-view voting, then merges
   detections of the same physical object across views using **3D IoU of the
   per-detection Gaussian sets** (not just centroid distance). This is what
   prevents you from getting four "bicycle" PLYs when there is one bicycle.
7. Saves one .ply per object plus a verification PNG showing what those
   Gaussians look like rendered on their own. If the .ply is good, the PNG
   shows just the object on a black background.

Usage
-----
    # Recommended (open-vocab; detect anything you can name):
    python gsplat_object_segmentation.py \
        --input bicycle.ply \
        --camera_json user1_bicycle.json \
        --outdir out/ \
        --views 8_corners --fov_scale 0.3 \
        --lateral_offsets "-1.5,-0.5,0,0.5,1.5" \
        --detector yoloworld \
        --yolo_model yolov8x-worldv2.pt \
        --text_prompts "bicycle,bench,tree,grass,path,trash can,person"

    # Plain YOLO + COCO (fast, but only 80 classes, no stuff):
    python gsplat_object_segmentation.py \
        --input bicycle.ply --camera_json user1_bicycle.json \
        --outdir out/ --detector yolo --yolo_model yolov8x.pt

Requirements
------------
    pip install numpy torch plyfile opencv-python ultralytics tqdm scipy gsplat
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rsp
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
os.environ["MAX_JOBS"] = "1"

try:
    from gsplat import rasterization as gsplat_rasterization
    HAS_GSPLAT = True
except Exception:
    gsplat_rasterization = None
    HAS_GSPLAT = False

try:
    from ultralytics import YOLO, YOLOWorld
    HAS_YOLO = True
except Exception:
    try:
        from ultralytics import YOLO
        YOLOWorld = None
        HAS_YOLO = True
    except Exception:
        YOLO = None; YOLOWorld = None
        HAS_YOLO = False


# ============================================================================
#  3DGS .ply I/O
# ============================================================================

@dataclass
class GaussianModel:
    xyz:        np.ndarray   # (N, 3)
    normals:    np.ndarray   # (N, 3)
    f_dc:       np.ndarray   # (N, 3)
    f_rest:     np.ndarray   # (N, K)
    opacity:    np.ndarray   # (N, 1) logit
    scale:      np.ndarray   # (N, 3) log
    rotation:   np.ndarray   # (N, 4) quaternion (w,x,y,z)
    extra_fields: List[str] = field(default_factory=list)

    def __len__(self):
        return self.xyz.shape[0]


def load_gaussian_ply(path: str) -> GaussianModel:
    print(f"[load] reading {path}")
    ply = PlyData.read(path)
    v = ply['vertex']
    names = v.data.dtype.names

    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    if {'nx', 'ny', 'nz'}.issubset(names):
        normals = np.stack([v['nx'], v['ny'], v['nz']], axis=1).astype(np.float32)
    else:
        normals = np.zeros_like(xyz)
    f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1).astype(np.float32)

    f_rest_names = sorted(
        [n for n in names if n.startswith('f_rest_')],
        key=lambda s: int(s.split('_')[-1]),
    )
    if f_rest_names:
        f_rest = np.stack([v[n] for n in f_rest_names], axis=1).astype(np.float32)
    else:
        f_rest = np.zeros((xyz.shape[0], 0), dtype=np.float32)

    opacity = np.asarray(v['opacity'], dtype=np.float32).reshape(-1, 1)
    scale = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
    rot = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1).astype(np.float32)

    print(f"[load] {xyz.shape[0]:,} gaussians, f_rest dim = {f_rest.shape[1]}")
    return GaussianModel(xyz=xyz, normals=normals, f_dc=f_dc, f_rest=f_rest,
                        opacity=opacity, scale=scale, rotation=rot,
                        extra_fields=f_rest_names)


def save_gaussian_ply(model: GaussianModel, mask: np.ndarray, path: str):
    idx = np.where(mask)[0]
    n = len(idx)
    if n == 0:
        print(f"[save] WARN: empty mask for {path}, skipping")
        return

    dtype = [
        ('x','f4'),('y','f4'),('z','f4'),
        ('nx','f4'),('ny','f4'),('nz','f4'),
        ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
    ]
    for nm in model.extra_fields:
        dtype.append((nm, 'f4'))
    dtype += [
        ('opacity','f4'),
        ('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
        ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4'),
    ]
    arr = np.empty(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = model.xyz[idx].T
    arr['nx'], arr['ny'], arr['nz'] = model.normals[idx].T
    arr['f_dc_0'], arr['f_dc_1'], arr['f_dc_2'] = model.f_dc[idx].T
    for k, nm in enumerate(model.extra_fields):
        arr[nm] = model.f_rest[idx, k]
    arr['opacity'] = model.opacity[idx, 0]
    arr['scale_0'], arr['scale_1'], arr['scale_2'] = model.scale[idx].T
    arr['rot_0'], arr['rot_1'], arr['rot_2'], arr['rot_3'] = model.rotation[idx].T
    PlyData([PlyElement.describe(arr, 'vertex')]).write(path)
    print(f"[save] {path}  ({n:,} gaussians)")


# ============================================================================
#  Camera (matching the user's existing convention from create_viewmat)
# ============================================================================

@dataclass
class Movement:
    name: str
    angle: float           # azimuth degrees
    elevation: float       # elevation degrees
    x: float; y: float; z: float
    fx: float; fy: float; cx: float; cy: float
    width: int; height: int
    profile: int = 0
    # Optional explicit world-to-camera matrix. If set, we use this instead of
    # building one from (angle, elevation, x, y, z). Lets the script accept
    # trajectory JSONs that store full view_matrix entries and ensures the
    # per-object preview renders match the client's exact viewpoint.
    viewmat_override: Optional[np.ndarray] = None


def get_viewmat(mv: 'Movement') -> torch.Tensor:
    """Return the 4x4 world-to-camera matrix for a Movement. Uses
    `viewmat_override` if set (trajectory format), otherwise builds it from
    (angle, elevation, x, y, z) using create_viewmat (legacy format)."""
    if mv.viewmat_override is not None:
        return torch.tensor(np.asarray(mv.viewmat_override),
                            dtype=torch.float32)
    return create_viewmat(mv.angle, mv.elevation, mv.x, mv.y, mv.z)


def create_viewmat(azimuth_deg: float, elevation_deg: float,
                   x: float, y: float, z: float) -> torch.Tensor:
    """EXACT copy of the user's working create_viewmat (scipy Euler version)."""
    rot = Rsp.from_euler("xyz", [elevation_deg, azimuth_deg, 0],
                         degrees=True).as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = np.array([x, y, z])
    w2c = np.linalg.inv(c2w)
    return torch.tensor(w2c, dtype=torch.float32)


def cam_axes_from_euler(azimuth_deg: float, elevation_deg: float
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (right, up, forward) world-space axes of the camera
    with the user's create_viewmat convention. forward is c2w[:, 2]."""
    rot = Rsp.from_euler("xyz", [elevation_deg, azimuth_deg, 0],
                         degrees=True).as_matrix()
    return rot[:, 0], rot[:, 1], rot[:, 2]


def derive_views(base: Movement,
                 view_specs: List[Tuple[float, float]],
                 lateral_offsets: List[float],
                 view_distance: Optional[float] = None,
                 fov_scale: float = 1.0) -> List[Movement]:
    """
    Build a set of cameras around an implicit look-at target.

    view_specs : list of (delta_azimuth, delta_elevation) in degrees, applied
                 to the BASE camera's orientation. Defines the orbit.
    lateral_offsets : list of distances (world units) to translate the look-at
                      target along the BASE camera's RIGHT axis. Useful for
                      elongated scenes (e.g. a bike on a path) so we sample
                      multiple positions, not just orbit a single point.

    Total cameras returned = len(view_specs) * len(lateral_offsets).
    Pass `[0.0]` for lateral_offsets to disable translation.

    fov_scale : multiply fx, fy. <1 widens the FOV (more visible per frame).
    """
    p_base = np.array([base.x, base.y, base.z], dtype=np.float64)
    right_base, _, fwd_base = cam_axes_from_euler(base.angle, base.elevation)

    if view_distance is None:
        view_distance = float(np.clip(np.linalg.norm(p_base), 1.0, 50.0))
    target_base = p_base + view_distance * fwd_base

    out: List[Movement] = []
    idx = 0
    for lat in lateral_offsets:
        target = target_base + float(lat) * right_base       # slide the focus
        for (dyaw, delev) in view_specs:
            new_az = base.angle + dyaw
            new_el = base.elevation + delev
            _, _, new_fwd = cam_axes_from_euler(new_az, new_el)
            new_pos = target - view_distance * new_fwd
            out.append(Movement(
                name=f"v{idx:02d}_az{int(round(dyaw)):+04d}"
                     f"_el{int(round(delev)):+03d}_lat{lat:+.2f}",
                angle=new_az, elevation=new_el,
                x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2]),
                fx=base.fx * fov_scale, fy=base.fy * fov_scale,
                cx=base.cx, cy=base.cy,
                width=base.width, height=base.height,
                profile=base.profile,
            ))
            idx += 1
    return out


def view_preset(name: str) -> List[Tuple[float, float]]:
    """(d_azimuth, d_elevation) pairs.
    Negative elevation = look more downward (like the bicycle camera at -38)."""
    name = str(name).lower()
    if name == "4":
        return [(0,0),(90,0),(180,0),(-90,0)]
    if name == "6":
        return [(0,0),(90,0),(180,0),(-90,0),(0,-60),(0,+60)]
    if name == "8":
        return [(0,0),(45,0),(90,0),(135,0),(180,0),(-135,0),(-90,0),(-45,0)]
    if name == "8_corners":
        return [(-45,-25),(+45,-25),(-135,-25),(+135,-25),
                (-45,+25),(+45,+25),(-135,+25),(+135,+25)]
    if name == "10":
        return [(0,0),(90,0),(180,0),(-90,0),
                (45,-30),(135,-30),(-135,-30),(-45,-30),
                (0,-75),(0,+75)]
    if name == "12":
        return [(60*i,0) for i in range(6)] + [(60*i+30,-30) for i in range(6)]
    if name == "24":
        # 8 azimuths at 3 elevations: heavy angular coverage for occlusion-prone
        # scenes (bench seat hidden behind grass, bicycle handlebars clipped, etc.)
        out = []
        for elev in (-30, 0, +30):
            for az in range(0, 360, 45):
                out.append((float(az), float(elev)))
        return out
    raise ValueError(f"unknown preset '{name}'. options: 4,6,8,8_corners,10,12,24")


def _movement_from_trajectory_frame(m: dict, default_w: int = 800,
                                    default_h: int = 600,
                                    name: str = "base") -> Movement:
    """Build a Movement from a trajectory entry that uses {view_matrix, fov,
    camera_position}. The view_matrix is taken to be world-to-camera (verified
    against the user's frames -- C = -R^T t matches camera_position to <1mm).
    fov is treated as horizontal FOV in degrees; fy is set so vertical FOV
    matches at the given image size."""
    V = np.asarray(m['view_matrix'], dtype=np.float64)
    if V.shape != (4, 4):
        raise ValueError(f"view_matrix must be 4x4, got {V.shape}")

    width  = int(m.get('width',  default_w))
    height = int(m.get('height', default_h))
    fov_h_deg = float(m['fov'])
    fx = 0.5 * width  / math.tan(math.radians(fov_h_deg) * 0.5)
    fy = fx * (height / width)   # square pixels => same focal in pixels for both axes
    cx = m.get('cx', 0.5 * width)
    cy = m.get('cy', 0.5 * height)

    pos = m.get('camera_position', None)
    if pos is None:
        # derive from view_matrix
        R = V[:3, :3]; t = V[:3, 3]
        pos = (-R.T @ t).tolist()

    return Movement(
        name=name,
        angle=0.0, elevation=0.0,                       # unused when override is set
        x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
        fx=fx, fy=fy, cx=float(cx), cy=float(cy),
        width=width, height=height,
        profile=int(m.get('profile', 0)),
        viewmat_override=V.astype(np.float32),
    )


def load_camera_json(path: str, frame_index: int = 0) -> Movement:
    """Load a base camera from JSON. Auto-detects two formats:
      1. legacy: list of {angle, elevation, x, y, z, fx, fy, cx, cy, width, height}
      2. trajectory: {"frames": [{view_matrix, fov, camera_position, ...}, ...]}
         OR a list of those entries directly.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and 'frames' in data:
        frames = data['frames']
        fmt = 'trajectory'
    elif isinstance(data, list):
        frames = data
        # detect format by looking at the first entry
        sample = frames[0] if frames else {}
        fmt = 'trajectory' if 'view_matrix' in sample else 'legacy'
    else:
        frames = [data]
        fmt = 'trajectory' if 'view_matrix' in data else 'legacy'

    if not frames:
        raise ValueError(f"{path} contains no frames")
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(f"frame_index {frame_index} out of range "
                         f"({len(frames)} entries)")
    m = frames[frame_index]
    print(f"[json] {path} format={fmt}, {len(frames)} frames; "
          f"using index {frame_index}")

    if fmt == 'trajectory':
        return _movement_from_trajectory_frame(m, name="base")

    # legacy format
    return Movement(
        name="base",
        angle=float(m["angle"]), elevation=float(m["elevation"]),
        x=float(m["x"]), y=float(m["y"]), z=float(m["z"]),
        fx=float(m["fx"]), fy=float(m["fy"]),
        cx=float(m["cx"]), cy=float(m["cy"]),
        width=int(m["width"]), height=int(m["height"]),
        profile=int(m.get("profile", 0)),
    )


def load_trajectory_as_movements(path: str, max_frames: Optional[int] = None
                                 ) -> List[Movement]:
    """Load *every* frame from a JSON file as a list of Movements. Used to
    render the per-object preview from the actual client trajectory."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'frames' in data:
        frames = data['frames']
    elif isinstance(data, list):
        frames = data
    else:
        frames = [data]
    if not frames:
        return []
    is_trajectory = 'view_matrix' in frames[0]

    if max_frames is not None and len(frames) > max_frames:
        # uniformly subsample
        idxs = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idxs]

    out: List[Movement] = []
    for i, m in enumerate(frames):
        name = f"traj{i:03d}"
        if is_trajectory:
            out.append(_movement_from_trajectory_frame(m, name=name))
        else:
            out.append(Movement(
                name=name,
                angle=float(m["angle"]), elevation=float(m["elevation"]),
                x=float(m["x"]), y=float(m["y"]), z=float(m["z"]),
                fx=float(m["fx"]), fy=float(m["fy"]),
                cx=float(m["cx"]), cy=float(m["cy"]),
                width=int(m["width"]), height=int(m["height"]),
                profile=int(m.get("profile", 0)),
            ))
    return out


# ============================================================================
#  Rendering with gsplat
# ============================================================================

def render_with_gsplat(model: GaussianModel, mv: Movement, device: str,
                       subset_mask: Optional[np.ndarray] = None,
                       colors_override: Optional[np.ndarray] = None) -> np.ndarray:
    """Render via gsplat.rasterization. If subset_mask is given, only those
    Gaussians are rendered (used for per-object verification).

    colors_override : optional (N, 3) array of SH-DC values to use INSTEAD of
        model.f_dc. Lets us tint gaussians by which layer they belong to for
        the layer-color-coded preview. If `subset_mask` is also given, the
        override array is indexed by the FULL N (we slice to subset, same as
        f_dc). Pass None to use the model's own colors (default).
    """
    if subset_mask is not None:
        idx = np.where(subset_mask)[0]
        if idx.size == 0:
            return np.zeros((mv.height, mv.width, 3), dtype=np.uint8)
        xyz_np  = model.xyz[idx]
        rot_np  = model.rotation[idx]
        sc_np   = model.scale[idx]
        op_np   = model.opacity[idx, 0]
        fdc_np  = (colors_override[idx] if colors_override is not None
                   else model.f_dc[idx])
    else:
        xyz_np = model.xyz; rot_np = model.rotation
        sc_np = model.scale; op_np = model.opacity[:, 0]
        fdc_np = colors_override if colors_override is not None else model.f_dc

    means     = torch.from_numpy(xyz_np).to(device)
    quats     = torch.from_numpy(rot_np).to(device)
    scales    = torch.exp(torch.from_numpy(sc_np).to(device))
    opacities = torch.sigmoid(torch.from_numpy(op_np).to(device))
    f_dc      = torch.from_numpy(fdc_np.astype(np.float32)).to(device).unsqueeze(1)

    viewmat = get_viewmat(mv).to(device).unsqueeze(0)
    K = torch.tensor([[mv.fx, 0, mv.cx],
                      [0, mv.fy, mv.cy],
                      [0,    0,  1]], dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        out = gsplat_rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=f_dc, viewmats=viewmat, Ks=K,
            width=mv.width, height=mv.height,
            packed=False, sh_degree=0, render_mode="RGB",
        )
    colors = out[0] if isinstance(out, tuple) else out
    img = colors[0].clamp(0, 1).detach().cpu().numpy()
    return (img * 255.0).astype(np.uint8)


# ---- minimal CPU fallback (only used when gsplat isn't available) ----------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    ], axis=1).reshape(-1, 3, 3)


def render_fallback(model, mv, device, bg=(0.05,0.05,0.05),
                    subset_mask: Optional[np.ndarray] = None,
                    colors_override: Optional[np.ndarray] = None) -> np.ndarray:
    """Anisotropic-aware NumPy rasterizer; only here so the script runs without
    gsplat for smoke tests. Slow. Use gsplat for real work.

    `colors_override`: optional (N, 3) array replacing model.f_dc, used by the
    layer-color-coded preview to tint each gaussian by its layer."""
    H, W = mv.height, mv.width
    fx, fy, cx_, cy_ = mv.fx, mv.fy, mv.cx, mv.cy
    viewmat = get_viewmat(mv).numpy()
    Rwc = viewmat[:3, :3]; twc = viewmat[:3, 3]

    if subset_mask is not None:
        sel = np.where(subset_mask)[0]
        if sel.size == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)
        xyz = model.xyz[sel]
        fdc = (colors_override[sel] if colors_override is not None
               else model.f_dc[sel])
        op  = model.opacity[sel, 0]; sc = np.exp(model.scale[sel])
        qt  = model.rotation[sel]
    else:
        xyz = model.xyz
        fdc = colors_override if colors_override is not None else model.f_dc
        op = model.opacity[:, 0]; sc = np.exp(model.scale); qt = model.rotation

    means_c = xyz @ Rwc.T + twc[None, :]
    z = means_c[:, 2]
    front = z > 1e-3
    if not front.any():
        return np.full((H, W, 3), [int(c*255) for c in bg], dtype=np.uint8)
    means_c = means_c[front]; z = z[front]
    fdc = fdc[front]; op = op[front]; sc = sc[front]; qt = qt[front]

    SH_C0 = 0.28209479177387814
    rgb = np.clip(0.5 + SH_C0 * fdc, 0, 1).astype(np.float32)
    alpha = (1.0 / (1.0 + np.exp(-op))).astype(np.float32)

    Rg = quat_to_rotmat(qt)
    Ssq = (sc**2)[:, None, :] * np.eye(3)[None, :, :]
    Sigma = np.einsum('nij,njk,nlk->nil', Rg, Ssq, Rg)
    SigmaC = np.einsum('ij,njk,lk->nil', Rwc, Sigma, Rwc)

    X, Y, Z = means_c[:, 0], means_c[:, 1], means_c[:, 2]
    iz = 1.0 / Z
    J = np.zeros((len(Z), 2, 3), dtype=np.float32)
    J[:, 0, 0] = fx * iz; J[:, 0, 2] = -fx * X * iz * iz
    J[:, 1, 1] = fy * iz; J[:, 1, 2] = -fy * Y * iz * iz
    Sigma2D = np.einsum('nij,njk,nlk->nil', J, SigmaC, J)
    Sigma2D[:, 0, 0] += 0.3; Sigma2D[:, 1, 1] += 0.3

    u = fx * X * iz + cx_; v = fy * Y * iz + cy_
    margin = 32
    keep = (u > -margin) & (u < W + margin) & (v > -margin) & (v < H + margin) & (alpha > 0.02)
    if not keep.any():
        return np.full((H, W, 3), [int(c*255) for c in bg], dtype=np.uint8)
    u = u[keep]; v = v[keep]; Sigma2D = Sigma2D[keep]
    rgb = rgb[keep]; alpha = alpha[keep]; z = z[keep]
    order = np.argsort(-z)
    u = u[order]; v = v[order]; Sigma2D = Sigma2D[order]
    rgb = rgb[order]; alpha = alpha[order]

    img = np.zeros((H, W, 3), dtype=np.float32) + np.array(bg, dtype=np.float32)
    acc = np.zeros((H, W), dtype=np.float32)

    a = Sigma2D[:, 0, 0]; b = Sigma2D[:, 0, 1]; c = Sigma2D[:, 1, 1]
    tr = a + c; det = a*c - b*b
    s = np.sqrt(np.clip(tr*tr/4 - det, 0, None))
    rad = np.clip(np.ceil(3 * np.sqrt(tr/2 + s)), 1, 32).astype(np.int32)
    inv_det = 1.0 / np.clip(det, 1e-6, None)
    inv00 =  c * inv_det; inv11 = a * inv_det; inv01 = -b * inv_det

    for i in range(u.shape[0]):
        if acc.min() > 0.999: break
        r = int(rad[i]); ui, vi = int(round(u[i])), int(round(v[i]))
        x0, x1 = max(0, ui - r), min(W, ui + r + 1)
        y0, y1 = max(0, vi - r), min(H, vi + r + 1)
        if x1 <= x0 or y1 <= y0: continue
        xs = np.arange(x0, x1) - u[i]; ys = np.arange(y0, y1) - v[i]
        XX, YY = np.meshgrid(xs, ys)
        e = inv00[i]*XX*XX + 2*inv01[i]*XX*YY + inv11[i]*YY*YY
        g = np.exp(-0.5 * e)
        rem = (1.0 - acc[y0:y1, x0:x1])
        contrib = (alpha[i] * g) * rem
        img[y0:y1, x0:x1] += contrib[..., None] * rgb[i][None, None, :]
        acc[y0:y1, x0:x1] += contrib
    return (np.clip(img, 0, 1) * 255.0).astype(np.uint8)


def render_one(model, mv, device, use_fallback=False,
               subset_mask: Optional[np.ndarray] = None) -> np.ndarray:
    if HAS_GSPLAT and not use_fallback and str(device).startswith("cuda"):
        return render_with_gsplat(model, mv, device, subset_mask=subset_mask)
    return render_fallback(model, mv, device, subset_mask=subset_mask)


# ============================================================================
#  Detection (YOLO / YOLO-World)
# ============================================================================

@dataclass
class Detection:
    view_idx: int
    class_id: int
    class_name: str
    score: float
    bbox: Tuple[int, int, int, int]


def run_detector(images_bgr: List[np.ndarray], model_path: str,
                 detector: str = "yolo",
                 text_prompts: Optional[List[str]] = None,
                 conf: float = 0.20, iou: float = 0.5
                 ) -> List[List[Detection]]:
    """detector ∈ {'yolo', 'yoloworld'}.

    For 'yoloworld' you typically want `model_path='yolov8x-worldv2.pt'` and a
    list of `text_prompts` such as ['bicycle','bench','tree','grass','path'].
    Without text prompts YOLO-World falls back to its default class set.
    """
    if not HAS_YOLO:
        raise RuntimeError("ultralytics is not installed. pip install ultralytics")

    if detector == "yoloworld":
        if YOLOWorld is None:
            raise RuntimeError("YOLOWorld not available; upgrade ultralytics: "
                               "pip install -U ultralytics")
        det = YOLOWorld(model_path)
        if text_prompts:
            det.set_classes(text_prompts)
            print(f"[yolo-world] classes set to: {text_prompts}")
    else:
        det = YOLO(model_path)

    all_dets: List[List[Detection]] = []
    for vi, img in enumerate(tqdm(images_bgr, desc=detector)):
        res = det.predict(img, conf=conf, iou=iou, verbose=False)[0]
        dets: List[Detection] = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                cls_id = int(b.cls[0].item())
                cls_name = det.names[cls_id] if hasattr(det, 'names') \
                    else (text_prompts[cls_id] if text_prompts else str(cls_id))
                dets.append(Detection(vi, cls_id, cls_name,
                                      float(b.conf[0].item()), tuple(xyxy)))
        all_dets.append(dets)
    return all_dets


# ============================================================================
#  Project Gaussians into a Movement camera
# ============================================================================

def project_to_view(xyz_t: torch.Tensor, mv: Movement, device
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    viewmat = get_viewmat(mv).to(device)
    Rwc = viewmat[:3, :3]; twc = viewmat[:3, 3]
    K = torch.tensor([[mv.fx, 0, mv.cx],
                      [0, mv.fy, mv.cy],
                      [0,    0,  1]], dtype=torch.float32, device=device)
    xyz_c = xyz_t @ Rwc.T + twc
    z = xyz_c[:, 2]
    safe = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
    xy = xyz_c[:, :2] / safe.unsqueeze(1)
    uv = xy @ K[:2, :2].T + K[:2, 2]
    return uv, z


def gaussians_in_bbox(xyz_t: torch.Tensor, mv: Movement,
                      bbox: Tuple[int,int,int,int], device) -> np.ndarray:
    uv, z = project_to_view(xyz_t, mv, device)
    x1, y1, x2, y2 = bbox
    inside = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & \
             (uv[:, 1] >= y1) & (uv[:, 1] <= y2) & (z > 1e-3)
    return inside.cpu().numpy()


# ============================================================================
#  Per-detection Gaussian set (this is the heart of the merging logic)
# ============================================================================

def per_detection_gaussian_mask(model: GaussianModel, xyz_t: torch.Tensor,
                                mv: Movement, bbox, device,
                                depth_band_frac: float = 0.60) -> np.ndarray:
    """
    For one 2D detection, return a boolean mask over the N Gaussians: True means
    "this Gaussian likely belongs to the object inside this bbox".

    Logic:
      1. Take all Gaussians whose center projects inside the bbox AND are in
         front of the camera.
      2. Among those, look at the *depth* distribution (z in camera space). The
         object spans a depth range; ground/trees behind it are typically
         farther. We keep Gaussians within a band starting at the near depth
         and extending `depth_band_frac` of the in-bbox depth range.

    The band must be WIDE ENOUGH to cover the whole physical object (front and
    back of a bicycle, both ends of a bench) so detections from different
    angles still overlap. Too narrow -> only the front face of the object is
    kept and same-object detections from opposite views won't merge.
    """
    uv, z = project_to_view(xyz_t, mv, device)
    x1, y1, x2, y2 = bbox
    inbb = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & \
           (uv[:, 1] >= y1) & (uv[:, 1] <= y2) & (z > 1e-3)
    inbb_np = inbb.cpu().numpy()
    if inbb_np.sum() < 5:
        return inbb_np

    z_np = z.cpu().numpy()
    z_in = z_np[inbb_np]
    if len(z_in) < 5:
        return inbb_np

    # near-band: from the 5th percentile depth, extend depth_band_frac of the
    # 5..95 inter-percentile depth range. Keeps the whole object, drops far
    # background.
    p5  = float(np.percentile(z_in, 5))
    p95 = float(np.percentile(z_in, 95))
    z_far = p5 + depth_band_frac * (p95 - p5)
    near = (z_np >= p5 * 0.9) & (z_np <= z_far)
    return inbb_np & near


def detection_centroid_from_mask(model: GaussianModel,
                                 mask: np.ndarray) -> Optional[np.ndarray]:
    if mask.sum() < 5: return None
    return model.xyz[mask].mean(axis=0).astype(np.float32)


# ============================================================================
#  Cross-view merging using 3D IoU of per-detection Gaussian sets
# ============================================================================

@dataclass
class ObjectInstance:
    obj_id: int
    class_name: str
    centroid: np.ndarray
    detections: List[Detection]               = field(default_factory=list)
    gaussian_set: Optional[np.ndarray] = None # boolean (N,) -- union of dets


def jaccard_index(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return 0.0 if union == 0 else inter / union


def merge_detections(detections_per_view: List[List[Detection]],
                     model: GaussianModel, xyz_t: torch.Tensor,
                     mvs: List[Movement], device: str,
                     iou_threshold: float = 0.20,
                     centroid_dist_max: Optional[float] = None
                     ) -> List[ObjectInstance]:
    """
    Greedy merge across views. Two detections are the SAME physical object if:
       (same class) AND (3D IoU of their per-detection Gaussian sets >= iou_threshold)
       OR
       (same class) AND (their 3D centroids are within centroid_dist_max,
        which acts as a fallback when neither set is well-localised).

    This solves the "we got 4 bicycles instead of 1" problem because each view's
    detection of the bicycle picks up overlapping Gaussian sets.
    """
    if centroid_dist_max is None:
        bbox_lo = np.percentile(model.xyz, 1, axis=0)
        bbox_hi = np.percentile(model.xyz, 99, axis=0)
        # 15% of the scene diagonal -- generous enough to merge detections of
        # the same object from opposite sides, since their per-view 3D centroids
        # will sit on the near surface, not the true center.
        centroid_dist_max = 0.15 * float(np.linalg.norm(bbox_hi - bbox_lo))
    print(f"[merge] iou_thr={iou_threshold}  centroid_dist_max={centroid_dist_max:.3f}")

    objects: List[ObjectInstance] = []
    total = sum(len(d) for d in detections_per_view)
    pbar = tqdm(total=total, desc="merge")
    for vi, dets in enumerate(detections_per_view):
        mv = mvs[vi]
        for d in dets:
            mask = per_detection_gaussian_mask(model, xyz_t, mv, d.bbox, device)
            n_in = int(mask.sum())
            if n_in < 5:
                pbar.update(1); continue
            centroid = detection_centroid_from_mask(model, mask)
            if centroid is None:
                pbar.update(1); continue

            # search for an existing object that matches
            best_j = -1; best_score = 0.0
            for j, obj in enumerate(objects):
                if obj.class_name != d.class_name:
                    continue
                iou = jaccard_index(obj.gaussian_set, mask)
                cdist = float(np.linalg.norm(obj.centroid - centroid))
                # accept if either criterion is satisfied; rank by IoU
                if iou >= iou_threshold or cdist < centroid_dist_max:
                    score = iou + (1.0 if cdist < centroid_dist_max else 0.0)
                    if score > best_score:
                        best_score = score; best_j = j

            if best_j >= 0:
                obj = objects[best_j]
                # update centroid as running mean over detections
                n = len(obj.detections)
                obj.centroid = (obj.centroid * n + centroid) / (n + 1)
                obj.detections.append(d)
                obj.gaussian_set = np.logical_or(obj.gaussian_set, mask)
            else:
                objects.append(ObjectInstance(
                    obj_id=len(objects), class_name=d.class_name,
                    centroid=centroid, detections=[d], gaussian_set=mask,
                ))
            pbar.update(1)
    pbar.close()
    print(f"[merge] {total} detections -> {len(objects)} unique objects")
    return objects


def _pad_bbox(bbox: Tuple[int,int,int,int], pad_factor: float,
              W: int, H: int) -> Tuple[int,int,int,int]:
    """Pad a 2D bbox by `(pad_factor - 1)` of its half-extent on each side.
    pad_factor=1.0 -> unchanged. pad_factor=1.5 -> 25% growth on every side
    (each dim becomes 1.5x). Result is clipped to the image size."""
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
    hw = 0.5 * (x2 - x1) * pad_factor
    hh = 0.5 * (y2 - y1) * pad_factor
    return (max(0, int(cx - hw)),    max(0, int(cy - hh)),
            min(W - 1, int(cx + hw)), min(H - 1, int(cy + hh)))


def build_final_object_masks(model: GaussianModel, xyz_t: torch.Tensor,
                             objects: List[ObjectInstance], mvs: List[Movement],
                             device, vote_threshold: float = 0.5,
                             mask_pad: float = 1.3,
                             spatial_radius_frac: float = 0.20,
                             depth_band_frac: float = 0.35,
                             ) -> Dict[int, np.ndarray]:
    """
    Build the final per-Gaussian mask for every merged object.

    Strategy:
      1. For each object's detections, PAD the 2D bbox by `mask_pad` AND apply
         a depth-band filter (per_detection_gaussian_mask). A Gaussian votes
         for an object in a view if BOTH (a) it projects into the padded bbox
         AND (b) its camera-space depth lies in the near-band of in-bbox depths.
         The depth filter is critical: without it, gaussians along the line of
         sight far behind/in-front of the object also vote, polluting the .ply
         with background gaussians that render correctly at synthetic viewpoints
         but appear at wildly wrong world positions when played back through a
         different trajectory.
      2. Apply a TIGHT spatial gate built from the object's own current
         gaussian_set extent (not scene diagonal). This kills any remaining
         far-background votes.
      3. Winner-take-all: each Gaussian goes to its top-voting object IF that
         vote >= vote_threshold. Otherwise it stays unassigned ("other").
    """
    N = len(model)
    if not objects: return {}
    print(f"[mask] final voting for {len(objects)} objects, {N:,} gaussians  "
          f"(mask_pad={mask_pad}, depth_band_frac={depth_band_frac})")

    votes = np.zeros((len(objects), N), dtype=np.float32)
    for oi, obj in enumerate(tqdm(objects, desc="votes")):
        if not obj.detections: continue
        for d in obj.detections:
            mv = mvs[d.view_idx]
            padded = _pad_bbox(d.bbox, mask_pad, mv.width, mv.height)
            # depth-band-filtered mask: only accept gaussians at the near-depth
            # band of this padded bbox, NOT the entire line of sight.
            inside = per_detection_gaussian_mask(model, xyz_t, mv, padded,
                                                 device,
                                                 depth_band_frac=depth_band_frac)
            votes[oi] += inside.astype(np.float32)
        votes[oi] /= len(obj.detections)

    # tight spatial gate per object: derive radius from the OBJECT's own
    # current gaussian_set extent (the union of per-detection depth-filtered
    # masks), so a small object gets a small gate and a big object gets a big
    # gate. This is what stops a bicycle from absorbing 4m of surrounding
    # ground.
    print(f"[mask] applying per-object spatial gates")
    for oi, obj in enumerate(objects):
        gs = obj.gaussian_set
        if gs is None or gs.sum() < 5:
            # fallback to the old global radius -- shouldn't happen in practice
            diag = float(np.linalg.norm(np.percentile(model.xyz, 99, axis=0)
                                        - np.percentile(model.xyz, 1, axis=0)))
            spatial_r = spatial_radius_frac * diag
        else:
            # 95th-percentile distance from centroid among object's own gaussians,
            # then pad by 30% to allow recovering missed parts.
            d2c = np.linalg.norm(model.xyz[gs] - obj.centroid[None, :], axis=1)
            spatial_r = float(np.percentile(d2c, 95)) * 1.30
        d = np.linalg.norm(model.xyz - obj.centroid[None, :], axis=1)
        votes[oi, d > spatial_r] = 0.0
        print(f"  obj {oi:2d} [{objects[oi].class_name:>14}]  "
              f"spatial_r={spatial_r:.3f}")

    # Multi-label assignment: each gaussian can belong to multiple objects.
    # This is intentional -- a gaussian on the bench/bicycle interface really
    # does belong to both, and forcing winner-take-all makes one object's .ply
    # incomplete when objects are spatially close. Each object simply keeps
    # every gaussian whose vote >= vote_threshold for that object.
    masks: Dict[int, np.ndarray] = {}
    for oi in range(len(objects)):
        m = votes[oi] >= vote_threshold
        masks[oi] = m
        print(f"  obj {oi:2d} [{objects[oi].class_name:>14}] "
              f"views={len(objects[oi].detections):2d}  "
              f"gaussians={m.sum():,}")
    return masks


def build_other_mask(masks: Dict[int, np.ndarray], n_total: int) -> np.ndarray:
    """Return the boolean mask of Gaussians not assigned to any object -- the
    'other' .ply (ground, trees, sky, anything the detector didn't catch)."""
    assigned = np.zeros(n_total, dtype=bool)
    for m in masks.values():
        assigned |= m
    return ~assigned


def _object_extent(model: GaussianModel, mask: np.ndarray) -> float:
    """Diameter of the bounding box of `mask`'s gaussian centers, in world
    units. Used to size the dilation radius for that object. Returns 0 when
    the mask has fewer than 5 points."""
    if mask.sum() < 5:
        return 0.0
    pts = model.xyz[mask]
    extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    return float(extent)


def _object_bbox_extent(model: GaussianModel, mask: np.ndarray
                         ) -> Tuple[np.ndarray, float]:
    """Return (centroid, max_principal_extent) for an object mask.
    Centroid is the percentile-midpoint (robust to outliers, unlike mean).
    Max principal extent is the largest single-axis extent in world units --
    NOT the diagonal, because the camera distance for nice framing should
    depend on the object's WIDEST visible dimension, not its diagonal."""
    if mask.sum() < 5:
        return np.zeros(3, dtype=np.float32), 0.0
    pts = model.xyz[mask]
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    centroid = ((lo + hi) * 0.5).astype(np.float32)
    extent = float((hi - lo).max())
    return centroid, extent


def _cam_world_axes(mv: Movement) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (right, up, forward) world-space axes of the camera, working
    for BOTH the legacy Euler convention and a viewmat_override (trajectory).
    For W2C = [R | t], the third row of R is world_forward."""
    if mv.viewmat_override is not None:
        V = np.asarray(mv.viewmat_override, dtype=np.float64)
        R = V[:3, :3]
        return (R[0, :].astype(np.float32),
                R[1, :].astype(np.float32),
                R[2, :].astype(np.float32))
    return cam_axes_from_euler(mv.angle, mv.elevation)


def build_camera_looking_at(target: np.ndarray, ref_mv: Movement,
                            object_extent: float,
                            fit_fraction: float = 0.70) -> Movement:
    """Build a camera that looks at `target` (a world-space point) and is
    placed so the object's max extent fills `fit_fraction` of the frame
    height. Orientation is inherited from the reference camera so the
    render convention (create_viewmat / scipy Euler) stays consistent.

    Why per-object distance matters: the synthetic detection cameras use one
    global view_distance, which works on average but means small objects
    (bowls) render as tiny dots and large objects (couches) get cropped.
    Per-object cameras fix both extremes."""
    _, _, fwd = _cam_world_axes(ref_mv)
    fy = max(1.0, ref_mv.fy)
    desired_pixels = fit_fraction * ref_mv.height
    extent_safe = max(object_extent, 0.05)
    # Pinhole: pixels = fy * extent / distance --> distance = fy*extent/pixels
    distance = fy * extent_safe / max(desired_pixels, 1.0)
    distance = float(np.clip(distance, 0.3, 50.0))
    pos = target - distance * fwd
    return Movement(
        name=f"obj_centered_d{distance:.2f}",
        angle=ref_mv.angle, elevation=ref_mv.elevation,
        x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
        fx=ref_mv.fx, fy=ref_mv.fy, cx=ref_mv.cx, cy=ref_mv.cy,
        width=ref_mv.width, height=ref_mv.height,
        profile=ref_mv.profile,
        # viewmat_override left None so create_viewmat builds the matrix
        # from our angle/elevation/x/y/z (correct framing guaranteed).
    )


def dilate_object_masks(model: GaussianModel,
                        seed_masks: Dict[int, np.ndarray],
                        objects: List[ObjectInstance],
                        radius_frac: float = 0.10,
                        radius_abs: Optional[float] = None,
                        radius_max_abs: Optional[float] = None,
                        scene_chunk: int = 200_000,
                        ) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """
    Grow each object's seed mask by a 3D radius, recovering gaussians that
    were missed by 2D bbox voting (e.g. the bench seat hidden behind grass,
    the bicycle handlebars clipped at the bbox edge).

    Per-object radius:
      r_o = radius_abs if set,
            else radius_frac * extent(seed_mask[o])
            optionally clamped to <= radius_max_abs.

    Conflict resolution between dilated objects (winner-take-all):
      - Seeds are sacred. A gaussian already in some object's seed never gets
        reassigned.
      - For non-seed gaussians, we find the nearest SEED point across all
        objects. The gaussian is added to that object if its nearest seed is
        within that object's radius r_o.

    Implementation note (memory):
      The naive approach -- for each seed point ask "give me all my neighbors
      within r" -- materializes a Python list-of-lists with potentially
      billions of entries (seed_count x avg_neighbors_per_seed). Out of memory
      on dense scenes.

      Instead we FLIP the query direction. We build ONE small KDTree over all
      seed points across all objects (not over the full 6M-point scene), then
      for each scene point ask "what is your single nearest seed?" -- one
      number per scene point, O(N) memory. This is also faster because the
      tree is much smaller (tens of thousands of seeds vs millions of scene
      points). We process the scene in chunks of `scene_chunk` points to keep
      peak memory bounded for huge scenes.

    Returns:
      (new_masks, stats) where stats[oi] holds {'seed', 'added', 'final',
      'radius', 'extent'}.
    """
    if not seed_masks:
        return seed_masks, {}

    from scipy.spatial import cKDTree
    N = len(model)

    # Compute per-object radius and extent.
    radii: Dict[int, float] = {}
    extents: Dict[int, float] = {}
    for oi, seed in seed_masks.items():
        ext = _object_extent(model, seed)
        if radius_abs is not None:
            r = float(radius_abs)
        else:
            r = radius_frac * ext
        if radius_max_abs is not None:
            r = min(r, float(radius_max_abs))
        radii[oi]   = r
        extents[oi] = ext

    # Build a SINGLE KDTree over all seed points combined (small, fits in RAM).
    # Each seed point carries a label (object id) so when we query "nearest
    # seed to scene point P" we know which object that seed belongs to.
    seed_points_chunks: List[np.ndarray] = []
    seed_labels_chunks: List[np.ndarray] = []
    in_any_seed = np.zeros(N, dtype=bool)
    total_seeds = 0
    for oi, seed in seed_masks.items():
        n = int(seed.sum())
        if n == 0: continue
        in_any_seed |= seed
        seed_points_chunks.append(model.xyz[seed])
        seed_labels_chunks.append(np.full(n, oi, dtype=np.int32))
        total_seeds += n

    if total_seeds == 0:
        return dict(seed_masks), {oi: dict(seed=0, added=0, final=0,
                                            radius=0.0, extent=0.0)
                                  for oi in seed_masks}

    seed_points = np.concatenate(seed_points_chunks, axis=0)
    seed_labels = np.concatenate(seed_labels_chunks, axis=0)

    print(f"[dilate] building KDTree over {total_seeds:,} seed points "
          f"(NOT the full {N:,} scene). radii={[f'{r:.3f}' for r in radii.values()]}")
    t0 = time.time()
    tree = cKDTree(seed_points)
    print(f"[dilate] KDTree built in {time.time()-t0:.2f}s")

    # For each scene point: nearest seed distance + which object that seed belongs to.
    # Process in chunks so peak memory stays bounded regardless of scene size.
    print(f"[dilate] querying {N:,} scene points in chunks of {scene_chunk:,}")
    t0 = time.time()
    nearest_dist  = np.empty(N, dtype=np.float32)
    nearest_label = np.empty(N, dtype=np.int32)
    for s in range(0, N, scene_chunk):
        e = min(s + scene_chunk, N)
        d, idx = tree.query(model.xyz[s:e], k=1, workers=-1)
        nearest_dist[s:e]  = d.astype(np.float32)
        nearest_label[s:e] = seed_labels[idx]
    print(f"[dilate] queried in {time.time()-t0:.2f}s")

    # For each object oi, a scene point becomes part of oi iff:
    #   (1) it's not already in any seed (sacred-seeds rule)
    #   (2) its nearest seed belongs to oi
    #   (3) that distance is within oi's radius
    final_masks: Dict[int, np.ndarray] = {}
    stats: Dict[int, dict] = {}
    for oi, seed in seed_masks.items():
        r = radii[oi]
        if r <= 0.0:
            final_masks[oi] = seed.copy()
            stats[oi] = dict(seed=int(seed.sum()), added=0,
                             final=int(seed.sum()),
                             radius=0.0, extent=extents[oi])
            cls = objects[oi].class_name if oi < len(objects) else f"obj{oi}"
            print(f"[dilate] obj {oi:2d} [{cls:>14}] r=0  no growth")
            continue
        added = (~in_any_seed) & (nearest_label == oi) & (nearest_dist <= r)
        final = seed | added
        final_masks[oi] = final
        n_seed  = int(seed.sum())
        n_added = int(added.sum())
        n_final = int(final.sum())
        stats[oi] = dict(seed=n_seed, added=n_added, final=n_final,
                         radius=r, extent=extents[oi])
        cls = objects[oi].class_name if oi < len(objects) else f"obj{oi}"
        print(f"[dilate] obj {oi:2d} [{cls:>14}] "
              f"extent={extents[oi]:.3f} r={r:.3f}  "
              f"seed={n_seed:,} -> +{n_added:,} = {n_final:,}")

    return final_masks, stats


# ============================================================================
#  Object clustering (group physically-close objects into cluster .plys)
# ============================================================================
#
# Motivation: when two objects are physically close (e.g. a bicycle leaning
# against a bench), the per-object spatial gates and depth-band filters can
# clip gaussians at the shared boundary. Streaming each object alone produces
# visible seams. A "cluster" .ply bundles all gaussians from the close
# objects, removing the boundary entirely.
#
# Clustering is OPT-IN (--clustering flag) and PRODUCED IN ADDITION TO the
# per-object .plys -- not as a replacement. The streaming server can choose
# at query time whether to serve individual objects or their containing
# cluster, depending on what's currently in the user's view.
#
# Method options:
#   - "gap" (default): single-linkage clustering on the gap distance between
#     each pair of objects' axis-aligned 3D bounding boxes. Two objects join
#     the same cluster if the gap between their bboxes is <= eps. This
#     handles touching/overlapping objects directly (negative gap merges).
#     Equivalent to DBSCAN with min_samples=1 under gap distance.
#   - "dbscan": sklearn DBSCAN on object centroids. Treats objects as points;
#     correct only when objects are reasonably point-like.
#   - "kmeans": for ablations only -- partitions ALL objects into K clusters
#     regardless of spatial closeness, so a lonely couch may end up grouped
#     with distant objects. Not recommended; included for paper comparisons.
#
# Distance threshold (eps): either absolute world units (--cluster_eps_abs)
# or a fraction of the scene diagonal (--cluster_eps_frac, default 0.03 i.e.
# 3% of the scene size). With the bicycle-bench example, 3% of a typical
# outdoor scene is ~0.3m, enough to merge touching objects but not enough to
# pull in something on the other side of the path.

def _object_aabb(model: GaussianModel, mask: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned 3D bounding box (lo, hi) for an object mask.
    Uses 5th/95th percentile rather than min/max so a few outlier gaussians
    don't blow up the bbox. Returns zeros for empty masks."""
    if mask.sum() < 5:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    pts = model.xyz[mask]
    return (np.percentile(pts, 5, axis=0).astype(np.float32),
            np.percentile(pts, 95, axis=0).astype(np.float32))


def _aabb_gap(lo_a: np.ndarray, hi_a: np.ndarray,
              lo_b: np.ndarray, hi_b: np.ndarray) -> float:
    """Minimum Euclidean distance between two axis-aligned 3D bboxes.
    Negative when the bboxes overlap (we return the negative overlap amount
    along the most-overlapping axis, so overlapping is "even closer than
    touching")."""
    # Per-axis gap: positive if disjoint along that axis, negative if
    # overlapping. Overall gap is sqrt(sum of positive parts) for the
    # standard "closest point on box A to closest point on box B" distance.
    gap_axis = np.maximum(lo_b - hi_a, lo_a - hi_b)  # (3,)
    pos = np.maximum(gap_axis, 0.0)
    if (gap_axis < 0).all():
        # all axes overlap -> bboxes intersect; return negative max overlap
        return float(gap_axis.max())  # largest (least negative) overlap
    return float(np.linalg.norm(pos))


def _scene_diagonal(model: GaussianModel) -> float:
    lo = np.percentile(model.xyz, 1, axis=0)
    hi = np.percentile(model.xyz, 99, axis=0)
    return float(np.linalg.norm(hi - lo))


def cluster_objects_gap(model: GaussianModel,
                        masks: Dict[int, np.ndarray],
                        eps: float) -> List[List[int]]:
    """Single-linkage clustering on bbox gap distance.

    For each pair of objects (i, j), compute the gap between their AABBs.
    If gap <= eps, they're in the same cluster. Returns a list of clusters,
    each a sorted list of object IDs.

    Implementation: build a union-find over object IDs, merge pairs whose
    gap <= eps, then collect components. O(N^2) pairs; fine for typical
    scenes with tens of objects.
    """
    oids = sorted(masks.keys())
    n = len(oids)
    if n == 0:
        return []

    # Precompute AABBs.
    aabbs = {oi: _object_aabb(model, masks[oi]) for oi in oids}

    # Union-find.
    parent = {oi: oi for oi in oids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for ii in range(n):
        for jj in range(ii + 1, n):
            oi, oj = oids[ii], oids[jj]
            gap = _aabb_gap(*aabbs[oi], *aabbs[oj])
            if gap <= eps:
                union(oi, oj)

    # Collect components.
    groups: Dict[int, List[int]] = {}
    for oi in oids:
        r = find(oi)
        groups.setdefault(r, []).append(oi)
    return [sorted(g) for g in groups.values()]


def cluster_objects_dbscan(model: GaussianModel,
                           masks: Dict[int, np.ndarray],
                           eps: float, min_samples: int = 1
                           ) -> List[List[int]]:
    """DBSCAN on object centroids (3D world space). Objects too far from any
    neighbor become singletons (their own cluster) when min_samples=1.
    Requires sklearn."""
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise RuntimeError("--cluster_method dbscan requires sklearn. "
                           "Install with: pip install scikit-learn")

    oids = sorted(masks.keys())
    if not oids:
        return []
    centroids = np.array([_object_bbox_extent(model, masks[oi])[0]
                          for oi in oids])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(centroids)
    groups: Dict[int, List[int]] = {}
    for k, oi in enumerate(oids):
        lbl = int(labels[k])
        # DBSCAN's "noise" label is -1 with min_samples > 1; each noise
        # point becomes its own singleton cluster.
        key = lbl if lbl >= 0 else f"noise_{oi}"
        groups.setdefault(key, []).append(oi)
    return [sorted(g) for g in groups.values()]


def cluster_objects_kmeans(model: GaussianModel,
                           masks: Dict[int, np.ndarray],
                           k: int) -> List[List[int]]:
    """K-means on object centroids. Forces a fixed K partition regardless of
    closeness. Provided for paper-comparison ablations only -- NOT recommended
    as a default because lonely objects get arbitrarily grouped with distant
    neighbors."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise RuntimeError("--cluster_method kmeans requires sklearn. "
                           "Install with: pip install scikit-learn")
    oids = sorted(masks.keys())
    if not oids:
        return []
    k = max(1, min(k, len(oids)))
    centroids = np.array([_object_bbox_extent(model, masks[oi])[0]
                          for oi in oids])
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(centroids)
    groups: Dict[int, List[int]] = {}
    for kk, oi in enumerate(oids):
        groups.setdefault(int(labels[kk]), []).append(oi)
    return [sorted(g) for g in groups.values()]


def build_cluster_masks(masks: Dict[int, np.ndarray],
                        clusters: List[List[int]],
                        n_total: int) -> List[np.ndarray]:
    """For each cluster (list of object IDs), build a boolean mask over the
    full set of N gaussians: the UNION of the constituent objects' masks.
    Because the original masks are multi-label (a gaussian on the
    bench-bicycle interface belongs to both), the union is the right semantic
    -- shared gaussians are kept exactly once."""
    out: List[np.ndarray] = []
    for grp in clusters:
        m = np.zeros(n_total, dtype=bool)
        for oi in grp:
            if oi in masks:
                m |= masks[oi]
        out.append(m)
    return out


# ============================================================================
#  Progressive (scalable) layer writer
# ============================================================================

def _gaussian_importance(model: GaussianModel, mask: np.ndarray,
                          rank_method: str = "opacity_scale") -> np.ndarray:
    """
    Score each gaussian in `mask` by its contribution to render quality.
    Returns an array of shape (mask.sum(),) of importance scores; HIGHER means
    more important (so the gaussian should appear in earlier layers).

    rank_method:
      - "opacity":        sigmoid(opacity_logit). Highest-opacity gaussians
                          first. Simple, intuitive.
      - "opacity_scale":  sigmoid(opacity) * exp(max(scale)). Approximates
                          screen-space contribution: a gaussian's visual
                          impact scales with how visible it is AND how large
                          it appears in image space. This is what 3DGS
                          compression papers use. RECOMMENDED default.
      - "scale":          exp(max(scale)) only. Big gaussians first.
    """
    sel = np.where(mask)[0]
    if len(sel) == 0:
        return np.zeros(0, dtype=np.float32)
    op_logit = model.opacity[sel, 0].astype(np.float64)
    opacity  = 1.0 / (1.0 + np.exp(-op_logit))
    if rank_method == "opacity":
        return opacity.astype(np.float32)
    # exp() of log-scales gives world-space scales; max along the 3 axes is
    # the longest principal axis of the gaussian. We use the max because
    # the LARGEST extent dominates visual contribution from typical viewpoints.
    log_scale = model.scale[sel].astype(np.float64)
    world_scale = np.exp(np.clip(log_scale, -10, 10))   # clip for numerical safety
    max_scale = world_scale.max(axis=1)
    if rank_method == "scale":
        return max_scale.astype(np.float32)
    # opacity_scale: product. Both terms in [0, ~few]; the product orders
    # gaussians by approximate screen-space contribution.
    return (opacity * max_scale).astype(np.float32)


def write_progressive_layers(model: GaussianModel,
                             object_mask: np.ndarray,
                             out_dir: str,
                             object_label: str,
                             percentages: List[float] = (20, 40, 60, 80, 100),
                             rank_method: str = "opacity_scale",
                             write_full: bool = True,
                             ) -> dict:
    """
    Write progressive layers for one object (or "other" / background).

    Each layer file contains ONLY the new gaussians added at that layer
    (a "delta" or "enhancement" layer in scalable-coding terms). The client
    obtains fidelity level k by loading layers 0, 1, ..., k and merging them.

    For example, with default percentages [20, 40, 60, 80, 100]:
      layer_0_base   : top 20% of gaussians ranked by importance      (delta vs nothing)
      layer_1_enh1   : the next 20% (gaussians ranked 20%-40%)        (delta vs base)
      layer_2_enh2   : the next 20% (gaussians ranked 40%-60%)        (delta vs base+enh1)
      layer_3_enh3   : the next 20% (gaussians ranked 60%-80%)        (delta vs base+enh1+enh2)
      layer_4_enh4   : the final 20% (gaussians ranked 80%-100%)      (delta vs base+enh1..3)

    The layers PARTITION the object's gaussians: every gaussian appears in
    exactly ONE layer .ply. The client reconstructs each fidelity level by
    union (set merge, not file merge).

    Output layout (created at `out_dir/`):
      layer_0_base_n<count>.ply
      layer_1_enh1_n<count>.ply
      layer_2_enh2_n<count>.ply
      layer_3_enh3_n<count>.ply
      layer_4_enh4_n<count>.ply
      manifest.json     -- describes layers, their cumulative percentages,
                            cumulative gaussian counts, file paths, and byte sizes.

    Returns:
      (manifest, layer_masks) where manifest is the JSON dict (also written to
      disk) and layer_masks is a list of boolean masks, one per layer, used
      by the layer visualization helpers below.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_total = int(object_mask.sum())
    if n_total == 0:
        manifest = {"object": object_label, "n_total": 0, "layers": []}
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[layers] {object_label}: 0 gaussians, skipping")
        return manifest, []

    # Score and sort. Highest importance first.
    scores = _gaussian_importance(model, object_mask, rank_method=rank_method)
    obj_indices = np.where(object_mask)[0]
    order = np.argsort(-scores)               # descending
    sorted_indices = obj_indices[order]

    # Compute layer boundaries from cumulative percentages.
    pcts = list(percentages)
    if pcts[0] <= 0 or pcts[-1] > 100:
        raise ValueError("percentages must be in (0, 100], strictly increasing")
    for i in range(1, len(pcts)):
        if pcts[i] <= pcts[i-1]:
            raise ValueError("percentages must be strictly increasing")
    cum_counts = [int(round(p / 100.0 * n_total)) for p in pcts]
    cum_counts[-1] = n_total                 # snap last layer to include everything

    # Per-layer index slices (delta layers): layer k contains
    # sorted_indices[cum_counts[k-1] : cum_counts[k]] (0 for k==0).
    starts = [0] + cum_counts[:-1]
    ends   = cum_counts

    # Standard streaming bitrates we report download-time estimates for.
    # Megabits/second, the units real streaming services advertise.
    BITRATES_MBPS = [1, 5, 10, 25, 50, 100]

    def _download_ms(size_bytes: int) -> Dict[str, float]:
        """Estimate transfer time at each standard bitrate. We report a
        simple bytes/throughput estimate; real latency adds RTT and protocol
        overhead, but for paper-scale tables this is the right back-of-envelope
        number."""
        out = {}
        for mbps in BITRATES_MBPS:
            bytes_per_s = mbps * 125_000.0  # 1 Mbps = 125_000 B/s (decimal)
            out[f"{mbps}Mbps_ms"] = round(1000.0 * size_bytes / bytes_per_s, 2)
        return out

    layers_meta = []
    layer_masks_list: List[np.ndarray] = []   # for visualization
    cumulative_bytes = 0
    for k, (s, e, p) in enumerate(zip(starts, ends, pcts)):
        if e <= s:
            print(f"[layers] {object_label}: layer {k} would be empty, skipping")
            continue
        layer_idx = sorted_indices[s:e]
        layer_mask = np.zeros(len(model), dtype=bool)
        layer_mask[layer_idx] = True
        n_layer = int(layer_mask.sum())
        cum_n = e
        layer_name = "base" if k == 0 else f"enh{k}"
        fname = f"layer_{k}_{layer_name}_n{n_layer}.ply"
        fpath = os.path.join(out_dir, fname)
        save_gaussian_ply(model, layer_mask, fpath)
        size_bytes = os.path.getsize(fpath)
        cumulative_bytes += size_bytes
        meta_entry = {
            "layer":              k,
            "name":               layer_name,
            "delta_gaussians":    n_layer,
            "cumulative_pct":     p,
            "cumulative_gaussians": cum_n,
            "file":               fname,
            "size_bytes":         size_bytes,
            "size_kb":            round(size_bytes / 1024.0, 2),
            "cumulative_bytes":   cumulative_bytes,
            "cumulative_kb":      round(cumulative_bytes / 1024.0, 2),
            "gaussians_per_kb":   round(n_layer / max(1, size_bytes / 1024.0), 2),
            "download_delta":     _download_ms(size_bytes),
            "download_cumulative": _download_ms(cumulative_bytes),
        }
        layers_meta.append(meta_entry)
        layer_masks_list.append(layer_mask)

    # (Optional) full single-file .ply for non-streaming clients.
    full_path = None
    if write_full:
        full_fname = f"full_n{n_total}.ply"
        full_path  = os.path.join(out_dir, full_fname)
        save_gaussian_ply(model, object_mask, full_path)

    manifest = {
        "object":       object_label,
        "n_total":      n_total,
        "rank_method":  rank_method,
        "layers":       layers_meta,
        "full_file":    os.path.basename(full_path) if full_path else None,
        "total_bytes":  cumulative_bytes,
        "total_kb":     round(cumulative_bytes / 1024.0, 2),
        "bytes_per_gaussian": round(cumulative_bytes / max(1, n_total), 2),
        "note": ("All layers contain the same NUMBER of gaussians per slice "
                 "(determined by --layer_percentages), and each 3DGS gaussian "
                 "is a fixed-size 62-float row, so the layer .ply byte sizes "
                 "are nearly identical regardless of --layer_rank. The rank "
                 "method changes WHICH gaussians go in WHICH layer, not how "
                 "many bytes each layer takes. To get different sizes per "
                 "layer, use non-uniform percentages e.g. '10,30,60,100'."),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[layers] {object_label}: wrote {len(layers_meta)} layers, "
          f"manifest -> {out_dir}/manifest.json")
    return manifest, layer_masks_list


# ============================================================================
#  Layer visualization
# ============================================================================
#
# Two distinct visualizations per object, both rendered from a single
# representative viewpoint (the highest-confidence YOLO detection view):
#
#   1. PROGRESSION strip (5 tiles, side-by-side): how the object looks at each
#      cumulative download stage. Tile k = base + enh1 + ... + enh_k rendered
#      normally with the original splat colors. Answers "what does the user
#      see if my client only downloaded up to layer K?"
#
#   2. COLOR overlay (single tile): each layer's gaussians rendered with a
#      distinct color and additively composited. Answers "where in the object
#      do the high-importance gaussians live vs the low-importance fill-in?"
#
# Layer-tint colors (RGB in [0, 1]). Designed to be perceptually distinct so
# you can tell at a glance which layer dominates each pixel of the overlay:
#   layer 0 base = red       (most-important core gaussians)
#   layer 1 enh1 = blue
#   layer 2 enh2 = yellow
#   layer 3 enh3 = green     ("least-important detail" -- visible only at full quality)
# Extra fallback colors for >4 layers (used when --layer_percentages adds more steps).
LAYER_COLORS_RGB: List[Tuple[float, float, float]] = [
    (0.95, 0.20, 0.20),    # layer 0: red
    (0.20, 0.40, 0.95),    # layer 1: blue
    (0.95, 0.85, 0.20),    # layer 2: yellow
    (0.30, 0.85, 0.30),    # layer 3: green
    (0.95, 0.55, 0.20),    # layer 4: orange (fallback)
    (0.75, 0.30, 0.95),    # layer 5: purple (fallback)
    (0.20, 0.85, 0.85),    # layer 6: cyan   (fallback)
]


# Inverse of the SH DC term encoding used elsewhere: rgb = clamp(0.5 + SH_C0 * f_dc, 0, 1)
# So to make a gaussian render as a target RGB color, set f_dc = (rgb - 0.5) / SH_C0.
_SH_C0 = 0.28209479177387814

def _rgb_to_f_dc(rgb: Tuple[float, float, float]) -> np.ndarray:
    return ((np.asarray(rgb, dtype=np.float32) - 0.5) / _SH_C0)


def _render_with_color_override(model: GaussianModel, mv: 'Movement',
                                subset_mask: np.ndarray,
                                rgb: Tuple[float, float, float],
                                device: str,
                                use_fallback: bool = False) -> np.ndarray:
    """Render only `subset_mask` gaussians, with every gaussian's f_dc replaced
    by the SH DC value that corresponds to the desired RGB color. Lets us tint
    a layer for the color overlay without touching the rasterizer."""
    if subset_mask.sum() == 0:
        return np.zeros((mv.height, mv.width, 3), dtype=np.uint8)
    # Make a shallow-copy GaussianModel that shares everything with `model`
    # except f_dc, which is overwritten per-gaussian to the requested color.
    f_dc_override = np.tile(_rgb_to_f_dc(rgb)[None, :],
                            (len(model), 1)).astype(np.float32)
    tinted = GaussianModel(
        xyz=model.xyz, normals=model.normals, f_dc=f_dc_override,
        f_rest=model.f_rest, opacity=model.opacity, scale=model.scale,
        rotation=model.rotation, extra_fields=model.extra_fields,
    )
    return render_one(tinted, mv, device,
                      use_fallback=use_fallback, subset_mask=subset_mask)


def make_combined_object_figure(detection_tile_bgr: Optional[np.ndarray],
                                progression_bgr: Optional[np.ndarray],
                                overlay_bgr: Optional[np.ndarray],
                                views_strip_bgr: Optional[np.ndarray],
                                target_width: int = 2400,
                                section_titles: bool = False,
                                ) -> np.ndarray:
    """Stack the per-object panels into one paper-ready figure.

    Layout (vertical), with thin black separators between sections:
      [1] detection tile (single view with bbox)         -- proves YOLO saw it
      [2] progression strip (N tiles)                    -- progressive download quality
      [3] color overlay (single image + legend)          -- per-pixel dominant layer
      [4] multi-view strip                               -- 3D correctness QC

    Each panel that's None is skipped. All panels are resized to fit
    target_width and stacked. No overlaid captions on the figure itself
    (figure caption belongs in the paper text). Section_titles draws a small
    label per panel for readability outside the paper context.
    """
    panels: List[np.ndarray] = []
    section_labels: List[str] = []
    sep_h = 4  # px black gap between sections
    sep = None  # built once we know the width

    def resize_to_width(img: np.ndarray, w: int) -> np.ndarray:
        if img.shape[1] == w: return img
        sc = w / img.shape[1]
        new_h = max(1, int(round(img.shape[0] * sc)))
        return cv2.resize(img, (w, new_h), interpolation=cv2.INTER_AREA)

    for label, img in [('detection', detection_tile_bgr),
                        ('layers progressive', progression_bgr),
                        ('layer composition', overlay_bgr),
                        ('multi-view', views_strip_bgr)]:
        if img is None or img.size == 0:
            continue
        panels.append(resize_to_width(img, target_width))
        section_labels.append(label)

    if not panels:
        return np.zeros((100, target_width, 3), dtype=np.uint8)

    # Optional small section titles on the left edge in a faint colour.
    if section_titles:
        labelled = []
        for img, lbl in zip(panels, section_labels):
            band_h = 22
            band = np.zeros((band_h, target_width, 3), dtype=np.uint8)
            cv2.putText(band, lbl, (8, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (180, 180, 180), 1, cv2.LINE_AA)
            labelled.append(band)
            labelled.append(img)
        panels = labelled

    # Concatenate with thin separators.
    sep = np.zeros((sep_h, target_width, 3), dtype=np.uint8)
    parts = []
    for i, p in enumerate(panels):
        if i > 0:
            parts.append(sep)
        parts.append(p)
    return np.vstack(parts)


def make_layer_progression_image(model: GaussianModel,
                                 layer_masks: List[np.ndarray],
                                 mv: 'Movement', device: str,
                                 use_fallback: bool,
                                 obj_label: str,
                                 layer_meta: List[dict],
                                 show_captions: bool = False,
                                 mode: str = "cumulative") -> np.ndarray:
    """Build a horizontal strip of tiles, one per progressive layer.

    mode = "cumulative" (default):
        Tile K renders gaussians from layers 0, 1, ..., K (set UNION).
        Reads as "what the client sees as more layers stream in." Each tile
        is strictly more detailed than the previous; the rightmost is the
        full object/cluster.

    mode = "layer_only":
        Tile K renders gaussians from ONLY layer K (no earlier layers). So
        the base tile shows just the high-importance core; the enh1 tile
        shows ONLY the gaussians ranked 25-50% (with nothing else behind
        them); etc. Useful for visualizing each enhancement layer's
        independent contribution.

    show_captions: if True, overlays per-tile text labels and a footer.
    Default OFF for paper-ready images (the caption goes in LaTeX, not the
    pixel data)."""
    if not layer_masks:
        return np.zeros((100, 400, 3), dtype=np.uint8)
    if mode not in ("cumulative", "layer_only"):
        raise ValueError(f"mode must be 'cumulative' or 'layer_only', got '{mode}'")

    target_w = max(180, min(400, 2400 // len(layer_masks)))
    sc = target_w / mv.width
    tw = int(mv.width * sc); th = int(mv.height * sc)
    strip = np.zeros((th, len(layer_masks) * tw, 3), dtype=np.uint8)
    cumulative = np.zeros(len(model), dtype=bool)
    for k, lm in enumerate(layer_masks):
        if mode == "cumulative":
            cumulative = cumulative | lm
            render_mask = cumulative
        else:  # layer_only
            render_mask = lm
        img = render_one(model, mv, device, use_fallback=use_fallback,
                         subset_mask=render_mask)
        tile = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if (tile.shape[1], tile.shape[0]) != (tw, th):
            tile = cv2.resize(tile, (tw, th))
        strip[:, k*tw:(k+1)*tw] = tile
        if show_captions:
            meta = layer_meta[k] if k < len(layer_meta) else {}
            if mode == "cumulative":
                cum_pct = meta.get('cumulative_pct', 100*(k+1)//len(layer_masks))
                cum_n   = meta.get('cumulative_gaussians', int(cumulative.sum()))
                tag = f"L{k} {meta.get('name', '')} cum={cum_pct}% n={cum_n:,}"
            else:
                delta_n = meta.get('delta_gaussians', int(lm.sum()))
                tag = f"L{k} {meta.get('name', '')} ONLY n={delta_n:,}"
            cv2.putText(strip, tag[:36], (k*tw + 6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2,
                        cv2.LINE_AA)
    if show_captions:
        footer = ("progressive download (left=base, right=full)"
                  if mode == "cumulative"
                  else "per-layer contribution (each tile in isolation)")
        cv2.putText(strip,
                    f"{obj_label} -- {footer}",
                    (8, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return strip


def make_layer_color_overlay(model: GaussianModel,
                             layer_masks: List[np.ndarray],
                             mv: 'Movement', device: str,
                             use_fallback: bool,
                             obj_label: str,
                             colors: List[Tuple[float, float, float]] = None,
                             show_captions: bool = False) -> np.ndarray:
    """Per-pixel dominant-layer coloring. See module docstring for full
    explanation. show_captions toggles the overlaid title text on/off
    (the legend swatch row is always kept since it's needed to read the
    color encoding). Default OFF for paper-ready output."""
    if not layer_masks:
        return np.zeros((100, 400, 3), dtype=np.uint8)
    if colors is None:
        colors = LAYER_COLORS_RGB

    H, W = mv.height, mv.width

    per_layer_luminance: List[np.ndarray] = []
    for k, lm in enumerate(layer_masks):
        if lm.sum() == 0:
            per_layer_luminance.append(np.zeros((H, W), dtype=np.float32))
            continue
        c = colors[k % len(colors)]
        img_rgb = _render_with_color_override(model, mv, lm, c, device,
                                              use_fallback=use_fallback)
        lum = img_rgb.max(axis=2).astype(np.float32) / 255.0
        per_layer_luminance.append(lum)

    lum_stack = np.stack(per_layer_luminance, axis=0)
    dominant  = np.argmax(lum_stack, axis=0)
    max_lum   = lum_stack.max(axis=0)
    coverage_thresh = 0.05
    has_coverage = max_lum >= coverage_thresh

    out_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for k in range(len(layer_masks)):
        c = np.asarray(colors[k % len(colors)], dtype=np.float32)
        sel = has_coverage & (dominant == k)
        brightness = max_lum[sel][:, None]
        out_rgb[sel] = c[None, :] * brightness

    out_rgb = np.clip(out_rgb, 0, 1)
    out = (out_rgb * 255.0).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # Legend strip on the bottom -- kept regardless of show_captions because
    # the color encoding NEEDS a legend to be readable. It IS the figure's
    # caption equivalent; if you want a pure-image figure, set
    # --no_layer_legend at the CLI.
    legend_h = 28
    legend = np.zeros((legend_h, out.shape[1], 3), dtype=np.uint8)
    n_layers = len(layer_masks)
    swatch_w = max(60, min(180, out.shape[1] // (n_layers + 1)))
    x = 6
    for k in range(n_layers):
        rgb = colors[k % len(colors)]
        bgr = (int(255*rgb[2]), int(255*rgb[1]), int(255*rgb[0]))
        cv2.rectangle(legend, (x, 4), (x + 18, legend_h - 4), bgr, -1)
        name = "base" if k == 0 else f"enh{k}"
        cv2.putText(legend, f"L{k} {name}", (x + 22, legend_h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        x += swatch_w
    out_with_legend = np.vstack([out, legend])
    if show_captions:
        cv2.putText(out_with_legend,
                    f"{obj_label} -- per-pixel dominant layer",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out_with_legend


# ============================================================================
#  Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--camera_json", required=True)
    ap.add_argument("--frame_index", type=int, default=0)
    ap.add_argument("--outdir", required=True)

    # view-set
    ap.add_argument("--views", default="8",
                    help="preset name (4,6,8,8_corners,10,12) or custom "
                         "'az,el;az,el;...' list (e.g. '0,0;90,0;0,-30')")
    ap.add_argument("--lateral_offsets", default="0.0",
                    help="comma-separated translations (world units) of the "
                         "look-at target along the camera's right axis. "
                         "Try '-1.5,-0.5,0.5,1.5' for elongated scenes. "
                         "Cameras = views x lateral_offsets")
    ap.add_argument("--fov_scale", type=float, default=0.3,
                    help="multiply fx,fy. <1 widens (more visible). "
                         "Default 0.3 worked well for the bicycle scene.")
    ap.add_argument("--view_distance", type=float, default=None)

    # detector
    ap.add_argument("--detector", choices=["yolo", "yoloworld"], default="yolo",
                    help="'yolo' = COCO 80 classes (no trees/grass). "
                         "'yoloworld' = open-vocab; pass --text_prompts.")
    ap.add_argument("--yolo_model", default="yolo26x.pt", #you may want to change it to lower versions ... "yolo8x.pt"
                    help="yolov8x.pt for COCO; yolov8x-worldv2.pt for YOLO-World") #  you may want to change it to 26x.
    ap.add_argument("--text_prompts", default="",
                    help="comma-separated prompts for yoloworld, e.g. "
                         "'bicycle,bench,tree,grass,path,trash can,person'")
    ap.add_argument("--yolo_conf", type=float, default=0.20)

    # merging / extraction
    ap.add_argument("--iou_threshold", type=float, default=0.20,
                    help="3D-IoU threshold for merging same-class detections "
                         "across views into one object. 0.15-0.30 is typical.")
    ap.add_argument("--vote_threshold", type=float, default=0.5)
    ap.add_argument("--mask_pad", type=float, default=1.3,
                    help="how much to grow each detection's 2D bbox before "
                         "voting. 1.0 = no padding. 1.3 = 30% larger on every "
                         "side (recommended). 1.6+ = aggressive (may include "
                         "neighbors).")
    ap.add_argument("--depth_band_frac", type=float, default=0.35,
                    help="fraction of the in-bbox depth range to keep when "
                         "voting. 0.35 = keep the front 35%% (good for crisp "
                         "extraction, rejects line-of-sight background). "
                         "Raise to 0.5-0.7 if your object has lots of depth "
                         "extent (e.g. a long fence) and parts of it are "
                         "missing from the .ply.")

    # 3D dilation (recovers parts the detector missed due to occlusion).
    ap.add_argument("--dilate_radius_frac", type=float, default=0.10,
                    help="grow each object's gaussian set by a 3D radius equal "
                         "to this fraction of the object's own extent. 0.10 = "
                         "10%% of object diameter (recommended). 0.0 disables "
                         "dilation. Higher values recover more parts but risk "
                         "absorbing nearby clutter.")
    ap.add_argument("--dilate_radius_abs", type=float, default=None,
                    help="absolute dilation radius in WORLD UNITS (overrides "
                         "--dilate_radius_frac). Use this when you know the "
                         "scene's scale -- e.g. 0.05 means 5cm dilation. None "
                         "= use the fractional radius.")
    ap.add_argument("--dilate_radius_max", type=float, default=None,
                    help="hard cap (world units) on the dilation radius for "
                         "any one object, applied AFTER fraction. Useful when "
                         "one object has unusually large extent and you don't "
                         "want it to expand wildly.")
    ap.add_argument("--no_dilate", action="store_true",
                    help="disable 3D dilation entirely (equivalent to "
                         "--dilate_radius_frac 0).")
    ap.add_argument("--dilate_chunk", type=int, default=200_000,
                    help="how many scene gaussians to query at once during "
                         "dilation. Lower values use less peak RAM. Default "
                         "200000 is fine for most workloads. Drop to 50000 if "
                         "you OOM on very large scenes (>20M gaussians).")

    ap.add_argument("--no_other_ply", action="store_true",
                    help="skip writing the 'other.ply' that contains all "
                         "Gaussians not assigned to any detected object.")

    # Progressive (scalable) layers.
    ap.add_argument("--no_layers", action="store_true",
                    help="skip writing progressive .ply layers per object. "
                         "When OFF (default), each object writes 5 .ply "
                         "delta-layers for streaming, plus a manifest.json.")
    ap.add_argument("--layer_percentages", default="25,50,75,100",
                    help="comma-separated CUMULATIVE percentages defining "
                         "where each layer ends. Default '25,50,75,100' "
                         "gives 4 layers: base (25%%) + 3 enhancements -- "
                         "matching the 4-color legend (red/blue/yellow/"
                         "green). Pass '20,40,60,80,100' for 5 layers (the "
                         "5th layer uses an orange fallback color).")
    ap.add_argument("--layer_rank", default="opacity_scale",
                    choices=["opacity", "opacity_scale", "scale"],
                    help="how to rank gaussians within an object so the most "
                         "important go in the base layer. 'opacity' = sigmoid "
                         "of opacity logit. 'opacity_scale' = opacity * max "
                         "world-space scale (approximates screen-space "
                         "contribution; recommended). 'scale' = max scale only.")
    ap.add_argument("--no_layer_full_ply", action="store_true",
                    help="don't also write a single full .ply per object "
                         "alongside the layers. With this flag, only the "
                         "delta-layer files are written (saves ~1x object "
                         "size on disk per object). Streaming clients don't "
                         "need the full file.")
    ap.add_argument("--no_layer_viz", action="store_true",
                    help="skip the per-object layer visualization PNGs "
                         "(progression strip + color overlay). Saves time "
                         "if you don't need to QC the layering visually.")
    ap.add_argument("--object_layer_only_viz", action="store_true",
                    help="also produce a 'layer_only' strip for each object "
                         "(each layer rendered in ISOLATION instead of "
                         "cumulatively). Useful for visualizing each "
                         "enhancement layer's independent contribution. "
                         "Always produced for clusters; this flag enables "
                         "it for individual objects as well.")
    ap.add_argument("--no_combined_figure", action="store_true",
                    help="skip the combined per-object PNG that stacks the "
                         "detection-view tile + progression strip + color "
                         "overlay + multi-view check into a single image. "
                         "Default ON (one combined figure per object, paper "
                         "ready). Pass this flag to suppress it.")
    ap.add_argument("--show_captions", action="store_true",
                    help="overlay descriptive captions on visualization PNGs "
                         "(per-tile labels, footer text). OFF by default for "
                         "paper-ready figures whose captions live in LaTeX, "
                         "not in the pixels.")
    ap.add_argument("--preview_fit_fraction", type=float, default=0.70,
                    help="when rendering per-object visualizations from an "
                         "object-centered camera, what fraction of the image "
                         "height the object's largest dimension should fill. "
                         "0.70 = object fills 70%% of the frame (leaves room "
                         "for context). Smaller = more zoomed out, larger = "
                         "more zoomed in.")

    # Object clustering (groups physically-close objects into joint .plys).
    ap.add_argument("--clustering", action="store_true",
                    help="ALSO produce per-cluster .plys that bundle "
                         "physically-close objects. When objects touch or "
                         "nearly touch (e.g. a bicycle leaning on a bench), "
                         "the per-object spatial gates can clip gaussians at "
                         "the shared boundary -- streaming a unified cluster "
                         ".ply removes the seam. Cluster files are produced "
                         "IN ADDITION TO individual object .plys; the "
                         "streaming server can choose at query time which to "
                         "send. Lone objects with no close neighbor become "
                         "singleton clusters.")
    ap.add_argument("--cluster_method", default="dbscan",
                    choices=["gap", "dbscan", "kmeans"],
                    help="how to cluster objects. 'gap' (default, recommended): "
                         "single-linkage on bbox gap distance -- two objects "
                         "are in the same cluster if their 3D bboxes are within "
                         "--cluster_eps_* of each other. 'dbscan': sklearn "
                         "DBSCAN on centroids. 'kmeans': forced K partition "
                         "(ablation only; ignores closeness).")
    ap.add_argument("--cluster_eps_frac", type=float, default=0.03,
                    help="cluster threshold as a FRACTION of the scene "
                         "diagonal (default 0.03 = 3%% of scene size). For a "
                         "typical outdoor scene of ~10m diagonal this is "
                         "0.3m, which merges touching objects without pulling "
                         "in distant ones. Only used when --cluster_eps_abs "
                         "is not set.")
    ap.add_argument("--cluster_eps_abs", type=float, default=None,
                    help="cluster threshold in ABSOLUTE world units "
                         "(overrides --cluster_eps_frac). For 'gap' this is "
                         "the max bbox gap in metres; for 'dbscan' it's the "
                         "max centroid distance.")
    ap.add_argument("--cluster_kmeans_k", type=int, default=4,
                    help="number of clusters for --cluster_method kmeans "
                         "(ablation only).")
    ap.add_argument("--cluster_skip_other", action="store_true",
                    help="don't include the 'other' (background) mask in "
                         "clustering output. By default the background gets "
                         "its own singleton cluster so the streaming server "
                         "has a uniform interface.")

    # rendering
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_fallback_rasterizer", action="store_true")
    ap.add_argument("--no_object_previews", action="store_true",
                    help="skip the per-object verification PNGs")
    ap.add_argument("--object_preview_views", default="all",
                    help="how many views to render each object from for "
                         "verification: 'all' (one render per pipeline view, "
                         "best for accuracy QC -- default), 'best' (single "
                         "highest-confidence view, fastest), or an integer N "
                         "to use the N most-detected views. Each object's "
                         "row in objects_contact_sheet.png will have one "
                         "tile per view rendered.")
    ap.add_argument("--preview_trajectory", default=None,
                    help="optional path to a SECOND camera JSON file (e.g. a "
                         "client video trajectory using view_matrix+fov "
                         "format). When set, per-object verification PNGs are "
                         "rendered from these EXACT camera frames instead of "
                         "the synthetic detection views. Use this to confirm "
                         "the extracted .ply will look correct in your client.")
    ap.add_argument("--preview_trajectory_max_frames", type=int, default=12,
                    help="max number of trajectory frames to use for previews "
                         "(uniformly subsampled across the trajectory).")
    ap.add_argument("--preview_only", action="store_true",
                    help="render the views and stop (no detection, no export)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img_dir = os.path.join(args.outdir, "renders");    os.makedirs(img_dir, exist_ok=True)
    det_dir = os.path.join(args.outdir, "detections"); os.makedirs(det_dir, exist_ok=True)
    obj_dir = os.path.join(args.outdir, "objects");    os.makedirs(obj_dir, exist_ok=True)

    # 1. load
    model = load_gaussian_ply(args.input)

    # 2. cameras
    base = load_camera_json(args.camera_json, args.frame_index)
    print(f"[cams] base: angle={base.angle:.2f} elev={base.elevation:.2f} "
          f"pos=({base.x:.3f},{base.y:.3f},{base.z:.3f}) "
          f"fx={base.fx} fy={base.fy} size={base.width}x{base.height}")

    if ';' in args.views:
        specs = [tuple(map(float, t.strip().split(','))) for t in args.views.split(';')]
    else:
        specs = view_preset(args.views)
    laterals = [float(x) for x in args.lateral_offsets.split(',')]
    print(f"[cams] views preset='{args.views}' ({len(specs)} angles) x "
          f"laterals={laterals} -> {len(specs)*len(laterals)} cameras  "
          f"fov_scale={args.fov_scale}")

    mvs = derive_views(base, specs, laterals,
                       view_distance=args.view_distance,
                       fov_scale=args.fov_scale)
    for mv in mvs:
        print(f"  -> {mv.name}  pos=({mv.x:+.3f},{mv.y:+.3f},{mv.z:+.3f}) "
              f"angle={mv.angle:+.2f}  elev={mv.elevation:+.2f}")

    # 3. render
    backend = "gsplat" if (HAS_GSPLAT and not args.use_fallback_rasterizer
                           and str(args.device).startswith("cuda")) else "fallback"
    print(f"[render] backend={backend}  device={args.device}")
    images_rgb: List[np.ndarray] = []
    t_total = 0.0
    for mv in mvs:
        t0 = time.time()
        img = render_one(model, mv, args.device,
                         use_fallback=args.use_fallback_rasterizer)
        dt = time.time() - t0; t_total += dt
        images_rgb.append(img)
        cv2.imwrite(os.path.join(img_dir, f"{mv.name}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  rendered {mv.name} in {dt:.2f}s")
    print(f"[render] total {t_total:.2f}s")

    # contact sheet (no black gaps; 4 cols)
    n = len(images_rgb)
    cols = min(4, n); rows = int(math.ceil(n / cols))
    scale = min(1.0, 480.0 / max(base.width, base.height))
    tw = int(base.width * scale); th = int(base.height * scale)
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, im in enumerate(images_rgb):
        r, c = i // cols, i % cols
        small = cv2.resize(im, (tw, th)) if scale < 1.0 else im
        sheet[r*th:(r+1)*th, c*tw:(c+1)*tw] = small
        if args.show_captions:
            cv2.putText(sheet, mvs[i].name, (c*tw + 8, r*th + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                        cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.outdir, "contact_sheet.png"),
                cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[render] contact sheet -> {args.outdir}/contact_sheet.png")

    if args.preview_only:
        print("[preview] preview_only set -- stopping.")
        return

    # 4. detect
    images_bgr = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images_rgb]
    text_prompts = [p.strip() for p in args.text_prompts.split(',') if p.strip()]
    detections_per_view = run_detector(images_bgr, args.yolo_model,
                                       detector=args.detector,
                                       text_prompts=text_prompts or None,
                                       conf=args.yolo_conf)
    for vi, (img, dets) in enumerate(zip(images_bgr, detections_per_view)):
        vis = img.copy()
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{d.class_name} {d.score:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(det_dir, f"{mvs[vi].name}_det.png"), vis)

    # 5. merge across views (this is the part that prevents duplicate plies)
    xyz_t = torch.from_numpy(model.xyz).to(args.device)
    objects = merge_detections(detections_per_view, model, xyz_t, mvs,
                               args.device, iou_threshold=args.iou_threshold)

    # 6. final per-Gaussian masks
    masks = build_final_object_masks(model, xyz_t, objects, mvs, args.device,
                                     vote_threshold=args.vote_threshold,
                                     mask_pad=args.mask_pad,
                                     depth_band_frac=args.depth_band_frac)

    # 6b. (optional) 3D dilation -- recovers occluded parts that 2D bbox voting
    # missed (e.g. bench seat hidden under grass; wheel hub clipped at bbox edge).
    # Each object grows by a 3D radius derived from its own extent. Conflicts
    # between dilated objects are resolved by "nearest seed wins"; gaussians
    # already in any seed are never overridden.
    if args.no_dilate or args.dilate_radius_frac <= 0.0:
        if args.dilate_radius_abs is None:
            print("[dilate] disabled (--no_dilate or radius_frac <= 0)")
            dilate_stats = {oi: dict(seed=int(m.sum()), added=0,
                                     final=int(m.sum()), radius=0.0,
                                     extent=_object_extent(model, m))
                            for oi, m in masks.items()}
        else:
            masks, dilate_stats = dilate_object_masks(
                model, masks, objects,
                radius_frac=0.0,
                radius_abs=args.dilate_radius_abs,
                radius_max_abs=args.dilate_radius_max,
                scene_chunk=args.dilate_chunk,
            )
    else:
        masks, dilate_stats = dilate_object_masks(
            model, masks, objects,
            radius_frac=args.dilate_radius_frac,
            radius_abs=args.dilate_radius_abs,
            radius_max_abs=args.dilate_radius_max,
            scene_chunk=args.dilate_chunk,
        )

    # 7. save .ply + verification renders per object (multi-view).
    # For each object we render its Gaussians from a chosen set of views and
    # build a horizontal strip of those renders. The strips become rows of the
    # final objects_contact_sheet.png so you can see each object from many
    # angles and judge whether the .ply contains the right Gaussians.
    def _select_preview_view_indices(obj: ObjectInstance, mode: str,
                                     n_views_total: int) -> List[int]:
        """Return the indices into `mvs` that we want to render the object from."""
        if mode == "best":
            return [max(obj.detections, key=lambda d: d.score).view_idx]
        if mode == "all":
            return list(range(n_views_total))
        # integer N: pick the N views in which this object was detected most
        # confidently; if it has fewer than N detections, supplement with the
        # closest-camera-angle views so we still have coverage.
        try:
            N = int(mode)
        except ValueError:
            raise ValueError(f"--object_preview_views must be 'all', 'best', "
                             f"or an integer; got '{mode}'")
        N = max(1, min(N, n_views_total))
        det_views = sorted({d.view_idx for d in obj.detections},
                           key=lambda vi: -max(d.score for d in obj.detections
                                                if d.view_idx == vi))
        chosen = det_views[:N]
        if len(chosen) < N:
            for i in range(n_views_total):
                if i not in chosen:
                    chosen.append(i)
                    if len(chosen) >= N: break
        return chosen[:N]

    def _bbox_for_view(obj: ObjectInstance, view_idx: int
                       ) -> Optional[Tuple[int,int,int,int]]:
        """If this object was detected in this view, return its highest-
        confidence bbox; else None."""
        candidates = [d for d in obj.detections if d.view_idx == view_idx]
        if not candidates: return None
        return max(candidates, key=lambda d: d.score).bbox

    def _make_object_strip(obj_or_none: Optional[ObjectInstance],
                           obj_label: str, n_assigned: int,
                           subset_mask: np.ndarray,
                           preview_mvs: List[Movement],
                           src_view_indices: Optional[List[int]] = None
                           ) -> np.ndarray:
        """Render `subset_mask` from each Movement in `preview_mvs`, concatenate
        horizontally, annotate. `src_view_indices` is the index into
        the original `mvs` list for each preview Movement (used to look up the
        bbox YOLO detected in that view); pass None when previewing from a
        client trajectory whose frames don't correspond to detection views.
        """
        target_tile_w = max(160, min(360, 2400 // max(1, len(preview_mvs))))
        # use the first preview movement's resolution for tile sizing
        ref_w = preview_mvs[0].width if preview_mvs else base.width
        ref_h = preview_mvs[0].height if preview_mvs else base.height
        scale = target_tile_w / max(1, ref_w)
        tw = int(ref_w * scale); th = int(ref_h * scale)
        strip = np.zeros((th, len(preview_mvs) * tw, 3), dtype=np.uint8)
        n_with_det = 0
        for k, mv in enumerate(preview_mvs):
            t0 = time.time()
            tile = render_one(model, mv, args.device,
                              use_fallback=args.use_fallback_rasterizer,
                              subset_mask=subset_mask)
            dt = time.time() - t0
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)

            bb = None
            src_vi = src_view_indices[k] if src_view_indices is not None else None
            if obj_or_none is not None and src_vi is not None:
                bb = _bbox_for_view(obj_or_none, src_vi)
            if bb is not None:
                cv2.rectangle(tile_bgr, (bb[0], bb[1]), (bb[2], bb[3]),
                              (0, 255, 0), 2)
                n_with_det += 1
            # resize: handle case where preview_mvs have different sizes
            if (tile_bgr.shape[1], tile_bgr.shape[0]) != (tw, th):
                tile_bgr = cv2.resize(tile_bgr, (tw, th))
            strip[:, k*tw:(k+1)*tw] = tile_bgr

            tag = mv.name if src_view_indices is None else \
                  f"v{src_vi:02d}" + (" *" if bb is not None else "")
            if args.show_captions:
                cv2.putText(strip, tag[:18], (k*tw + 6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                            cv2.LINE_AA)
            print(f"    obj={obj_label:<14} {mv.name}  {dt:.2f}s")
        if args.show_captions:
            label = (f"{obj_label} | n={n_assigned:,}"
                     + (f" | detected_in={n_with_det}/{len(preview_mvs)} views"
                        if src_view_indices is not None else ""))
            cv2.putText(strip, label, (8, th - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return strip

    # If --preview_trajectory is set, load it once: per-object preview tiles
    # will come from these EXACT camera frames (so what you see is what the
    # client sees). Otherwise we fall back to the synthetic detection views.
    trajectory_preview_mvs: Optional[List[Movement]] = None
    if args.preview_trajectory:
        trajectory_preview_mvs = load_trajectory_as_movements(
            args.preview_trajectory,
            max_frames=args.preview_trajectory_max_frames,
        )
        print(f"[preview] loaded {len(trajectory_preview_mvs)} frames from "
              f"trajectory {args.preview_trajectory}; per-object PNGs will be "
              f"rendered from these (the client's actual viewpoints).")

    # Parse layer percentages once.
    try:
        layer_pcts = [float(x.strip()) for x in args.layer_percentages.split(',')
                      if x.strip()]
    except ValueError:
        raise ValueError(f"--layer_percentages must be a comma-separated list "
                         f"of numbers, got '{args.layer_percentages}'")
    if not args.no_layers:
        print(f"[layers] writing progressive layers at cumulative percentages "
              f"{layer_pcts} (rank='{args.layer_rank}')")

    # collect manifests for the top-level summary
    object_manifests: List[Tuple[int, str, dict]] = []   # (oi or -1, label, manifest)

    print(f"[preview] rendering per-object strips "
          f"(--object_preview_views={args.object_preview_views})")
    object_strips: List[Tuple[int, str, np.ndarray]] = []
    for oi, obj in enumerate(objects):
        m = masks.get(oi, np.zeros(len(model), dtype=bool))
        if m.sum() == 0:
            continue
        base_name = f"obj_{oi:03d}_{obj.class_name}_n{int(m.sum())}"

        # Write progressive layers (delta layers + manifest) into a per-object dir.
        per_obj_layer_masks: List[np.ndarray] = []
        per_obj_layer_meta: List[dict] = []
        if not args.no_layers:
            layer_subdir = os.path.join(obj_dir, base_name)
            manifest, per_obj_layer_masks = write_progressive_layers(
                model, m, layer_subdir,
                object_label=f"{obj.class_name}_obj{oi:03d}",
                percentages=layer_pcts,
                rank_method=args.layer_rank,
                write_full=not args.no_layer_full_ply,
            )
            per_obj_layer_meta = manifest.get("layers", [])
            object_manifests.append((oi, obj.class_name, manifest))
            ply_path = os.path.join(layer_subdir,
                                    manifest.get("full_file") or
                                    manifest["layers"][-1]["file"])
        else:
            # legacy single-file behaviour
            ply_path = os.path.join(obj_dir, base_name + ".ply")
            save_gaussian_ply(model, m, ply_path)

        if args.no_object_previews:
            continue
        if trajectory_preview_mvs is not None:
            preview_mvs_for_obj = trajectory_preview_mvs
            src_idxs = None       # client trajectory has no detection bboxes
        else:
            view_idxs = _select_preview_view_indices(
                obj, args.object_preview_views, len(mvs))
            preview_mvs_for_obj = [mvs[i] for i in view_idxs]
            src_idxs = view_idxs
        strip = _make_object_strip(obj, obj.class_name, int(m.sum()), m,
                                   preview_mvs_for_obj, src_idxs)
        png_path = os.path.join(obj_dir, base_name + "_views.png")
        cv2.imwrite(png_path, strip)
        object_strips.append((oi, ply_path, strip))

        # Layer-quality visualizations (one tile per layer + a color overlay),
        # Layer-quality visualizations rendered from a camera CENTERED ON THIS
        # OBJECT (not the YOLO detection view, which may have framed the
        # object too small or off-center). The detection-view tile is still
        # produced as a separate panel for the combined figure.
        if (not args.no_layers and not args.no_layer_viz
                and per_obj_layer_masks):
            best_det = max(obj.detections, key=lambda d: d.score)
            det_mv = mvs[best_det.view_idx]

            # Build an object-centered camera that frames just this object.
            centroid, extent = _object_bbox_extent(model, m)
            obj_mv = build_camera_looking_at(
                centroid, det_mv, extent,
                fit_fraction=args.preview_fit_fraction,
            )
            print(f"[layer-viz] {obj.class_name} obj{oi:03d}: "
                  f"centroid={np.round(centroid, 3).tolist()} "
                  f"extent={extent:.3f} -- rendering from obj-centered cam")

            prog = make_layer_progression_image(
                model, per_obj_layer_masks, obj_mv, args.device,
                args.use_fallback_rasterizer,
                f"{obj.class_name} (obj {oi})", per_obj_layer_meta,
                show_captions=args.show_captions,
                mode="cumulative",
            )
            cv2.imwrite(os.path.join(obj_dir,
                                     base_name + "_layers_progression.png"),
                        prog)
            # Optionally also produce the layer-only strip for this object.
            # Off by default to keep run-time small; clusters always get it.
            if args.object_layer_only_viz:
                layer_only = make_layer_progression_image(
                    model, per_obj_layer_masks, obj_mv, args.device,
                    args.use_fallback_rasterizer,
                    f"{obj.class_name} (obj {oi})", per_obj_layer_meta,
                    show_captions=args.show_captions,
                    mode="layer_only",
                )
                cv2.imwrite(os.path.join(obj_dir,
                                         base_name + "_layers_only.png"),
                            layer_only)
            overlay = make_layer_color_overlay(
                model, per_obj_layer_masks, obj_mv, args.device,
                args.use_fallback_rasterizer,
                f"{obj.class_name} (obj {oi})",
                show_captions=args.show_captions,
            )
            cv2.imwrite(os.path.join(obj_dir,
                                     base_name + "_layers_colors.png"),
                        overlay)

            # Combined figure: detection-view tile + progression + overlay +
            # multi-view check, stacked vertically. Paper-ready by default
            # (no overlaid captions).
            if not args.no_combined_figure:
                # Detection view (single tile WITH bbox drawn).
                det_img = render_one(model, det_mv, args.device,
                                     use_fallback=args.use_fallback_rasterizer,
                                     subset_mask=m)
                det_tile = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
                x1, y1, x2, y2 = best_det.bbox
                cv2.rectangle(det_tile, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # multi-view strip = the existing per-object views PNG we
                # already built ('strip' variable a few lines above)
                combined = make_combined_object_figure(
                    detection_tile_bgr=det_tile,
                    progression_bgr=prog,
                    overlay_bgr=overlay,
                    views_strip_bgr=strip,
                    target_width=2400,
                    section_titles=args.show_captions,
                )
                cv2.imwrite(os.path.join(obj_dir,
                                         base_name + "_combined.png"),
                            combined)

    # 8. "other" PLY = everything not assigned to a detected object.
    other_mask = build_other_mask(masks, len(model))
    n_other = int(other_mask.sum())
    n_assigned = len(model) - n_other
    print(f"[other] {n_assigned:,} assigned to objects, "
          f"{n_other:,} remaining -> other.ply")
    if not args.no_other_ply and n_other > 0:
        other_layer_masks: List[np.ndarray] = []
        other_layer_meta: List[dict] = []
        if not args.no_layers:
            other_subdir = os.path.join(obj_dir, f"other_n{n_other}")
            manifest, other_layer_masks = write_progressive_layers(
                model, other_mask, other_subdir,
                object_label="other",
                percentages=layer_pcts,
                rank_method=args.layer_rank,
                write_full=not args.no_layer_full_ply,
            )
            other_layer_meta = manifest.get("layers", [])
            object_manifests.append((-1, "other", manifest))
            other_ply_path = os.path.join(other_subdir,
                                          manifest.get("full_file") or
                                          manifest["layers"][-1]["file"])
        else:
            other_ply_path = os.path.join(obj_dir, f"other_n{n_other}.ply")
            save_gaussian_ply(model, other_mask, other_ply_path)

        if not args.no_object_previews:
            if trajectory_preview_mvs is not None:
                preview_mvs_for_other = trajectory_preview_mvs
                src_idxs_other = None
            else:
                if args.object_preview_views == "all":
                    step = max(1, len(mvs) // 8)
                    idxs = list(range(0, len(mvs), step))[:8]
                elif args.object_preview_views == "best":
                    idxs = [0]
                else:
                    try:
                        N = max(1, min(int(args.object_preview_views), len(mvs)))
                    except ValueError:
                        N = 4
                    step = max(1, len(mvs) // N)
                    idxs = list(range(0, len(mvs), step))[:N]
                preview_mvs_for_other = [mvs[i] for i in idxs]
                src_idxs_other = idxs

            strip = _make_object_strip(None, "other", n_other, other_mask,
                                       preview_mvs_for_other, src_idxs_other)
            other_png = os.path.join(obj_dir, f"other_n{n_other}_views.png")
            cv2.imwrite(other_png, strip)
            object_strips.append((-1, other_ply_path, strip))

            # Layer-quality visualizations for the "other" mask. The
            # background spans most of the scene so we don't try to "center
            # the camera on the object" -- we keep the trajectory's first
            # frame (or the first synthetic detection view) which already
            # frames the bulk of the scene reasonably.
            if (not args.no_layers and not args.no_layer_viz
                    and other_layer_masks):
                viz_mv = (trajectory_preview_mvs[0]
                          if trajectory_preview_mvs else mvs[0])
                print(f"[layer-viz] other: rendering progression + color overlay "
                      f"from view {viz_mv.name}")
                prog = make_layer_progression_image(
                    model, other_layer_masks, viz_mv, args.device,
                    args.use_fallback_rasterizer,
                    "other (background)", other_layer_meta,
                    show_captions=args.show_captions,
                )
                cv2.imwrite(os.path.join(obj_dir,
                                         f"other_n{n_other}_layers_progression.png"),
                            prog)
                overlay = make_layer_color_overlay(
                    model, other_layer_masks, viz_mv, args.device,
                    args.use_fallback_rasterizer,
                    "other (background)",
                    show_captions=args.show_captions,
                )
                cv2.imwrite(os.path.join(obj_dir,
                                         f"other_n{n_other}_layers_colors.png"),
                            overlay)

                if not args.no_combined_figure:
                    # 'other' has no YOLO detection -> no bbox tile.
                    combined = make_combined_object_figure(
                        detection_tile_bgr=None,
                        progression_bgr=prog,
                        overlay_bgr=overlay,
                        views_strip_bgr=strip,
                        target_width=2400,
                        section_titles=args.show_captions,
                    )
                    cv2.imwrite(os.path.join(obj_dir,
                                             f"other_n{n_other}_combined.png"),
                                combined)

    # objects contact sheet -- stack one strip per object as rows
    if object_strips and not args.no_object_previews:
        max_w = max(s.shape[1] for _, _, s in object_strips)
        total_h = sum(s.shape[0] for _, _, s in object_strips)
        sheet = np.zeros((total_h, max_w, 3), dtype=np.uint8)
        y = 0
        for _oi, _ply, strip in object_strips:
            h, w = strip.shape[:2]
            sheet[y:y+h, :w] = strip
            y += h
        cv2.imwrite(os.path.join(args.outdir, "objects_contact_sheet.png"), sheet)
        print(f"[done] objects contact sheet -> "
              f"{args.outdir}/objects_contact_sheet.png  "
              f"({sheet.shape[1]}x{sheet.shape[0]})")

    # ========================================================================
    #  Clustering pass -- optional, opt-in via --clustering. Groups physically
    #  close objects into cluster .plys ALONGSIDE the per-object .plys. The
    #  cluster .ply is the gaussian union of its member objects, which
    #  removes seam artifacts at touching boundaries (e.g. bicycle leaning
    #  on bench) since the dilated, multi-label masks naturally overlap there.
    # ========================================================================
    cluster_manifests: List[Tuple[int, str, dict]] = []
    cluster_groups: List[List[int]] = []
    if args.clustering and masks:
        cluster_dir = os.path.join(args.outdir, "clusters")
        os.makedirs(cluster_dir, exist_ok=True)

        # Resolve eps. For gap/dbscan, eps is in world units; for kmeans it's
        # unused. We let --cluster_eps_abs override --cluster_eps_frac.
        if args.cluster_eps_abs is not None:
            cluster_eps = float(args.cluster_eps_abs)
            eps_src = "abs"
        else:
            cluster_eps = args.cluster_eps_frac * _scene_diagonal(model)
            eps_src = f"frac={args.cluster_eps_frac}"
        print(f"\n[cluster] method='{args.cluster_method}' "
              f"eps={cluster_eps:.3f} ({eps_src})")

        # Optionally include 'other' as a cluster-eligible mask. The
        # background is huge and would gobble everything under gap-distance,
        # so we treat it specially: it gets its own singleton cluster
        # afterwards, NOT merged into other clusters' gap computation.
        if args.cluster_method == "gap":
            cluster_groups = cluster_objects_gap(model, masks, cluster_eps)
        elif args.cluster_method == "dbscan":
            cluster_groups = cluster_objects_dbscan(model, masks, cluster_eps,
                                                    min_samples=1)
        elif args.cluster_method == "kmeans":
            cluster_groups = cluster_objects_kmeans(model, masks,
                                                    args.cluster_kmeans_k)

        # Cluster summary
        print(f"[cluster] formed {len(cluster_groups)} cluster(s) from "
              f"{len(masks)} object(s):")
        for ci, grp in enumerate(cluster_groups):
            members = [objects[oi].class_name if oi < len(objects) else f"obj{oi}"
                       for oi in grp]
            print(f"  cluster {ci:02d}: {len(grp)} objects -> {members}")

        # Build cluster masks (union of member objects' gaussians).
        cluster_masks = build_cluster_masks(masks, cluster_groups, len(model))

        # Helper: pick a reference camera for a cluster's visualizations.
        # We use the highest-confidence YOLO detection view from whichever
        # member object has the most-confident detection. This gives a
        # reasonable starting orientation for the object-centered camera
        # (which then auto-frames the cluster's full extent). For the
        # "other"/background cluster, fall back to the first detection view.
        def _ref_mv_for_cluster(group: List[int]) -> Movement:
            best_score = -1.0
            best_view = 0
            for oi in group:
                if oi < 0 or oi >= len(objects):
                    continue
                for d in objects[oi].detections:
                    if d.score > best_score:
                        best_score = d.score
                        best_view = d.view_idx
            return mvs[best_view] if mvs else base

        # Helper: render all four cluster visualizations and save them.
        # `mode_cumulative` and `mode_layer_only` use the SAME object-centered
        # camera, so the two strips align spatially -- you can read tile k of
        # one alongside tile k of the other and see "this is what enh_k adds
        # on its own" vs "this is what the client has after enh_k arrives".
        def _emit_cluster_viz(ci: int, base_name: str, sub: str,
                              cmask: np.ndarray, cmf: dict,
                              ref_mv: Movement) -> Tuple[Optional[np.ndarray],
                                                          Optional[np.ndarray],
                                                          Optional[np.ndarray]]:
            """Returns (cumulative_strip, layer_only_strip, color_overlay)
            in BGR, or (None, None, None) if viz is disabled or empty."""
            if (args.no_layers or args.no_layer_viz or
                    not cmf.get("layers") or cmask.sum() == 0):
                return None, None, None
            # Re-derive the per-layer masks from the manifest. Each layer
            # entry tells us which gaussians belong to it via the importance
            # ordering. The manifest doesn't store the masks directly (they'd
            # blow up its size), so we rebuild them from the rank.
            scores = _gaussian_importance(model, cmask,
                                          rank_method=args.layer_rank)
            obj_indices = np.where(cmask)[0]
            order = np.argsort(-scores)
            sorted_indices = obj_indices[order]
            layer_masks_local: List[np.ndarray] = []
            for L in cmf["layers"]:
                start = L["cumulative_gaussians"] - L["delta_gaussians"]
                end   = L["cumulative_gaussians"]
                lm = np.zeros(len(model), dtype=bool)
                lm[sorted_indices[start:end]] = True
                layer_masks_local.append(lm)

            # Object-centered camera that frames the WHOLE cluster.
            centroid, extent = _object_bbox_extent(model, cmask)
            cam = build_camera_looking_at(
                centroid, ref_mv, extent,
                fit_fraction=args.preview_fit_fraction,
            )
            label_pretty = f"cluster {ci}"

            # Persist the resolved object-centered camera into the cluster's
            # manifest so downstream tools (e.g. the VMAF orbit-trajectory
            # generator) can reproduce the EXACT same first frame the user
            # sees in the paper figure. Stored in legacy format -- the same
            # convention create_viewmat() consumes, so anyone reading this
            # back can plug it straight into a Movement.
            cmf["preview_camera"] = {
                "format":   "legacy",
                "angle":    float(cam.angle),
                "elevation": float(cam.elevation),
                "x":        float(cam.x),
                "y":        float(cam.y),
                "z":        float(cam.z),
                "fx":       float(cam.fx),
                "fy":       float(cam.fy),
                "cx":       float(cam.cx),
                "cy":       float(cam.cy),
                "width":    int(cam.width),
                "height":   int(cam.height),
                "centroid": [float(c) for c in centroid],
                "extent":   float(extent),
                "fit_fraction": float(args.preview_fit_fraction),
            }

            # 1) Cumulative progression strip (what the client sees over time).
            prog = make_layer_progression_image(
                model, layer_masks_local, cam, args.device,
                args.use_fallback_rasterizer,
                label_pretty, cmf["layers"],
                show_captions=args.show_captions,
                mode="cumulative",
            )
            cv2.imwrite(os.path.join(args.outdir, "clusters",
                                     base_name + "_layers_progression.png"),
                        prog)

            # 2) Per-layer-only strip (each layer rendered in isolation).
            #    Tile 0 = base alone. Tile 1 = enh1 alone (NOT base+enh1).
            #    Tile 2 = enh2 alone. Etc.
            layer_only = make_layer_progression_image(
                model, layer_masks_local, cam, args.device,
                args.use_fallback_rasterizer,
                label_pretty, cmf["layers"],
                show_captions=args.show_captions,
                mode="layer_only",
            )
            cv2.imwrite(os.path.join(args.outdir, "clusters",
                                     base_name + "_layers_only.png"),
                        layer_only)

            # 3) Per-pixel dominant-layer color overlay.
            overlay = make_layer_color_overlay(
                model, layer_masks_local, cam, args.device,
                args.use_fallback_rasterizer,
                label_pretty,
                show_captions=args.show_captions,
            )
            cv2.imwrite(os.path.join(args.outdir, "clusters",
                                     base_name + "_layers_colors.png"),
                        overlay)
            return prog, layer_only, overlay

        # Write each cluster as progressive layers + manifest + visualizations.
        for ci, (grp, cmask) in enumerate(zip(cluster_groups, cluster_masks)):
            if cmask.sum() == 0: continue
            classes = sorted({objects[oi].class_name for oi in grp
                              if oi < len(objects)})
            label_str = "_".join(classes) if classes else "cluster"
            base_name = f"cluster_{ci:03d}_{label_str}_n{int(cmask.sum())}"
            sub = os.path.join(cluster_dir, base_name)
            mf, _ = write_progressive_layers(
                model, cmask, sub,
                object_label=f"cluster_{ci:03d}",
                percentages=layer_pcts,
                rank_method=args.layer_rank,
                write_full=not args.no_layer_full_ply,
            )
            mf["object_ids"]      = grp
            mf["object_classes"]  = [objects[oi].class_name for oi in grp
                                     if oi < len(objects)]
            mf["dir_relative"]    = os.path.relpath(sub, args.outdir).replace(os.sep, "/")
            cluster_manifests.append((ci, label_str, mf))

            # Visualizations (paper-ready: no bbox, no multi-view QC strip --
            # clusters aren't object detections so we omit those panels).
            print(f"[cluster-viz] cluster {ci:03d} [{label_str}]: rendering")
            prog, layer_only, overlay = _emit_cluster_viz(
                ci, base_name, sub, cmask, mf, _ref_mv_for_cluster(grp))

            # Combined paper-ready figure: cumulative + layer-only + color
            # overlay, stacked vertically. NO YOLO bbox panel (clusters
            # aren't single detections), NO multi-view QC strip (object-level
            # concern, not cluster-level).
            if not args.no_combined_figure and prog is not None:
                combined = make_combined_object_figure(
                    detection_tile_bgr=None,        # clusters: no bbox panel
                    progression_bgr=prog,
                    overlay_bgr=overlay,
                    views_strip_bgr=layer_only,     # repurpose the 4th slot
                    target_width=2400,
                    section_titles=args.show_captions,
                )
                cv2.imwrite(os.path.join(cluster_dir,
                                         base_name + "_combined.png"),
                            combined)

        # Optionally write 'other' as its own singleton cluster, so the
        # streaming server sees a uniform "everything is a cluster" world.
        if (not args.cluster_skip_other) and n_other > 0:
            ci = len(cluster_manifests)
            base_name = f"cluster_{ci:03d}_other_n{n_other}"
            sub = os.path.join(cluster_dir, base_name)
            mf, _ = write_progressive_layers(
                model, other_mask, sub,
                object_label=f"cluster_{ci:03d}_other",
                percentages=layer_pcts,
                rank_method=args.layer_rank,
                write_full=not args.no_layer_full_ply,
            )
            mf["object_ids"]     = []
            mf["object_classes"] = ["other"]
            mf["dir_relative"]   = os.path.relpath(sub, args.outdir).replace(os.sep, "/")
            cluster_manifests.append((ci, "other", mf))

            # 'other' visualization uses the first detection view as
            # reference. The background spans most of the scene so it'll be
            # framed by build_camera_looking_at to fit its full extent.
            print(f"[cluster-viz] cluster {ci:03d} [other]: rendering")
            prog, layer_only, overlay = _emit_cluster_viz(
                ci, base_name, sub, other_mask, mf,
                mvs[0] if mvs else base)
            if not args.no_combined_figure and prog is not None:
                combined = make_combined_object_figure(
                    detection_tile_bgr=None,
                    progression_bgr=prog,
                    overlay_bgr=overlay,
                    views_strip_bgr=layer_only,
                    target_width=2400,
                    section_titles=args.show_captions,
                )
                cv2.imwrite(os.path.join(cluster_dir,
                                         base_name + "_combined.png"),
                            combined)

        # Cluster catalog -- a separate JSON so clients that only care about
        # clusters can pull this one file. (The top-level manifest below also
        # references the clusters.)
        catalog = {
            "scene":           os.path.basename(args.input),
            "method":          args.cluster_method,
            "eps":             cluster_eps,
            "eps_source":      eps_src,
            "n_clusters":      len(cluster_manifests),
            "n_source_objects": len(masks),
            "clusters":        [],
        }
        for ci, lbl, mf in cluster_manifests:
            catalog["clusters"].append({
                "cluster_id":     ci,
                "label":          lbl,
                "n_objects":      len(mf.get("object_ids", [])),
                "object_ids":     mf.get("object_ids", []),
                "object_classes": mf.get("object_classes", []),
                "n_gaussians":    mf["n_total"],
                "total_bytes":    mf.get("total_bytes", 0),
                "total_kb":       mf.get("total_kb", 0.0),
                "dir_relative":   mf.get("dir_relative", ""),
                "n_layers":       len(mf.get("layers", [])),
                # Camera that produced the cluster's preview images. Stored
                # so downstream tools (VMAF orbit generator) can rebuild the
                # exact same first-frame viewpoint without re-running
                # segmentation.
                "preview_camera": mf.get("preview_camera"),
            })
        with open(os.path.join(args.outdir, "clusters_catalog.json"), "w") as f:
            json.dump(catalog, f, indent=2)
        print(f"[cluster] catalog -> {args.outdir}/clusters_catalog.json")

    # Top-level manifest -- single JSON file for the streaming server to read.
    # Lists every object (and "other") with its layers, file paths, sizes,
    # and cumulative gaussian counts. The server uses this as the catalog of
    # what can be streamed at each fidelity level.
    if not args.no_layers and object_manifests:
        # Per-layer-index aggregates ACROSS all objects: when a client
        # requests "everything at quality level k", how many bytes/gaussians
        # in total does the server send? This is the key streaming-paper
        # number.
        n_layers_max = max(len(mf["layers"]) for _, _, mf in object_manifests)
        per_level_aggregate = []
        BITRATES_MBPS = [1, 5, 10, 25, 50, 100]
        for k in range(n_layers_max):
            delta_bytes = 0
            cum_bytes = 0
            delta_g = 0
            cum_g = 0
            for _, _, mf in object_manifests:
                if k < len(mf["layers"]):
                    L = mf["layers"][k]
                    delta_bytes += L["size_bytes"]
                    delta_g     += L["delta_gaussians"]
                    cum_bytes   += L["cumulative_bytes"]
                    cum_g       += L["cumulative_gaussians"]
                else:
                    # this object has fewer layers; its "level k" = its last layer
                    last = mf["layers"][-1]
                    cum_bytes += last["cumulative_bytes"]
                    cum_g     += last["cumulative_gaussians"]
            download_ms = {
                f"{r}Mbps_ms": round(1000.0 * cum_bytes / (r * 125_000.0), 2)
                for r in BITRATES_MBPS
            }
            per_level_aggregate.append({
                "level":                  k,
                "name":                   "base" if k == 0 else f"enh{k}",
                "delta_bytes_scene":      delta_bytes,
                "delta_kb_scene":         round(delta_bytes / 1024.0, 2),
                "delta_gaussians_scene":  delta_g,
                "cumulative_bytes_scene": cum_bytes,
                "cumulative_kb_scene":    round(cum_bytes / 1024.0, 2),
                "cumulative_mb_scene":    round(cum_bytes / (1024.0**2), 3),
                "cumulative_gaussians_scene": cum_g,
                "download_cumulative_ms": download_ms,
            })

        total_bytes_scene = sum(mf.get("total_bytes", 0)
                                for _, _, mf in object_manifests)
        top_manifest = {
            "scene":             os.path.basename(args.input),
            "n_total_gaussians": len(model),
            "rank_method":       args.layer_rank,
            "layer_percentages": layer_pcts,
            "total_bytes_scene": total_bytes_scene,
            "total_kb_scene":    round(total_bytes_scene / 1024.0, 2),
            "total_mb_scene":    round(total_bytes_scene / (1024.0**2), 3),
            # Streaming-paper friendly: per-level aggregated across objects.
            # A streaming client that fetches level=k for ALL objects sees
            # these aggregate sizes and download-time estimates.
            "per_level_aggregate": per_level_aggregate,
            "note": ("File sizes are independent of --layer_rank because "
                     "each gaussian is a fixed-size row and percentages "
                     "determine row count per layer. The rank method changes "
                     "WHICH gaussians populate each layer, affecting visual "
                     "quality progression but not byte count. For variable "
                     "layer sizes, use non-uniform --layer_percentages such "
                     "as '10,30,60,100'."),
            "objects":           [],
        }
        for oi, label, mf in object_manifests:
            entry = dict(mf)
            entry["object_id"] = oi
            entry["object_class"] = label
            # rewrite file paths to be relative to outdir for portability.
            # For oid=-1 ('other'), the on-disk directory is named
            # f"other_n{n_total}" -- NOT just mf["object"] which is the
            # label string "other". Earlier versions used mf["object"]
            # which produced bogus paths like 'objects/other' that don't
            # exist on disk; downstream consumers (e.g. VMAF renderer)
            # failed to find the full ply.
            if oi == -1:
                subdir_name = f"other_n{mf['n_total']}"
            else:
                subdir_name = f"obj_{oi:03d}_{label}_n{mf['n_total']}"
            subdir_rel = os.path.relpath(
                os.path.join(obj_dir, subdir_name),
                args.outdir)
            entry["dir_relative_to_outdir"] = subdir_rel.replace(os.sep, "/")
            top_manifest["objects"].append(entry)

        # Embed cluster info if --clustering was enabled. The server can use
        # this either as a primary catalog (cluster-first streaming) or as
        # supplementary metadata (per-object streaming with occasional cluster
        # fallback for tightly-coupled object groups).
        if cluster_manifests:
            top_manifest["clustering"] = {
                "enabled":   True,
                "method":    args.cluster_method,
                "eps":       (args.cluster_eps_abs
                              if args.cluster_eps_abs is not None
                              else args.cluster_eps_frac * _scene_diagonal(model)),
                "n_clusters": len(cluster_manifests),
                "clusters":  [],
            }
            for ci, lbl, mf in cluster_manifests:
                top_manifest["clustering"]["clusters"].append({
                    "cluster_id":     ci,
                    "label":          lbl,
                    "object_ids":     mf.get("object_ids", []),
                    "object_classes": mf.get("object_classes", []),
                    "n_gaussians":    mf["n_total"],
                    "total_bytes":    mf.get("total_bytes", 0),
                    "dir_relative":   mf.get("dir_relative", ""),
                    "layers":         mf.get("layers", []),
                })
        else:
            top_manifest["clustering"] = {"enabled": False}

        top_manifest_path = os.path.join(args.outdir, "manifest.json")
        with open(top_manifest_path, "w") as f:
            json.dump(top_manifest, f, indent=2)
        print(f"[layers] top-level manifest -> {top_manifest_path}  "
              f"({total_bytes_scene/1024**2:.2f} MB total)")

    # summary
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"input: {args.input}\n")
        f.write(f"camera_json: {args.camera_json}\n")
        f.write(f"detector: {args.detector}    model: {args.yolo_model}\n")
        if text_prompts:
            f.write(f"text_prompts: {text_prompts}\n")
        f.write(f"gaussians: {len(model):,}\n")
        f.write(f"views: {len(mvs)}\n")
        for vi, dets in enumerate(detections_per_view):
            f.write(f"  view {vi:02d} ({mvs[vi].name}): {len(dets)} detections "
                    f"[{', '.join(d.class_name for d in dets)}]\n")
        f.write(f"\nobjects (after cross-view merge): {len(objects)}\n")
        for oi, obj in enumerate(objects):
            n = int(masks.get(oi, np.zeros(0)).sum())
            stats = dilate_stats.get(oi, {})
            f.write(f"  obj {oi:03d} class={obj.class_name:<14} "
                    f"merged_from={len(obj.detections):2d}_detections "
                    f"centroid={np.round(obj.centroid, 3).tolist()} "
                    f"seed={stats.get('seed', n):,} "
                    f"+dilated={stats.get('added', 0):,} "
                    f"final={n:,} "
                    f"r={stats.get('radius', 0.0):.3f} "
                    f"extent={stats.get('extent', 0.0):.3f}\n")
        f.write(f"\nother (unassigned): {n_other:,} gaussians "
                f"({100*n_other/len(model):.1f}% of scene)\n")
        f.write(f"mask_pad: {args.mask_pad}\n")
        f.write(f"depth_band_frac: {args.depth_band_frac}\n")
        f.write(f"dilate_radius_frac: {args.dilate_radius_frac}\n")
        if args.dilate_radius_abs is not None:
            f.write(f"dilate_radius_abs: {args.dilate_radius_abs}\n")
        if args.dilate_radius_max is not None:
            f.write(f"dilate_radius_max: {args.dilate_radius_max}\n")
        if not args.no_layers:
            f.write(f"\nprogressive layers (rank='{args.layer_rank}', "
                    f"pct={layer_pcts}):\n")
            for oi, label, mf in object_manifests:
                f.write(f"  {label:<14} (n={mf['n_total']:,}):\n")
                for L in mf["layers"]:
                    f.write(f"    layer {L['layer']} ({L['name']:<5}): "
                            f"+{L['delta_gaussians']:,} gaussians  "
                            f"(cum={L['cumulative_gaussians']:,} = "
                            f"{L['cumulative_pct']}%)  "
                            f"{L['size_bytes']/1024:.1f} KB  -> {L['file']}\n")
        if args.clustering and cluster_manifests:
            f.write(f"\nclusters (method='{args.cluster_method}'):\n")
            for ci, lbl, mf in cluster_manifests:
                f.write(f"  cluster {ci:03d} [{lbl}] "
                        f"n_objects={len(mf.get('object_ids', []))} "
                        f"gaussians={mf['n_total']:,} "
                        f"total={mf.get('total_kb', 0):.1f} KB "
                        f"members={mf.get('object_classes', [])}\n")
    print("[done]")


if __name__ == "__main__":
    main()