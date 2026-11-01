"""Minimal GPU rasterisation wrapper.

This module wraps a simple differentiable Gaussian splatting rasteriser.
CUDA requests require ``gsplat`` and propagate native failures. Explicit CPU
requests use a basic depth-sorted DC-color renderer for debugging; it is not
the full-SH renderer used for fidelity experiments.

All rendering output is written to disk (headless requirement).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import structlog

from moq3dgs.decorators import gpu_bound
from moq3dgs.viewport.frustum import camera_intrinsics_from_fov

logger = structlog.get_logger(__name__)


@gpu_bound
def render_gaussians(
    means: torch.Tensor,           # (N, 3)
    opacities: torch.Tensor,       # (N, 1)  sigmoid-space
    sh_coeffs: torch.Tensor,       # (N, C)
    scales: torch.Tensor,          # (N, 3)  log-space
    rotations: torch.Tensor,       # (N, 4)  quaternion
    view_matrix: torch.Tensor,     # (4, 4)  world-to-camera
    proj_matrix: torch.Tensor,     # (4, 4)  GL projection (for CPU fallback NDC)
    camera_pos: torch.Tensor,      # (3,)
    image_width: int = 1920,
    image_height: int = 1080,
    fov_y_deg: float = 60.0,       # vertical FOV – used to build intrinsic K
    bg_color: Optional[torch.Tensor] = None,
    device: str = "cuda:0",
    fov_angles: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """Render Gaussians to an image tensor.

    Returns an (H, W, 3) uint8 tensor on CPU.

    Falls back to a naive depth-sorted splatting if no GPU rasteriser
    is installed.

    Args:
        means: Gaussian centres.
        opacities: Per-Gaussian opacity (sigmoid space).
        sh_coeffs: Spherical harmonics coefficients (flattened).
        scales: Log-space scales.
        rotations: Quaternions (wxyz).
        view_matrix: 4×4 world-to-camera transform.
        proj_matrix: 4×4 projection matrix.
        camera_pos: Camera world-space position.
        image_width: Output width in pixels.
        image_height: Output height in pixels.
        bg_color: (3,) background colour; defaults to black.
        device: CUDA device string.

    Returns:
        (H, W, 3) uint8 image tensor on CPU.
    """
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)

    # Build pixel-space camera intrinsic matrix K from FOV.
    # gsplat requires K (not a GL projection matrix).
    K = torch.from_numpy(camera_intrinsics_from_fov(
        image_width, image_height, fov_y_deg, fov_angles,
    ))
    fx_px, fy_px = float(K[0, 0]), float(K[1, 1])

    # Try hardware-accelerated rasteriser first
    try:
        if torch.device(device).type == "cpu":
            raise ImportError("Explicit CPU debug rendering requested")
        return _render_gsplat(
            means, opacities, sh_coeffs, scales, rotations,
            view_matrix, K,
            image_width, image_height, bg_color, device,
        )
    except Exception as e:
        if torch.device(device).type == "cuda":
            raise RuntimeError("Requested CUDA gsplat rendering failed; no CPU fallback") from e
        logger.debug("gsplat_failed", error=str(e))

    # Fallback: naive CPU renderer (uses GL proj for NDC, K for focal lengths)
    logger.warning("gpu_rasteriser_unavailable_falling_back_to_cpu")
    return _render_cpu_fallback(
        means, opacities, sh_coeffs, scales, rotations,
        view_matrix, proj_matrix, camera_pos,
        image_width, image_height, bg_color,
        fx_px=fx_px, fy_px=fy_px,
        device=device,
    )


def _render_gsplat(
    means: torch.Tensor, opacities: torch.Tensor,
    sh_coeffs: torch.Tensor, scales: torch.Tensor,
    rotations: torch.Tensor, view_matrix: torch.Tensor,
    K: torch.Tensor,          # (3, 3) camera intrinsics
    w: int, h: int, bg: torch.Tensor, device: str,
) -> torch.Tensor:
    """Render via the gsplat library (if available).

    Args:
        K: Pixel-space camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
    """
    import gsplat  # noqa: F811

    # Move everything to device
    m = means.to(device)
    o = torch.sigmoid(opacities.to(device)).squeeze(-1)
    s = torch.exp(scales.to(device))
    r = rotations.to(device)
    # Trace/culling use OpenGL (-Z forward, +Y up). gsplat uses OpenCV
    # (+Z forward, +Y down). Convert the camera axes, never the world scene.
    camera_axes = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))
    vm = camera_axes @ view_matrix.to(device)
    Ks = K.to(device).unsqueeze(0)  # (1, 3, 3)

    # Preserve every transmitted SH coefficient. Passing DC-derived RGB here
    # made all enhancement coefficients irrelevant to the rendered image.
    sh = sh_coeffs.to(device)
    if sh.dim() == 1:
        sh = sh.reshape(len(m), -1)
    
    coefficients = sh.reshape(len(m), -1, 3)
    degree = int(np.sqrt(coefficients.shape[1])) - 1
    if (degree + 1) ** 2 != coefficients.shape[1]:
        raise ValueError("SH coefficient count must be a complete square degree")

    outputs = gsplat.rasterization(
        means=m, quats=r, scales=s, opacities=o,
        colors=coefficients, sh_degree=degree, viewmats=vm.unsqueeze(0),
        Ks=Ks,
        width=w, height=h,
        backgrounds=bg.reshape(1, 3),
    )
    rendered = outputs[0]  # (1, H, W, 3)
    img = rendered[0].clamp(0, 1) * 255
    return img.byte().cpu()


def _render_cpu_fallback(
    means: torch.Tensor, opacities: torch.Tensor,
    sh_coeffs: torch.Tensor, scales: torch.Tensor,
    rotations: torch.Tensor, view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor, camera_pos: torch.Tensor,
    w: int, h: int, bg: torch.Tensor,
    fx_px: float = 276.0,
    fy_px: float = 276.0,
    device: str = "cpu",
) -> torch.Tensor:
    """A PyTorch-based rasterizer to draw proper 3DGS splats.
    Uses GPU if tensors are placed there by the caller.
    """
    with torch.no_grad():
        if means.device.type == "cpu" and "cuda" in device:
            means = means.to(device)
            opacities = opacities.to(device)
            sh_coeffs = sh_coeffs.to(device) if sh_coeffs is not None else None
            scales = scales.to(device)
            rotations = rotations.to(device)
            view_matrix = view_matrix.to(device)
            proj_matrix = proj_matrix.to(device)
            camera_pos = camera_pos.to(device)
            bg = bg.to(device)
            
        device = means.device

    n = len(means)
    
    # 1. Colors from SH0
    if sh_coeffs is not None and sh_coeffs.numel() >= n * 3:
        sh = sh_coeffs.reshape(n, -1)
        colors = sh[:, :3]
        SH_C0 = 0.28209479177387814
        colors = torch.clamp(colors * SH_C0 + 0.5, 0.0, 1.0)
    else:
        colors = torch.full((n, 3), 0.5, device=device)
        
    alpha = torch.sigmoid(opacities).squeeze(-1)
    
    # 2. View Transform
    # view_matrix is usually (4, 4) world-to-cam
    means_hom = torch.cat([means, torch.ones(n, 1, device=device)], dim=1)
    means_cam = (view_matrix @ means_hom.T).T  # (N, 4)
    depths = means_cam[:, 2]
    
    logger.debug("render_debug", 
                 n=n, 
                 depth_min=depths.min().item(), 
                 depth_max=depths.max().item(),
                 depth_mean=depths.mean().item())
    
    # Frustum cull (auto-detect Z direction)
    num_pos = (depths > 0.1).sum().item()
    num_neg = (depths < -0.1).sum().item()
    
    if num_neg > num_pos:
        z_sign = -1.0
        valid = depths < -0.1
    else:
        z_sign = 1.0
        valid = depths > 0.1
        
    if not valid.any():
        logger.warning("render_all_points_culled", num_pos=num_pos, num_neg=num_neg)
        return (bg * 255).byte().cpu().expand(h, w, 3)
        
    # Filter
    means_cam = means_cam[valid]
    colors = colors[valid]
    alpha = alpha[valid]
    depths = depths[valid]
    scales_v = torch.exp(scales[valid])
    rots_v = rotations[valid]
    
    # 3. Projection to 2D
    vp = proj_matrix @ view_matrix
    clip = (vp @ means_hom[valid].T).T
    
    # Perspective division (use abs(w) to handle both conventions)
    ndc = clip[:, :3] / (torch.abs(clip[:, 3:4]) + 1e-6)
    px = ((ndc[:, 0] + 1.0) * 0.5 * w)
    py = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * h)
    
    logger.debug("render_coords",
                 px_min=px.min().item(), px_max=px.max().item(),
                 py_min=py.min().item(), py_max=py.max().item(),
                 color_mean=colors.mean().item(),
                 alpha_mean=alpha.mean().item())
    
    # 4. 2D Covariance
    # Quat to Rot matrix
    qr, qi, qj, qk = rots_v[:, 0], rots_v[:, 1], rots_v[:, 2], rots_v[:, 3]
    R = torch.zeros((len(rots_v), 3, 3), device=device)
    R[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qi*qj - qr*qk)
    R[:, 0, 2] = 2 * (qi*qk + qr*qj)
    R[:, 1, 0] = 2 * (qi*qj + qr*qk)
    R[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2 * (qj*qk - qr*qi)
    R[:, 2, 0] = 2 * (qi*qk - qr*qj)
    R[:, 2, 1] = 2 * (qj*qk + qr*qi)
    R[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)
    
    # Scale matrix
    S = torch.zeros((len(scales_v), 3, 3), device=device)
    S[:, 0, 0] = scales_v[:, 0]
    S[:, 1, 1] = scales_v[:, 1]
    S[:, 2, 2] = scales_v[:, 2]
    
    # 3D Covariance Sigma = R * S * S^T * R^T
    M = R @ S
    Sigma = M @ M.mT
    
    # Jacobian of perspective projection — use pixel-space focal lengths
    fx = fx_px
    fy = fy_px
    
    tx = means_cam[:, 0]
    ty = means_cam[:, 1]
    tz = means_cam[:, 2]
    
    J = torch.zeros((len(tx), 2, 3), device=device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -(fx * tx) / (tz**2)
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -(fy * ty) / (tz**2)
    
    W = view_matrix[:3, :3].unsqueeze(0).expand(len(tx), 3, 3)
    T = J @ W
    
    # 2D Covariance
    Cov2D = T @ Sigma @ T.mT
    Cov2D[:, 0, 0] += 0.3
    Cov2D[:, 1, 1] += 0.3
    
    # Invert 2D Covariance
    det = Cov2D[:, 0, 0] * Cov2D[:, 1, 1] - Cov2D[:, 0, 1] * Cov2D[:, 1, 0]
    det = torch.clamp(det, min=1e-8)
    invCov = torch.zeros_like(Cov2D)
    invCov[:, 0, 0] = Cov2D[:, 1, 1] / det
    invCov[:, 1, 1] = Cov2D[:, 0, 0] / det
    invCov[:, 0, 1] = -Cov2D[:, 0, 1] / det
    invCov[:, 1, 0] = -Cov2D[:, 1, 0] / det
    
    # Radii
    eigen1 = 0.5 * (Cov2D[:, 0, 0] + Cov2D[:, 1, 1] + torch.sqrt(torch.clamp((Cov2D[:, 0, 0] - Cov2D[:, 1, 1])**2 + 4.0 * Cov2D[:, 0, 1]**2, min=0.0)))
    radius = torch.ceil(3.0 * torch.sqrt(eigen1)).int()
    
    # Sort
    order = torch.argsort(depths, descending=True)
    
    # 5. Draw
    canvas = bg.clone().expand(h, w, 3).contiguous()
    
    for idx in order:
        cx, cy = int(px[idx]), int(py[idx])
        r = int(radius[idx])
        if r <= 0 or r > min(w, h): continue
        
        x_min = max(0, cx - r)
        x_max = min(w, cx + r + 1)
        y_min = max(0, cy - r)
        y_max = min(h, cy + r + 1)
        
        if x_min >= x_max or y_min >= y_max: continue
        
        yy, xx = torch.meshgrid(
            torch.arange(y_min, y_max, device=device, dtype=torch.float32),
            torch.arange(x_min, x_max, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        dx = xx - px[idx]
        dy = yy - py[idx]
        
        power = -0.5 * (invCov[idx, 0, 0] * dx**2 + 2.0 * invCov[idx, 0, 1] * dx * dy + invCov[idx, 1, 1] * dy**2)
        valid_mask = power <= 0
        G = torch.exp(power) * alpha[idx]
        G = G * valid_mask
        G = G.unsqueeze(-1)
        
        patch = canvas[y_min:y_max, x_min:x_max]
        canvas[y_min:y_max, x_min:x_max] = patch * (1.0 - G) + colors[idx] * G

    return (canvas * 255.0).clamp(0, 255).byte().cpu()
