"""Asynchronous MoQ client over QUIC.

Connects to the server, receives the manifest, replays a camera trace,
sends viewport updates, and receives Gaussian cluster frames which are
deposited into the local SplatCache.  A separate render task drains the
cache each frame and writes images + metrics to disk.

The client is the primary deliverable of this project.  It implements:

1. **Connection & manifest negotiation** — TCP (QUIC placeholder).
2. **Viewport-driven subscription** — compares the local cache against
   the manifest to subscribe only to missing / stale chunks.
3. **Priority-aware reception** — decodes binary cluster frames and
   inserts them into the persistent SplatCache.
4. **Render loop** — assembles cached splats each frame and writes
   the rendered image + per-frame metrics to disk.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import structlog
import torch

from moq3dgs.decorators import gpu_bound, network_bound
from moq3dgs.models import (
    ImportanceTier,
    MoQSubscription,
    SceneManifest,
    TrackInfo,
    ViewportUpdate,
)
from moq3dgs.render.cache import SplatCache
from moq3dgs.render.rasterizer import render_gaussians
from moq3dgs.render.writer import FrameWriter
from moq3dgs.transport.protocol import decode_cluster
from moq3dgs.viewport.frustum import (
    check_aabb_frustum,
    extract_frustum,
    projection_matrix_from_fov,
)
from moq3dgs.viewport.priority import compute_priority
from moq3dgs.viewport.trace import (
    camera_forward_from_view_matrix,
    load_trace,
    replay_trace,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Client core
# ---------------------------------------------------------------------------


class MoQClient:
    """Production MoQ streaming client with rendering pipeline.

    Lifecycle:
        1. ``connect()``        — TCP handshake, receive manifest.
        2. ``start_receiving()`` — background task reading frames.
        3. ``run_trace()``       — replay a camera trace, subscribe,
                                   receive, render, and save each frame.
        4. ``disconnect()``      — tear down.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4433,
        client_id: str = "client-0",
        output_dir: str | Path = "./output",
        device: str = "cuda:1",
        image_width: int = 1920,
        image_height: int = 1080,
        render_enabled: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.output_dir = Path(output_dir)
        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.render_enabled = render_enabled

        # Network state
        self.manifest: Optional[SceneManifest] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None

        # Data state
        self.cache = SplatCache()
        self._frame_writer = FrameWriter(
            self.output_dir / "frames", 
            width=self.image_width, 
            height=self.image_height
        )
        self._pending_subscriptions: Dict[Tuple[str, str, int, int], float] = {}
        self._subscription_retry_seconds = 30.0
        
        # Adaptive Budget Management
        self._gaussian_budget = 5_000_000  # Start with 5M splat budget
        self._last_render_ms = 33.3        # Target ~30 FPS baseline
        self._budget_increase_rate = 1.1   # +10% if fast
        self._budget_decrease_rate = 0.9   # -10% if slow
        self._perf_target_ms = 40.0        # 24 FPS target threshold
        self._perf_low_threshold_ms = 30.0 # Threshold to increase budget
        # Metrics
        self._bytes_received = 0
        self._clusters_received = 0
        self._frames_rendered = 0

    # -- connection ----------------------------------------------------------

    @network_bound
    async def connect(self) -> SceneManifest:
        """Establish connection and receive the scene manifest.

        Returns:
            The :class:`SceneManifest` describing all available tracks/groups.
        """
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port,
        )
        logger.info(
            "connected",
            host=self.host,
            port=self.port,
            client_id=self.client_id,
        )

        msg = await self._read_json()
        if msg is None or msg.get("type") != "manifest":
            raise RuntimeError("Expected manifest message from server")

        self.manifest = SceneManifest(**msg["data"])
        logger.info(
            "manifest_received",
            tracks=len(self.manifest.tracks),
            total_gaussians=self.manifest.total_gaussians,
        )

        # Save manifest to disk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(self.manifest.model_dump_json(indent=2))
        logger.info("manifest_saved", path=str(manifest_path))

        return self.manifest

    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._writer:
            self._writer.close()
            
        if self._frame_writer:
            self._frame_writer.close()
            
        logger.info(
            "disconnected",
            bytes_received=self._bytes_received,
            clusters_received=self._clusters_received,
            frames_rendered=self._frames_rendered,
        )

    # -- background receiver -------------------------------------------------

    async def start_receiving(self) -> None:
        """Start the background receive loop."""
        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Read length-prefixed frames from the server continuously.

        JSON messages (control) are handled inline; binary frames (cluster
        data) are decoded and inserted into the SplatCache.
        """
        assert self._reader is not None
        try:
            while self._running:
                hdr = await self._reader.read(4)
                if len(hdr) < 4:
                    break
                length = int.from_bytes(hdr, "little")
                data = await self._reader.readexactly(length)
                self._bytes_received += len(data)

                # Try JSON first (control messages)
                try:
                    msg = json.loads(data)
                    self._handle_control(msg)
                    continue
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

                # Binary cluster frame
                try:
                    cluster = decode_cluster(data)
                    # New protocol uses object_id as quality_level
                    quality = cluster["object_id"]
                    
                    # Performance: move to GPU immediately
                    for k in ["means", "opacities", "sh_coeffs", "scales", "rotations"]:
                        if cluster.get(k) is not None:
                            cluster[k] = cluster[k].to(self.device)
                    
                    self.cache.put(cluster)
                    self._pending_subscriptions.pop((cluster["track_id"], cluster["group_id"],
                                                     cluster["subgroup_id"], cluster["object_id"]), None)
                    self._clusters_received += 1
                    self._bytes_received += len(data)
                    logger.debug(
                        "cluster_received",
                        track=cluster["track_id"],
                        group=cluster["group_id"],
                        subgroup_id=cluster["subgroup_id"],
                        quality=cluster["object_id"],
                        n=cluster["num_gaussians"],
                    )
                except Exception as e:
                    logger.error("decode_failed", error=str(e))

        except (asyncio.CancelledError, ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            self._running = False
            logger.debug("receive_loop_exited")

    def _handle_control(self, msg: dict) -> None:
        """Process incoming control messages (subscribe_ack, etc.)."""
        msg_type = msg.get("type", "")
        if msg_type == "subscribe_ack":
            data = msg.get("data", {})
            if data.get("accepted"):
                logger.debug(
                    "subscribe_ack",
                    track=data.get("track_id"),
                    group=data.get("group_id"),
                )
        else:
            logger.debug("control_message", type=msg_type)

    # -- subscription management ---------------------------------------------

    async def send_viewport_update(self, update: ViewportUpdate) -> None:
        """Send a viewport update and subscribe to newly visible clusters.

        Subscription strategy (viewport-aware ABR):
        ─────────────────────────────────────────────
        1. Compute the view frustum from the camera pose.
        2. For every visible track/group in the manifest, compute a
           *priority score* (distance + screen position).
        3. Sort visible groups by priority (closest/central = first).
        4. Issue subscriptions **breadth-first by Importance Tier**:
           - First pass: subscribe all unsubscribed visible groups for
             BASE_LARGE (tier 0) only.  This gives the entire visible
             scene a coarse skeleton quickly.
           - Second pass (next frame): request BASE_MEDIUM (tier 0+1) for
             groups already seen.
           - And so on up to ENHANCE_MEDIUM (tier 4, full quality).

        This prevents the "one cluster looks great, rest is black" problem
        that occurs when all tiers of one group are fetched before others.
        """
        assert self._writer is not None
        assert self.manifest is not None

        await self._send_json({
            "type": "viewport_update",
            "data": update.model_dump(),
        })

        # Compute frustum for subscription decisions
        view_matrix = np.array(update.view_matrix, dtype=np.float64)
        proj = projection_matrix_from_fov(
            update.fov, aspect=self.image_width / self.image_height,
            fov_angles=update.fov_angles,
        )
        vp = proj @ view_matrix
        frustum = extract_frustum(vp)
        camera_pos = np.array(update.camera_position, dtype=np.float64)
        camera_fwd = camera_forward_from_view_matrix(update.view_matrix)

        # ── Step 1: collect and prioritise visible groups ────────────────────
        candidates: list[tuple[int, str, str]] = []  # (priority, track_id, group_id)
        for track in self.manifest.tracks:
            track_vis = check_aabb_frustum(
                np.array(track.bbox_min),
                np.array(track.bbox_max),
                frustum,
            )
            if track_vis == 0:
                continue

            for group in track.groups:
                group_vis = check_aabb_frustum(
                    np.array(group.bbox_min),
                    np.array(group.bbox_max),
                    frustum,
                )
                if group_vis == 0:
                    continue

                priority = compute_priority(
                    camera_pos,
                    camera_fwd,
                    np.array(group.bbox_min),
                    np.array(group.bbox_max),
                    frustum,
                    ImportanceTier.LARGE,
                )
                candidates.append((priority, track.track_id, group.group_id))

        # Sort ascending = highest priority (0 = best) first
        candidates.sort(key=lambda x: x[0])

        # Request at most one missing tier per group and update. Pending tiers
        # do not block later ones: a server may omit an empty subgroup.
        for priority, track_id, group_id in candidates:
            now = time.monotonic()
            missing = []
            for quality in (0, 1):
                for subgroup in (0, 1, 2):
                    key = (track_id, group_id, subgroup, quality)
                    if self.cache.has(*key):
                        self._pending_subscriptions.pop(key, None)
                    else:
                        missing.append(key)
            # Prefer never-requested tiers over retries, so an empty tier
            # cannot starve the rest of a group.
            key = next((k for k in missing if k not in self._pending_subscriptions), None)
            if key is None:
                key = next((k for k in missing if now - self._pending_subscriptions[k]
                            >= self._subscription_retry_seconds), None)
            if key is not None:
                await self._subscribe(*key, priority)
                self._pending_subscriptions[key] = time.monotonic()

    async def _subscribe(
        self,
        track_id: str,
        group_id: str,
        subgroup_id: int = 0,
        quality_level: int = 0,
        priority: int = 128,
    ) -> None:
        """Send a MoQ subscription request."""
        sub = MoQSubscription(
            track_id=track_id,
            group_id=group_id,
            subgroup_id=subgroup_id,
            quality_level=quality_level,
            priority=priority,
        )
        await self._send_json({
            "type": "subscribe",
            "data": sub.model_dump(),
        })
        logger.debug(
            "subscribed",
            track=track_id,
            group=group_id,
            subgroup_id=subgroup_id,
            quality=quality_level,
            priority=priority,
        )


    # -- render loop ---------------------------------------------------------

    def get_visible_keys(self, view_matrix: np.ndarray, fov: float, margin: float = 0.2,
                         fov_angles: tuple[float, float, float, float] | None = None,
                         ) -> Set[Tuple[str, str]]:
        """Identify which track/group IDs are visible in the current viewport.
        
        Uses a 'halo' frustum (expanded by margin) to prevent objects from 
        popping in/out at the edges of the screen.
        """
        # Expand FOV by margin to include a halo of off-screen clusters
        relaxed_fov = min(179.0, fov * (1.0 + margin))
        relaxed_angles = None
        if fov_angles is not None:
            from moq3dgs.viewport.frustum import fov_tangents
            left, right, down, up = fov_tangents(fov_angles)
            center_x, center_y = (left + right) / 2, (down + up) / 2
            half_x, half_y = (right - left) * (1 + margin) / 2, (up - down) * (1 + margin) / 2
            angles = np.arctan(
                [center_x - half_x, center_x + half_x, center_y - half_y, center_y + half_y])
            relaxed_angles = (float(angles[0]), float(angles[1]), float(angles[2]), float(angles[3]))
        proj = projection_matrix_from_fov(
            relaxed_fov, aspect=self.image_width / self.image_height, fov_angles=relaxed_angles,
        )
        vp = proj @ view_matrix
        frust = extract_frustum(vp)
        
        visible = set()
        for track in self.manifest.tracks:
            # First check the whole track
            track_vis = check_aabb_frustum(
                np.array(track.bbox_min), np.array(track.bbox_max), frust
            )
            if track_vis == 0:  # OUTSIDE
                continue
            
            # Then check groups within the track
            for group in track.groups:
                group_vis = check_aabb_frustum(
                    np.array(group.bbox_min), np.array(group.bbox_max), frust
                )
                if group_vis > 0:  # INSIDE or INTERSECTING
                    visible.add((track.track_id, group.group_id))
        
        return visible

    @gpu_bound
    def render_frame(
        self,
        frame_idx: int,
        view_matrix: List[List[float]],
        fov: float,
        camera_position: List[float],
        fov_angles: tuple[float, float, float, float] | None = None,
    ) -> Optional[Path]:
        """Render the current cache contents and save to disk.

        Assembles all base-layer cached clusters, runs the rasteriser,
        and writes the image + metrics.

        Args:
            frame_idx: Frame number for file naming.
            view_matrix: 4×4 row-major view matrix.
            fov: Vertical FOV in degrees.
            camera_position: Camera world position.

        Returns:
            Path to the saved image, or None if cache is empty or
            rendering is disabled.
        """
        if not self.render_enabled:
            return None

        # 1. Compute view state for prioritisation
        vm_np = np.array(view_matrix)
        proj_np = projection_matrix_from_fov(
            fov, aspect=self.image_width / self.image_height, fov_angles=fov_angles,
        )
        vp_np = proj_np @ vm_np
        frust = extract_frustum(vp_np)
        cam_fwd = camera_forward_from_view_matrix(view_matrix)
        cam_pos_np = np.array(camera_position)

        # 2. Get visible clusters with a halo margin (prevents edge popping)
        visible_keys = self.get_visible_keys(vm_np, fov, margin=0.2, fov_angles=fov_angles)
        
        # 3. Prioritise visible clusters for budget management
        # We want to fill the budget with the highest priority clusters first.
        scored_candidates = []
        # Access tracks via manifest to get BBoxes
        track_lookup = {t.track_id: t for t in self.manifest.tracks}
        
        for tid, gid in visible_keys:
            # We only care about cached clusters
            track = track_lookup.get(tid)
            if not track: continue
            group = next((g for g in track.groups if g.group_id == gid), None)
            if not group: continue

            # Priority 0 = highest, 255 = lowest
            p = compute_priority(
                cam_pos_np, cam_fwd,
                np.array(group.bbox_min), np.array(group.bbox_max),
                frust, ImportanceTier.LARGE
            )
            
            # Add subsets 0, 1, 2 to the candidate pool
            for sub in range(3):
                # Pick BEST available quality for this subset
                best_q = -1
                if self.cache.has(tid, gid, sub, 1):
                    best_q = 1
                elif self.cache.has(tid, gid, sub, 0):
                    best_q = 0
                
                if best_q >= 0:
                    cluster = self.cache.get(tid, gid, sub, best_q)
                    # Priority for budget: Base geometry (sub 0) > others.
                    # Quality 1 gets a slight priority boost to prefer detail if budget allows.
                    scored_candidates.append({
                        "key": (tid, gid, sub, best_q),
                        "priority": p + (sub * 50) - (best_q * 10),
                        "n": cluster["num_gaussians"]
                    })

        # Sort by priority (ascending: 0 is best)
        scored_candidates.sort(key=lambda x: x["priority"])

        # 4. Fill budget
        selected_keys = set()
        current_n = 0
        for cand in scored_candidates:
            if current_n + cand["n"] > self._gaussian_budget:
                break
            selected_keys.add((cand["key"][0], cand["key"][1], cand["key"][2], cand["key"][3]))
            current_n += cand["n"]

        # 5. Assemble and render
        # We pass a flat list of keys to assemble_base (customised call)
        assembled = self.cache.assemble_specific(device=self.device, keys=selected_keys)
        
        if assembled is None or assembled["num_gaussians"] == 0:
            logger.debug("render_skip_empty_budget", frame=frame_idx, budget=self._gaussian_budget)
            return None

        t0 = time.perf_counter()

        vm = torch.tensor(vm_np, dtype=torch.float32)
        proj = torch.tensor(proj_np.astype(np.float32), dtype=torch.float32)
        cam_pos = torch.tensor(cam_pos_np, dtype=torch.float32)

        image = render_gaussians(
            means=assembled["means"],
            opacities=assembled["opacities"],
            sh_coeffs=assembled["sh_coeffs"],
            scales=assembled["scales"],
            rotations=assembled["rotations"],
            view_matrix=vm,
            proj_matrix=proj,
            camera_pos=cam_pos,
            image_width=self.image_width,
            image_height=self.image_height,
            fov_y_deg=fov,
            fov_angles=fov_angles,
            device=self.device,
        )

        render_time = time.perf_counter() - t0
        render_ms = render_time * 1000
        self._last_render_ms = render_ms
        self._frames_rendered += 1

        metrics = {
            "num_gaussians": assembled["num_gaussians"],
            "cache_entries": self.cache.num_entries,
            "bytes_received": self._bytes_received,
            "clusters_received": self._clusters_received,
            "render_time_ms": render_time * 1000,
        }

        path = self._frame_writer.save_frame(image, frame_idx, metrics)
        logger.info(
            "frame_rendered",
            idx=frame_idx,
            gaussians=assembled["num_gaussians"],
            render_ms=f"{render_time*1000:.1f}",
            path=str(path),
        )
        return path

    # -- full trace run ------------------------------------------------------

    async def run_trace(
        self,
        trace_path: str | Path,
        inter_frame_delay: float = 0.05,
    ) -> dict:
        """Replay a camera trace end-to-end.

        For each frame:
        1. Send viewport update to server.
        2. Wait briefly for data to arrive.
        3. Render the current cache state.
        4. Save image + metrics.

        Args:
            trace_path: Path to the JSON trace file.
            inter_frame_delay: Seconds to wait between frames for data
                arrival. Set to 0 for maximum speed.

        Returns:
            Summary dict with aggregate metrics.
        """
        frames = load_trace(trace_path)
        total = len(frames)
        logger.info("trace_started", total_frames=total, trace=str(trace_path))

        t_start = time.perf_counter()
        rendered_paths: List[str] = []

        for i, update in enumerate(replay_trace(frames, self.client_id)):
            # 1. Send viewport update
            await self.send_viewport_update(update)

            # 2. Let data arrive
            if inter_frame_delay > 0:
                await asyncio.sleep(inter_frame_delay)

            # 3. Render
            path = self.render_frame(
                frame_idx=i,
                view_matrix=update.view_matrix,
                fov=update.fov,
                camera_position=update.camera_position,
                fov_angles=update.fov_angles,
            )
            if path:
                rendered_paths.append(str(path))

            if (i + 1) % 10 == 0 or i == total - 1:
                logger.info(
                    "trace_progress",
                    frame=f"{i+1}/{total}",
                    cache=self.cache.num_entries,
                    rendered=self._frames_rendered,
                )

        elapsed = time.perf_counter() - t_start

        summary = {
            "total_frames": total,
            "frames_rendered": self._frames_rendered,
            "bytes_received": self._bytes_received,
            "clusters_received": self._clusters_received,
            "cache_entries_final": self.cache.num_entries,
            "elapsed_seconds": elapsed,
            "avg_fps": total / max(elapsed, 1e-6),
            "rendered_paths": rendered_paths,
        }

        # Save summary
        summary_path = self.output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(
            "trace_complete",
            frames=total,
            rendered=self._frames_rendered,
            elapsed_s=f"{elapsed:.1f}",
            summary=str(summary_path),
        )

        return summary

    # -- I/O helpers ---------------------------------------------------------

    async def _send_json(self, obj: dict) -> None:
        """Send a length-prefixed JSON message."""
        assert self._writer is not None
        payload = json.dumps(obj).encode("utf-8")
        self._writer.write(len(payload).to_bytes(4, "little") + payload)
        await self._writer.drain()

    async def _read_json(self) -> Optional[dict]:
        """Read a length-prefixed JSON message; returns None on EOF."""
        assert self._reader is not None
        hdr = await self._reader.read(4)
        if len(hdr) < 4:
            return None
        length = int.from_bytes(hdr, "little")
        payload = await self._reader.readexactly(length)
        return json.loads(payload.decode("utf-8"))


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


async def run_client_trace(
    host: str,
    port: int,
    trace_path: str | Path,
    output_dir: str | Path,
    client_id: str = "client-0",
    device: str = "cuda:0",
    image_width: int = 1920,
    image_height: int = 1080,
    render_enabled: bool = True,
    inter_frame_delay: float = 0.05,
) -> dict:
    """Connect, replay a trace, render, and save — one-shot entry point.

    Args:
        host: Server address.
        port: Server port.
        trace_path: Path to the camera movement trace JSON.
        output_dir: Directory for output frames and metrics.
        client_id: Client identifier.
        device: CUDA device for rendering.
        image_width: Render width.
        image_height: Render height.
        render_enabled: Whether to render frames (disable for
            transport-only benchmarks).
        inter_frame_delay: Seconds between frames.

    Returns:
        Summary dict with aggregate metrics.
    """
    client = MoQClient(
        host=host,
        port=port,
        client_id=client_id,
        output_dir=output_dir,
        device=device,
        image_width=image_width,
        image_height=image_height,
        render_enabled=render_enabled,
    )

    try:
        await client.connect()
        await client.start_receiving()
        summary = await client.run_trace(
            trace_path,
            inter_frame_delay=inter_frame_delay,
        )
        return summary
    finally:
        await client.disconnect()
