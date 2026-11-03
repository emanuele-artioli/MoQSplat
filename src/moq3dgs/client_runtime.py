"""Disk-backed, budgeted residency for continuously rendered MoQSplat layers.

GPU admission uses a conservative estimate and a per-process Torch allocator
limit. Driver allocations are excluded; scene-dependent rasterizer scratch is
measured. OOM retries discard enhancements, never base layers.
"""

from __future__ import annotations

import heapq
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from moq3dgs.render.rasterizer import render_gaussians
from moq3dgs.transport.moq_backend import viewport_priority
from moq3dgs.viewport.frustum import projection_matrix_from_fov

_KEYS = ("means", "opacities", "scales", "rotations", "sh_dc", "sh_rest")


@dataclass
class _Entry:
    layer: object
    path: Path
    size: int
    centroid: np.ndarray


class LayerStore:
    """Retain verified layer files, cache bounded CPU arrays and GPU tensors.

    All completed base layers (subgroup zero) are mandatory. Enhancements are
    admitted in contiguous order and ranked by the latest camera pose. Budget
    reserves twice the admitted attribute bytes plus 64 MiB for concatenation
    and rasterization; real CUDA peak is measured independently.
    """

    def __init__(self, max_gpu_memory_mib, cpu_cache_mib, device="cuda:0"):
        if max_gpu_memory_mib <= 0 or cpu_cache_mib < 0 or not str(device).startswith("cuda"):
            raise ValueError("positive GPU budget, nonnegative CPU budget and CUDA device required")
        self.device = str(device)
        self.gpu_budget = int(max_gpu_memory_mib * 1024 * 1024)
        self.cpu_budget = int(cpu_cache_mib * 1024 * 1024)
        self.high_watermark = int(self.gpu_budget * 0.9)
        self.low_watermark = int(self.gpu_budget * 0.8)
        self.workspace_reserve = 64 * 1024 * 1024
        self.entries = {}
        self.cpu = OrderedDict()
        self.cpu_bytes = 0
        self.gpu = {}
        self.gpu_bytes = 0
        self.promotions = 0
        self.evictions = 0

    def set_gpu_budget(self, max_gpu_memory_mib):
        """Set the admission and measured-peak budget for the next frame."""
        if max_gpu_memory_mib <= 0:
            raise ValueError("GPU memory budget must be positive")
        self.gpu_budget = int(max_gpu_memory_mib * 1024 * 1024)
        self.high_watermark = int(self.gpu_budget * 0.9)
        self.low_watermark = int(self.gpu_budget * 0.8)

    def add_completed(self, layer, path, centroid=None):
        """Register a completed, already transport-verified NPZ on disk."""
        path = Path(path).resolve(strict=True)
        if layer.name in self.entries:
            raise ValueError(f"duplicate completed layer: {layer.name}")
        with zipfile.ZipFile(path) as archive:
            files = {entry.filename: entry for entry in archive.infolist()}
            if set(files) != {f"{name}.npy" for name in _KEYS}:
                raise ValueError(f"invalid layer NPZ attributes: {path}")
            size = sum(files[f"{name}.npy"].file_size for name in _KEYS)
        if centroid is None:
            with np.load(path, allow_pickle=False) as data:
                centroid = np.asarray(data["means"].mean(axis=0), dtype=float)
        centroid = np.asarray(centroid, dtype=float)
        if centroid.shape != (3,) or not np.isfinite(centroid).all():
            raise ValueError("layer centroid must contain three finite coordinates")
        self.entries[layer.name] = _Entry(layer, path, size, centroid)

    def _arrays(self, entry):
        name = entry.layer.name
        if name in self.cpu:
            self.cpu.move_to_end(name)
            return self.cpu[name]
        with np.load(entry.path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in _KEYS}
        if any(
            array.dtype != np.float32 or not np.isfinite(array).all() for array in arrays.values()
        ):
            raise ValueError(f"invalid float32 layer arrays: {name}")
        n = arrays["means"].shape[0] if arrays["means"].ndim == 2 else -1
        expected = {
            "means": (n, 3),
            "opacities": (n, 1),
            "scales": (n, 3),
            "rotations": (n, 4),
            "sh_dc": (n, 1, 3),
        }
        rest = arrays["sh_rest"]
        if (
            n <= 0
            or any(arrays[k].shape != shape for k, shape in expected.items())
            or rest.ndim != 3
            or rest.shape[0] != n
            or rest.shape[1] not in (0, 3, 8, 15)
            or rest.shape[2] != 3
        ):
            raise ValueError(f"invalid layer attribute shapes: {name}")
        size = sum(a.nbytes for a in arrays.values())
        if size <= self.cpu_budget:
            while self.cpu_bytes + size > self.cpu_budget:
                _, old = self.cpu.popitem(last=False)
                self.cpu_bytes -= sum(a.nbytes for a in old.values())
            self.cpu[name] = arrays
            self.cpu_bytes += size
        return arrays

    def _drop(self, name):
        _, size = self.gpu.pop(name)
        self.gpu_bytes -= size
        self.evictions += 1

    def _rank(self, entry, pose):
        return (
            viewport_priority(
                entry.centroid, pose["camera_position"], pose["view_matrix"], entry.layer.subgroup
            ),
            entry.layer.name,
        )

    def _select(self, pose):
        bases = [e for e in self.entries.values() if e.layer.subgroup == 0]
        if not bases:
            return []
        selected = sorted(bases, key=lambda e: self._rank(e, pose))
        total = sum(e.size for e in selected)
        if 3 * total + self.workspace_reserve > self.gpu_budget:
            raise RuntimeError("mandatory base layers cannot fit declared GPU memory budget")
        potential = 3 * sum(e.size for e in self.entries.values()) + self.workspace_reserve
        target = max(
            3 * total + self.workspace_reserve,
            self.low_watermark if potential > self.high_watermark else self.high_watermark,
        )
        groups = {(e.layer.track, e.layer.group): {0} for e in bases}
        levels = {
            (e.layer.track, e.layer.group, e.layer.subgroup): e for e in self.entries.values()
        }
        candidates = [
            (self._rank(e, pose), e)
            for e in self.entries.values()
            if e.layer.subgroup == 1 and (e.layer.track, e.layer.group) in groups
        ]
        heapq.heapify(candidates)
        while candidates:
            _, entry = heapq.heappop(candidates)
            if 3 * (total + entry.size) + self.workspace_reserve <= target:
                selected.append(entry)
                groups[(entry.layer.track, entry.layer.group)].add(entry.layer.subgroup)
                total += entry.size
                following = levels.get(
                    (entry.layer.track, entry.layer.group, entry.layer.subgroup + 1)
                )
                if following is not None:
                    heapq.heappush(candidates, (self._rank(following, pose), following))
        return selected

    def _promote(self, entry):
        arrays = self._arrays(entry)
        tensors = {name: torch.from_numpy(value).to(self.device) for name, value in arrays.items()}
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        self.gpu[entry.layer.name] = (tensors, size)
        self.gpu_bytes += size
        self.promotions += 1

    def _limit_allocator(self):
        """This client owns its process; cap Torch allocations, excluding driver context."""
        total = torch.cuda.get_device_properties(self.device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, self.gpu_budget / total), self.device)

    def _draw(self, selected, pose, width, height):
        parts = {
            key: torch.cat([self.gpu[e.layer.name][0][key] for e in selected], dim=0)
            for key in _KEYS
            if key != "sh_rest"
        }
        rests = [self.gpu[e.layer.name][0]["sh_rest"] for e in selected]
        sh_width = max(t.shape[1] for t in rests)
        parts["sh_rest"] = torch.cat(
            [torch.nn.functional.pad(t, (0, 0, 0, sh_width - t.shape[1])) for t in rests], dim=0
        )
        sh = torch.cat((parts["sh_dc"], parts["sh_rest"]), dim=1).reshape(len(parts["means"]), -1)
        view = torch.tensor(pose["view_matrix"], dtype=torch.float32, device=self.device)
        proj = torch.tensor(
            projection_matrix_from_fov(
                pose["fov"], width / height, fov_angles=pose.get("fov_angles")
            ),
            dtype=torch.float32,
            device=self.device,
        )
        position = torch.tensor(pose["camera_position"], dtype=torch.float32, device=self.device)
        return render_gaussians(
            parts["means"],
            parts["opacities"],
            sh,
            parts["scales"],
            parts["rotations"],
            view,
            proj,
            position,
            image_width=width,
            image_height=height,
            fov_y_deg=pose["fov"],
            fov_angles=pose.get("fov_angles"),
            device=self.device,
        )

    def render(self, pose, width, height):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for client rendering")
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        selected = self._select(pose)
        if not selected:
            raise RuntimeError("no complete base layers available")
        observed_peak = 0
        retries = 0
        while True:
            keep = {entry.layer.name for entry in selected}
            evicted = False
            for name in list(self.gpu):
                if name not in keep:
                    self._drop(name)
                    evicted = True
            if evicted:
                torch.cuda.empty_cache()
            self._limit_allocator()
            torch.cuda.reset_peak_memory_stats(self.device)
            try:
                for entry in selected:
                    if entry.layer.name not in self.gpu:
                        self._promote(entry)
                image = self._draw(selected, pose, width, height)
                if image.dtype != torch.uint8 or tuple(image.shape) != (height, width, 3):
                    raise RuntimeError("renderer returned invalid image")
                peak = torch.cuda.max_memory_allocated(self.device)
                observed_peak = max(observed_peak, peak)
                if peak > self.gpu_budget:
                    del image
                    enhancements = [entry for entry in selected if entry.layer.subgroup > 0]
                    if not enhancements:
                        raise RuntimeError(
                            "measured CUDA peak exceeds GPU budget for mandatory base layers"
                        )
                    selected.remove(enhancements[-1])
                    retries += 1
                    continue
                break
            except RuntimeError as exc:
                if not isinstance(exc, torch.cuda.OutOfMemoryError) and not isinstance(
                    exc.__cause__, torch.cuda.OutOfMemoryError
                ):
                    raise
                enhancements = [entry for entry in selected if entry.layer.subgroup > 0]
                if not enhancements:
                    raise RuntimeError("CUDA OOM while rendering mandatory base layers") from exc
                selected.remove(enhancements[-1])
                observed_peak = max(observed_peak, torch.cuda.max_memory_allocated(self.device))
                retries += 1
                torch.cuda.empty_cache()
        metrics = {
            "resident_layers": [e.layer.name for e in selected],
            "selected_layer_ids": [e.layer.name for e in selected],
            "splat_count": sum(self.gpu[e.layer.name][0]["means"].shape[0] for e in selected),
            "cpu_cache_bytes": self.cpu_bytes,
            "gpu_resident_bytes": self.gpu_bytes,
            "gpu_budget_bytes": self.gpu_budget,
            "gpu_high_watermark_bytes": self.high_watermark,
            "gpu_low_watermark_bytes": self.low_watermark,
            "estimated_peak_bytes": 3 * sum(e.size for e in selected) + self.workspace_reserve,
            "cuda_peak_allocated_bytes": peak,
            "cuda_attempt_peak_allocated_bytes": observed_peak,
            "budget_adjustment_retries": retries,
            "cuda_reserved_bytes": torch.cuda.memory_reserved(self.device),
            "promotions": self.promotions,
            "evictions": self.evictions,
        }
        return image.cpu().numpy(), metrics
