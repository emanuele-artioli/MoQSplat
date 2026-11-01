"""Modern MoQ functional reproduction, not the paper's recovered subgroup wire format.

One application layer is one native MoQ track and one finite QUIC group. The
application hierarchy remains explicit in the authenticated catalog and hashes.
Only moq-rs 0.5.2 / moq-ffi 0.4.20 are supported: moving-main APIs differ.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

VERSIONS = {"moq-rs": "0.5.2", "moq-ffi": "0.4.20"}
CATALOG = "moqsplat/catalog"
SCHEMA = "moqsplat-modern-layer-tracks-v1"
PRIORITY_POLICY = {
    "distance_weight": 0.5,
    "angle_weight": 0.5,
    "maximum_distance": 100.0,
    "formula": "utility=.5*max(0,1-min(distance,100)/100)+.5*max(0,cos(theta))",
    "quantization": "bin=min(41,floor(utility*42)); paper_priority=(41-bin)*6+subgroup",
    "tie_break": "quantized subscriber viewport utility first, publisher layer index second",
    "interval_seconds": 0.1,
}


def viewport_priority(
    centroid: list[float],
    camera_position: list[float],
    view_matrix: list[list[float]],
    subgroup: int,
) -> int:
    """Use the actual OpenGL world-to-camera pose; lower paper priority wins."""
    if len(centroid) != 3 or len(camera_position) != 3:
        raise ValueError("centroid and camera position must have three coordinates")
    if not 0 <= subgroup <= 4:
        raise ValueError("viewport policy requires quality layer 0..4")
    direction = [float(c) - float(p) for c, p in zip(centroid, camera_position, strict=True)]
    distance = math.sqrt(sum(value * value for value in direction))
    forward = [-float(value) for value in view_matrix[2][:3]]
    norm = math.sqrt(sum(value * value for value in forward))
    if norm == 0 or not all(math.isfinite(value) for value in (*direction, *forward)):
        raise ValueError("invalid camera pose or group centroid")
    alignment = (
        1.0
        if distance == 0
        else sum(a * b for a, b in zip(direction, forward, strict=True)) / (distance * norm)
    )
    utility = 0.5 * max(0.0, 1.0 - min(distance, 100.0) / 100.0) + 0.5 * max(0.0, alignment)
    utility_bin = min(41, int(max(0.0, utility) * 42))
    return (41 - utility_bin) * 6 + subgroup


def native_priority(paper_priority: int) -> int:
    """Express paper 0=critical priorities in signed higher-first native order."""
    if type(paper_priority) is not int or not 0 <= paper_priority <= 255:
        raise ValueError("paper priority must be an integer in 0..255")
    return 127 - paper_priority


def native_module() -> Any:
    """Fail explicitly instead of silently selecting an untested native ABI."""
    for name, expected in VERSIONS.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"requires {name}=={expected}; found {actual}")
    import moq

    return moq


def quic_client(url: str, *, tls_verify: bool = True) -> Any:
    """Disable transport fallback and reconnect so failures remain observable."""
    if urlparse(url).scheme != "https":
        raise ValueError("native MoQ requires an https QUIC/WebTransport relay URL")
    return native_module().Client(
        url,
        tls_verify=tls_verify,
        websocket_enabled=False,
        reconnect=False,
    )


def quic_stats(client: Any) -> dict[str, Any]:
    """Require native QUIC counters; missing values never count as zero."""
    if client.session is None:
        raise RuntimeError("no connected native MoQ session")
    stats = client.session.stats()
    if stats.rtt_us is None or stats.packets_sent is None or stats.packets_received is None:
        raise RuntimeError("session does not expose required native QUIC statistics")
    return {
        name: getattr(stats, name)
        for name in (
            "rtt_us",
            "bytes_sent",
            "bytes_received",
            "bytes_lost",
            "packets_sent",
            "packets_received",
            "packets_lost",
        )
    }


@dataclass(frozen=True)
class Layer:
    """A byte-exact, independently subscribable application subgroup."""

    track: str
    group: int
    subgroup: int
    size: int
    sha256: str
    priority: int

    def __post_init__(self) -> None:
        native_priority(self.priority)
        if not self.track or "/" in self.track or self.group < 0 or self.subgroup < 0:
            raise ValueError("invalid layer identity")
        if self.size < 0 or len(self.sha256) != 64:
            raise ValueError("invalid layer size or SHA256")
        try:
            bytes.fromhex(self.sha256)
        except ValueError as error:
            raise ValueError("invalid layer SHA256") from error

    @property
    def name(self) -> str:
        return f"layers/{self.track}/{self.group}/{self.subgroup}"

    @classmethod
    def from_payload(
        cls, track: str, group: int, subgroup: int, payload: bytes, priority: int
    ) -> Layer:
        native_priority(priority)
        if not track or "/" in track or group < 0 or subgroup < 0:
            raise ValueError("invalid layer identity")
        return cls(
            track, group, subgroup, len(payload), hashlib.sha256(payload).hexdigest(), priority
        )

    def verify(self, payload: bytes) -> None:
        if len(payload) != self.size or hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ValueError(f"incomplete or corrupted layer: {self.name}")


def subscription(priority: int) -> Any:
    """Pinned FFI is unsigned higher-first: shift signed native order by 128."""
    return native_module().Subscription(
        priority=native_priority(priority) + 128,
        max_age_us=86_400_000_000,
        group_start=0,
        group_end=1,
    )


def publish_layers(
    broadcast: Any,
    layers: list[tuple[Layer, bytes]],
    *,
    metadata: dict[str, Any],
    chunk_size: int = 65536,
) -> tuple[dict[str, Any], list[Any]]:
    """Retain finite groups so late subscribers can recover static scene layers."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len({layer.name for layer, _ in layers}) != len(layers):
        raise ValueError("duplicate layer identity")
    moq = native_module()
    manifest = {
        "schema": SCHEMA,
        "versions": VERSIONS,
        "metadata": metadata,
        "layers": [asdict(layer) for layer, _ in layers],
    }
    handles = []
    catalog = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    for name, payload, priority in [(CATALOG, catalog, 0)] + [
        (layer.name, payload, layer.priority) for layer, payload in layers
    ]:
        if name != CATALOG:
            next(layer for layer, _ in layers if layer.name == name).verify(payload)
        track = broadcast.publish_track(
            name,
            moq.TrackInfo(
                priority=native_priority(priority) + 128,
                max_age_us=None,
            ),
        )
        group = track.append_group()
        for offset in range(0, len(payload), chunk_size):
            group.write_frame(payload[offset : offset + chunk_size])
        group.finish()
        handles.append(track)
    broadcast.announce()
    return manifest, handles


class LayerReceiver:
    """Deduplicate completed/in-flight layers and retry only incomplete transfers."""

    def __init__(
        self,
        broadcast: Any,
        *,
        timeout: float = 30,
        retries: int = 1,
        max_payload: int = 1 << 30,
        max_total_payload: int = 4 << 30,
    ) -> None:
        if timeout <= 0 or retries < 0 or max_payload < 0:
            raise ValueError("invalid receiver timeout, retries, or payload budget")
        self.broadcast = broadcast
        self.timeout = timeout
        self.retries = retries
        self.max_payload = max_payload
        self.max_total_payload = max_total_payload
        self.cache: dict[str, bytes] = {}
        self.active: dict[str, Any] = {}
        self.pending: dict[str, asyncio.Task[bytes]] = {}
        self.requests = 0
        self.updates = 0

    async def close(self) -> None:
        """Release only this receiver's transfers, retaining verified cached bytes."""
        pending = list(self.pending.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self.pending.clear()

    async def _read(self, name: str, priority: int, limit: int) -> bytes:
        track = await self.broadcast.subscribe_track(name, subscription(priority))
        self.active[name] = track
        try:
            chunks = []
            size = 0
            async for group in track:
                async for frame in group:
                    size += len(frame.payload)
                    if size > limit:
                        raise ValueError(f"payload exceeds declared limit: {name}")
                    chunks.append(frame.payload)
                break  # Static layer tracks contain exactly one complete group.
            return b"".join(chunks)
        finally:
            self.active.pop(name, None)
            track.cancel()

    async def catalog(self) -> dict[str, Any]:
        raw = await asyncio.wait_for(self._read(CATALOG, 0, 16 << 20), self.timeout)
        result = json.loads(raw)
        if result.get("schema") != SCHEMA:
            raise ValueError("unsupported MoQSplat manifest schema")
        entries = result.get("layers", [])
        if not entries or len(entries) > 10000:
            raise ValueError("catalog layer count outside 1..10000")
        layers = [Layer(**item) for item in entries]
        if len({layer.name for layer in layers}) != len(layers):
            raise ValueError("duplicate catalog layer identity")
        if (
            any(layer.size > self.max_payload for layer in layers)
            or sum(layer.size for layer in layers) > self.max_total_payload
        ):
            raise ValueError("catalog exceeds configured payload budget")
        return result

    def update(self, layer: Layer, priority: int) -> bool:
        preferences = subscription(priority)
        track = self.active.get(layer.name)
        if track is None:
            return False
        track.update(preferences)
        self.updates += 1
        return True

    async def _download(self, layer: Layer) -> bytes:
        for attempt in range(self.retries + 1):
            try:
                self.requests += 1
                payload = await asyncio.wait_for(
                    self._read(layer.name, layer.priority, layer.size),
                    self.timeout,
                )
                layer.verify(payload)
                self.cache[layer.name] = payload
                return payload
            except Exception:
                if attempt == self.retries:
                    raise
        raise RuntimeError("unreachable retry state")

    async def receive(self, layer: Layer) -> bytes:
        if not 0 <= layer.size <= self.max_payload:
            raise ValueError("layer exceeds configured payload budget")
        if layer.name in self.cache:
            layer.verify(self.cache[layer.name])
            return self.cache[layer.name]
        if layer.name not in self.pending:
            self.pending[layer.name] = asyncio.create_task(self._download(layer))
        task = self.pending[layer.name]
        try:
            payload = await asyncio.shield(task)
            layer.verify(payload)
            return payload
        finally:
            if task.done():
                self.pending.pop(layer.name, None)
