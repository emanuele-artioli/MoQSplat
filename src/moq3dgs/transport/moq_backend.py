"""Modern MoQ functional reproduction, not the paper's recovered subgroup wire format.

One application layer is one native MoQ track with bounded finite QUIC groups. The
application hierarchy remains explicit in the authenticated catalog and hashes.
Only moq-rs 0.5.2 / moq-ffi 0.4.20 are supported: moving-main APIs differ.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

VERSIONS = {"moq-rs": "0.5.2", "moq-ffi": "0.4.20"}
CATALOG = "moqsplat/catalog"
SCHEMA = "moqsplat-modern-layer-tracks-v2"
# moq-net's pinned native implementation hard-caps a group at 32 MiB/8192 frames.
# FFI offers no override. Keep payloads intact by splitting transport groups.
TRANSPORT_GROUP_BYTES = 16 << 20
ORIGIN_CACHE_BYTES = 4 << 30
# The pinned FFI cannot override the origin pool's separate 30-second idle expiry.
# A long media window prevents timeline eviction; it does not promise permanent cache.
PUBLISHER_MAX_AGE_US = 86_400_000_000
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


def quic_client(
    url: str,
    *,
    tls_verify: bool = True,
    reconnect: bool = False,
    reconnect_timeout: float = 30,
    cache_capacity_bytes: int = ORIGIN_CACHE_BYTES,
) -> Any:
    """Disable transport fallback and reconnect so failures remain observable."""
    if not math.isfinite(reconnect_timeout) or reconnect_timeout <= 0:
        raise ValueError(
            "reconnect_timeout must be finite and positive; zero means unlimited in SDK"
        )
    if urlparse(url).scheme != "https":
        raise ValueError("native MoQ requires an https QUIC/WebTransport relay URL")
    if cache_capacity_bytes <= 0:
        raise ValueError("native origin cache capacity must be positive")
    moq = native_module()
    return moq.Client(
        url,
        tls_verify=tls_verify,
        websocket_enabled=False,
        reconnect=reconnect,
        backoff=moq.Backoff(
            initial_us=100000,
            multiplier=2,
            max_us=1000000,
            timeout_us=int(reconnect_timeout * 1_000_000),
        ),
        publish=moq.OriginProducer(cache_capacity_bytes=cache_capacity_bytes),
        subscribe=moq.OriginProducer(cache_capacity_bytes=cache_capacity_bytes),
    )


class CatalogIdentityError(ValueError):
    """A recovered connection serves a different catalog; mixing scenes is forbidden."""


class ReconnectingBroadcast:
    """Bounded fresh-session recovery with catalog identity verified before reuse.

    Native auto-reconnect is disabled deliberately: fresh origins cannot answer the
    identity check from a previous connection's cached catalog. LayerReceiver retries
    incomplete layers from byte zero; completed application caches stay untouched.
    """

    def __init__(
        self,
        url: str,
        broadcast_path: str,
        *,
        tls_verify: bool = True,
        timeout: float = 30,
        max_reconnects: int = 2,
        backoff_initial: float = 0.1,
        backoff_max: float = 2,
        cache_capacity_bytes: int = 128 << 20,
    ) -> None:
        if timeout <= 0 or max_reconnects < 0 or not 0 <= backoff_initial <= backoff_max:
            raise ValueError("invalid reconnect limits")
        self.url, self.broadcast_path = url, broadcast_path
        self.tls_verify, self.timeout = tls_verify, timeout
        self.max_reconnects = max_reconnects
        self.backoff_initial, self.backoff_max = backoff_initial, backoff_max
        self.cache_capacity_bytes = cache_capacity_bytes
        self.client: Any = None
        self.remote: Any = None
        self.manifest: dict[str, Any] | None = None
        self.identity: str | None = None
        self.generation = 0
        self.reconnects = 0
        self._lock = asyncio.Lock()
        self._closed = False
        self._failure: BaseException | None = None
        self._closing = asyncio.Event()
        self._dial_task: asyncio.Task[Any] | None = None

    async def __aenter__(self) -> ReconnectingBroadcast:
        await self._establish()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._closed = True
        self._closing.set()
        dial = self._dial_task
        if dial is not None:
            dial.cancel()
            await asyncio.gather(dial, return_exceptions=True)
        await self._disconnect()

    def assert_healthy(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError("reconnecting broadcast is closed")

    def validate_layer(self, layer: Layer) -> None:
        self.assert_healthy()
        if self.manifest is None or asdict(layer) not in self.manifest["layers"]:
            raise CatalogIdentityError("requested layer is not in the verified catalog")

    async def _disconnect(self) -> None:
        client, self.client, self.remote = self.client, None, None
        if client is not None:
            await client.__aexit__(None, None, None)

    async def _establish(self) -> None:
        client = quic_client(
            self.url, tls_verify=self.tls_verify, cache_capacity_bytes=self.cache_capacity_bytes
        )

        async def handshake() -> tuple[Any, dict[str, Any], str]:
            await client.__aenter__()
            remote = await client.announced_broadcast(self.broadcast_path).available()
            manifest = await LayerReceiver(remote, timeout=self.timeout, retries=0).catalog()
            identity = hashlib.sha256(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            if self.identity is not None and identity != self.identity:
                raise CatalogIdentityError(
                    "catalog changed across reconnect; scene cache cannot be mixed"
                )
            return remote, manifest, identity

        try:
            dial = asyncio.create_task(handshake())
            self._dial_task = dial
            remote, manifest, identity = await asyncio.wait_for(dial, self.timeout)
            self.assert_healthy()
        except BaseException:
            await client.__aexit__(None, None, None)
            raise
        finally:
            self._dial_task = None
        self.client, self.remote = client, remote
        self.manifest, self.identity = manifest, identity
        self.generation += 1

    async def recover(self, failed_generation: int) -> None:
        async with self._lock:
            self.assert_healthy()
            if self.generation != failed_generation:
                return  # A concurrent failed layer already repaired this session.
            await self._disconnect()
            last_error: BaseException | None = None
            while self.reconnects < self.max_reconnects:
                delay = min(self.backoff_initial * 2**self.reconnects, self.backoff_max)
                self.reconnects += 1
                try:
                    await asyncio.wait_for(self._closing.wait(), delay)
                    raise RuntimeError("reconnecting broadcast is closed")
                except asyncio.TimeoutError:
                    pass
                self.assert_healthy()
                try:
                    await self._establish()
                    return
                except CatalogIdentityError as error:
                    self._failure = error
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    last_error = error
            self._failure = RuntimeError("native MoQ reconnect budget exhausted")
            raise self._failure from last_error

    async def subscribe_track(self, name: str, preferences: Any) -> Any:
        async with self._lock:
            self.assert_healthy()
            if self.remote is None:
                raise RuntimeError("native MoQ session is disconnected")
            return await self.remote.subscribe_track(name, preferences)


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

    @property
    def transport_groups(self) -> int:
        return max(1, (self.size + TRANSPORT_GROUP_BYTES - 1) // TRANSPORT_GROUP_BYTES)

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


def subscription(priority: int, group_count: int = 1, group_start: int = 0) -> Any:
    """Pinned FFI is unsigned higher-first: shift signed native order by 128."""
    return native_module().Subscription(
        priority=native_priority(priority) + 128,
        max_age_us=86_400_000_000,
        group_start=group_start,
        group_end=group_start + group_count,
    )


def publish_layers(
    broadcast: Any,
    layers: list[tuple[Layer, bytes]],
    *,
    metadata: dict[str, Any],
    chunk_size: int = 65536,
) -> tuple[dict[str, Any], list[Any]]:
    """Retain finite groups so late subscribers can recover static scene layers."""
    if not 2048 <= chunk_size <= TRANSPORT_GROUP_BYTES:
        raise ValueError("chunk_size must fit the native 8192-frame/32-MiB group budget")
    if len({layer.name for layer, _ in layers}) != len(layers):
        raise ValueError("duplicate layer identity")
    moq = native_module()
    manifest = {
        "schema": SCHEMA,
        "versions": VERSIONS,
        "metadata": metadata,
        "transport_group_bytes": TRANSPORT_GROUP_BYTES,
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
                max_age_us=PUBLISHER_MAX_AGE_US,
            ),
        )
        for start in range(0, max(1, len(payload)), TRANSPORT_GROUP_BYTES):
            group = track.append_group()
            stop = min(start + TRANSPORT_GROUP_BYTES, len(payload))
            for offset in range(start, stop, chunk_size):
                group.write_frame(payload[offset : min(offset + chunk_size, stop)])
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
        max_active_files: int = 4,
    ) -> None:
        if timeout <= 0 or retries < 0 or max_payload < 0 or max_active_files <= 0:
            raise ValueError("invalid receiver timeout, retries, or payload budget")
        self.broadcast = broadcast
        self.timeout = timeout
        self.retries = retries
        self.max_payload = max_payload
        self.max_total_payload = max_total_payload
        self.cache: dict[str, bytes] = {}
        self.active: dict[str, Any] = {}
        self.active_ranges: dict[str, tuple[int, int]] = {}
        self.priorities: dict[str, int] = {}
        self.pending: dict[str, asyncio.Task[bytes]] = {}
        self.file_cache: dict[str, Path] = {}
        self.pending_files: dict[tuple[str, str], asyncio.Task[Path]] = {}
        self._file_slots = asyncio.Semaphore(max_active_files)
        self._file_credit = asyncio.Condition()
        self._active_file_bytes = 0
        self.requests = 0
        self.updates = 0

    async def close(self) -> None:
        """Release only this receiver's transfers, retaining verified cached bytes."""
        pending = [*self.pending.values(), *self.pending_files.values()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self.pending.clear()
        self.pending_files.clear()

    async def _read(self, name: str, priority: int, limit: int) -> bytes:
        return b"".join([chunk async for chunk in self._chunks(name, priority, limit)])

    async def _chunks(self, name: str, priority: int, limit: int) -> Any:
        """Stream a full-range native subscription without a Python whole-layer buffer.

        Full-range demand preserves pinned SDK late-subscriber semantics and live
        priority updates. The native cache target is not a bound on active buffers.
        """
        group_count = max(1, (limit + TRANSPORT_GROUP_BYTES - 1) // TRANSPORT_GROUP_BYTES)
        size = 0
        track = await self.broadcast.subscribe_track(
            name,
            subscription(self.priorities.get(name, priority), group_count),
        )
        self.active[name] = track
        self.active_ranges[name] = (0, group_count)
        try:
            sequence = 0
            async for group in track:
                if group.sequence != sequence:
                    raise ValueError(f"missing/out-of-order transport group for {name}")
                group_size = 0
                async for frame in group:
                    size += len(frame.payload)
                    group_size += len(frame.payload)
                    if size > limit:
                        raise ValueError(f"payload exceeds declared limit: {name}")
                    if group_size > TRANSPORT_GROUP_BYTES:
                        raise ValueError(f"transport group exceeds budget: {name}")
                    yield frame.payload
                sequence += 1
                if sequence < group_count and group_size != TRANSPORT_GROUP_BYTES:
                    raise ValueError(f"partial non-final transport group for {name}")
                if sequence == group_count:
                    break
            if sequence != group_count:
                raise ValueError(f"incomplete transport group range for {name}")
        finally:
            self.active.pop(name, None)
            self.active_ranges.pop(name, None)
            track.cancel()

    async def catalog(self) -> dict[str, Any]:
        raw = await asyncio.wait_for(self._read(CATALOG, 0, 16 << 20), self.timeout)
        result = json.loads(raw)
        if result.get("schema") != SCHEMA:
            raise ValueError("unsupported MoQSplat manifest schema")
        if result.get("transport_group_bytes") != TRANSPORT_GROUP_BYTES:
            raise ValueError("unsupported transport group segmentation")
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
        start, count = self.active_ranges.get(layer.name, (0, layer.transport_groups))
        preferences = subscription(priority, count, start)
        track = self.active.get(layer.name)
        if track is None:
            return False
        try:
            track.update(preferences)
        except native_module().Error:
            return False  # The in-flight download owns bounded reconnection/retry.
        self.priorities[layer.name] = priority
        self.updates += 1
        return True

    async def _download(self, layer: Layer) -> bytes:
        for attempt in range(self.retries + 1):
            generation = getattr(self.broadcast, "generation", 0)
            try:
                self.requests += 1
                payload = await asyncio.wait_for(
                    self._read(layer.name, layer.priority, layer.size),
                    self.timeout,
                )
                layer.verify(payload)
                self.cache[layer.name] = payload
                return payload
            except CatalogIdentityError:
                raise
            except Exception:
                if attempt == self.retries:
                    raise
                if isinstance(self.broadcast, ReconnectingBroadcast):
                    await self.broadcast.recover(generation)
        raise RuntimeError("unreachable retry state")

    async def receive(self, layer: Layer) -> bytes:
        if isinstance(self.broadcast, ReconnectingBroadcast):
            self.broadcast.validate_layer(layer)
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

    @staticmethod
    def _file_verified(path: Path, layer: Layer) -> bool:
        if not path.is_file() or path.stat().st_size != layer.size:
            return False
        checksum = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                checksum.update(block)
        return checksum.hexdigest() == layer.sha256

    @staticmethod
    async def _file_operation(function: Any, *args: Any) -> Any:
        """Join a cancelled worker write before closing/removing its owned temp file."""
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def _download_file(self, layer: Layer, directory: Path) -> Path:
        async with self._file_slots:
            final = directory / f"{layer.sha256}.bin"
            if await asyncio.to_thread(self._file_verified, final, layer):
                self.file_cache[layer.name] = final
                return final
            capacity = getattr(self.broadcast, "cache_capacity_bytes", None)
            if capacity is None:
                return await self._download_file_active(layer, directory)
            # Full-range subscriptions must fit the pinned SDK's native pool.
            # Keep catalog space and limit aggregate in-flight layer demand;
            # completed files need no network/native-buffer reservation.
            available = capacity - (16 << 20)
            if layer.size > available:
                minimum = (layer.size + (16 << 20) + (1 << 20) - 1) // (1 << 20)
                raise ValueError(
                    f"layer exceeds native cache target; use --native-cache-mib {minimum} "
                    "or prepare smaller spatial groups"
                )
            async with self._file_credit:
                await self._file_credit.wait_for(
                    lambda: self._active_file_bytes + layer.size <= available
                )
                self._active_file_bytes += layer.size
            try:
                return await self._download_file_active(layer, directory)
            finally:
                async with self._file_credit:
                    self._active_file_bytes -= layer.size
                    self._file_credit.notify_all()

    async def _download_file_active(self, layer: Layer, directory: Path) -> Path:
        final = directory / f"{layer.sha256}.bin"
        if await asyncio.to_thread(self._file_verified, final, layer):
            self.file_cache[layer.name] = final
            return final
        for attempt in range(self.retries + 1):
            generation = getattr(self.broadcast, "generation", 0)
            stream = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=directory,
                prefix=f"{layer.sha256}.",
                suffix=".part",
                delete=False,
            )
            partial = Path(stream.name)
            checksum = hashlib.sha256()
            size = 0
            try:
                self.requests += 1

                async def transfer() -> None:
                    nonlocal size
                    async for chunk in self._chunks(layer.name, layer.priority, layer.size):
                        size += len(chunk)
                        checksum.update(chunk)
                        await self._file_operation(stream.write, chunk)

                await asyncio.wait_for(transfer(), self.timeout)
                if size != layer.size or checksum.hexdigest() != layer.sha256:
                    raise ValueError(f"incomplete or corrupted layer: {layer.name}")
                await self._file_operation(stream.flush)
                await self._file_operation(os.fsync, stream.fileno())
                stream.close()
                os.replace(partial, final)
                self.file_cache[layer.name] = final
                return final
            except CatalogIdentityError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception:
                if attempt == self.retries:
                    raise
                if isinstance(self.broadcast, ReconnectingBroadcast):
                    await self.broadcast.recover(generation)
            finally:
                stream.close()
                partial.unlink(missing_ok=True)
        raise RuntimeError("unreachable file retry state")

    async def receive_file(self, layer: Layer, cache_dir: str | Path) -> Path:
        """Stream to a verified atomic content-addressed file; retry partials from zero.

        Native cache capacity is a separate configured target, not an RSS guarantee.
        Completed files survive receiver close, reconnect, and a fresh receiver.
        """
        if isinstance(self.broadcast, ReconnectingBroadcast):
            self.broadcast.validate_layer(layer)
        if not 0 <= layer.size <= self.max_payload:
            raise ValueError("layer exceeds configured payload budget")
        directory = Path(cache_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        key = (layer.name, str(directory))
        if key not in self.pending_files:
            self.pending_files[key] = asyncio.create_task(self._download_file(layer, directory))
        task = self.pending_files[key]
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self.pending_files.pop(key, None)
