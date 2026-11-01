"""Transport-only tests; byte fixtures are not real-scene reproduction evidence."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from moq3dgs.transport.moq_backend import (
    ORIGIN_CACHE_BYTES,
    TRANSPORT_GROUP_BYTES,
    Layer,
    LayerReceiver,
    native_module,
    native_priority,
    publish_layers,
    quic_client,
    quic_stats,
    subscription,
    viewport_priority,
)


def test_priority_and_integrity():
    assert [native_priority(p) for p in (0, 127, 255)] == [127, 0, -128]
    assert [subscription(p).priority for p in (0, 127, 255)] == [255, 128, 0]
    for invalid in (-1, 256, True):
        with pytest.raises(ValueError):
            native_priority(invalid)
    layer = Layer.from_payload("scene", 0, 0, b"complete", 0)
    with pytest.raises(ValueError, match="corrupted"):
        layer.verify(b"incomple")
    with pytest.raises(ValueError):
        quic_client("ws://localhost:4433")
    with pytest.raises(RuntimeError, match="QUIC"):
        quic_stats(
            SimpleNamespace(
                session=SimpleNamespace(
                    stats=lambda: SimpleNamespace(
                        rtt_us=None, packets_sent=None, packets_received=None
                    )
                )
            )
        )


def test_real_pose_rotation_changes_viewport_priority_and_quality_ties():
    front = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    away = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    near = viewport_priority([0, 0, -10], [0, 0, 0], front, 0)
    turned = viewport_priority([0, 0, -10], [0, 0, 0], away, 0)
    assert near < turned
    assert viewport_priority([0, 0, -10], [0, 0, 0], front, 4) == near + 4
    assert viewport_priority([0, 0, -90], [0, 0, 0], front, 0) > near


def test_pinned_native_group_budget_requires_segmentation():
    moq = native_module()
    broadcast = moq.BroadcastProducer()
    track = broadcast.publish_track("native-size-limit")
    group = track.append_group()
    group.write_frame(b"x" * (16 << 20))
    group.write_frame(b"y" * (16 << 20))
    with pytest.raises(moq.Error, match="group too large"):
        group.write_frame(b"z")
    broadcast.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "groups",
    [
        [(1, b"abcd"), (2, b"ef")],  # Missing group zero / out-of-order delivery.
        [(0, b"abc"), (1, b"def")],  # Non-final group is incomplete.
        [(0, b"abcd"), (1, b"e")],  # Final group shorter than catalog-declared bytes.
    ],
)
async def test_invalid_segmented_layer_is_never_cached(monkeypatch, groups):
    import moq3dgs.transport.moq_backend as backend

    monkeypatch.setattr(backend, "TRANSPORT_GROUP_BYTES", 4)

    class Group:
        def __init__(self, sequence, payload):
            self.sequence, self.payload = sequence, payload

        async def __aiter__(self):
            yield SimpleNamespace(payload=self.payload)

    class Track:
        async def __aiter__(self):
            for sequence, payload in groups:
                yield Group(sequence, payload)

        def cancel(self):
            pass

    async def subscribe(*_args):
        return Track()

    receiver = LayerReceiver(SimpleNamespace(subscribe_track=subscribe), retries=0)
    layer = Layer.from_payload("scene", 0, 0, b"abcdef", 0)
    with pytest.raises(ValueError):
        await receiver.receive(layer)
    assert receiver.cache == {}


@pytest.mark.asyncio
async def test_retry_hash_failure_and_cached_deduplication():
    layer = Layer.from_payload("scene", 0, 0, b"correct", 10)
    receiver = LayerReceiver(None, retries=1)
    attempts = []

    async def read(*args):
        attempts.append(args)
        return b"wrong" if len(attempts) == 1 else b"correct"

    receiver._read = read
    result = await asyncio.gather(receiver.receive(layer), receiver.receive(layer))
    assert result == [b"correct", b"correct"]
    assert await receiver.receive(layer) == b"correct"
    assert receiver.requests == 2
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_incomplete_layer_is_not_cached_and_update_targets_active_track():
    layer = Layer.from_payload("scene", 0, 0, b"complete", 10)
    receiver = LayerReceiver(None, retries=0)
    changes = []
    receiver.active[layer.name] = SimpleNamespace(update=changes.append)
    assert receiver.update(layer, 0)
    assert changes[0].priority == 255
    assert not receiver.update(Layer.from_payload("other", 0, 0, b"x", 0), 0)

    async def read(*_args):
        return b"incomplete"

    receiver._read = read
    with pytest.raises(ValueError, match="corrupted"):
        await receiver.receive(layer)
    assert not receiver.cache
    assert not receiver.pending


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_size", [180000, (32 << 20) + 65537])
async def test_native_quic_two_clients_and_late_subscription(payload_size, late_delay=0):
    """Exercise real Rust QUIC and explicit server transport discrimination."""
    moq = native_module()
    payload = (b"native-quic-object" * ((payload_size + 17) // 18))[:payload_size]
    layer = Layer.from_payload("scene", 0, 0, payload, 0)
    accepted = []
    sessions = []
    origin = moq.OriginProducer(cache_capacity_bytes=ORIGIN_CACHE_BYTES)
    async with moq.Server(
        "127.0.0.1:0", tls_generate=["localhost"], publish=origin, subscribe=origin
    ) as server:

        async def accept():
            async for request in server:
                assert request.transport == moq.Transport.QUIC
                accepted.append(request.transport)
                sessions.append(await request.accept())

        task = asyncio.create_task(accept())
        try:
            url = "https://" + server.local_addr
            async with quic_client(url, tls_verify=False) as publisher:
                broadcast = publisher.create_broadcast("test/scene")
                manifest, handles = publish_layers(broadcast, [(layer, payload)], metadata={})
                assert handles
                assert manifest["transport_group_bytes"] == TRANSPORT_GROUP_BYTES
                for _ in range(2):
                    if _ == 1 and late_delay:
                        await asyncio.sleep(late_delay)
                    async with quic_client(url, tls_verify=False) as subscriber:
                        remote = await asyncio.wait_for(
                            subscriber.announced_broadcast("test/scene").available(),
                            10,
                        )
                        receiver = LayerReceiver(remote, timeout=10)
                        assert await receiver.catalog() == manifest
                        assert await receiver.receive(layer) == payload
                        assert await receiver.receive(layer) == payload
                        assert receiver.requests == 1
                        assert quic_stats(subscriber)["packets_received"] > 0
                broadcast.close()
                assert len(accepted) == 3
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            for session in sessions:
                session.shutdown()
