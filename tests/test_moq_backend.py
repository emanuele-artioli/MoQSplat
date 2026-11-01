"""Transport-only tests; byte fixtures are not real-scene reproduction evidence."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from moq3dgs.transport.moq_backend import (
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
async def test_native_quic_two_clients_and_late_subscription():
    """Exercise real Rust QUIC and explicit server transport discrimination."""
    moq = native_module()
    payload = b"native-quic-object" * 10000
    layer = Layer.from_payload("scene", 0, 0, payload, 0)
    accepted = []
    sessions = []
    async with moq.Server("127.0.0.1:0", tls_generate=["localhost"]) as server:

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
                for _ in range(2):
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
