"""Real native QUIC recovery contracts; byte fixtures are not scene evidence."""

import asyncio
import hashlib
import json
from contextlib import suppress

import pytest

from moq3dgs.transport.moq_backend import (
    CATALOG,
    ORIGIN_CACHE_BYTES,
    PUBLISHER_MAX_AGE_US,
    CatalogIdentityError,
    Layer,
    LayerReceiver,
    ReconnectingBroadcast,
    native_module,
    publish_layers,
    quic_client,
)


class Relay:
    async def start(self, bind="127.0.0.1:0"):
        moq = native_module()
        origin = moq.OriginProducer(cache_capacity_bytes=ORIGIN_CACHE_BYTES)
        self.server = moq.Server(bind, tls_generate=["localhost"], publish=origin, subscribe=origin)
        await self.server.__aenter__()
        self.bind = self.server.local_addr
        self.url = "https://" + self.bind
        self.sessions = []
        self.task = asyncio.create_task(self.accept())
        return self

    async def accept(self):
        async for request in self.server:
            assert request.transport == native_module().Transport.QUIC
            self.sessions.append(await request.accept())

    async def stop(self):
        for session in self.sessions:
            session.shutdown()
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task
        await self.server.__aexit__(None, None, None)


class ObservedReceiver(LayerReceiver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_chunk = asyncio.Event()

    async def _chunks(self, name, priority, limit):
        async for chunk in super()._chunks(name, priority, limit):
            yield chunk
            if name != CATALOG:
                self.first_chunk.set()


@pytest.mark.asyncio
async def test_native_restart_retries_partial_file_preserves_complete_and_reannounces(tmp_path):
    moq = native_module()
    relay = await Relay().start()
    complete_bytes = b"verified-before-restart" * 100
    slow_bytes = b"exact-incomplete-layer" * 10000
    complete = Layer.from_payload("scene", 0, 0, complete_bytes, 0)
    slow = Layer.from_payload("scene", 0, 1, slow_bytes, 20)
    try:
        async with quic_client(
            relay.url, tls_verify=False, reconnect=True, reconnect_timeout=5
        ) as publisher:
            broadcast = publisher.create_broadcast("recovery/scene")
            # Publish the manifest and small layer, but leave one actual native
            # group unfinished so its transfer cannot complete from buffered data.
            manifest, handles = publish_layers(broadcast, [(complete, complete_bytes)], metadata={})
            manifest["layers"].append(slow.__dict__)
            # New exact catalog track must precede announcement; use another broadcast.
            broadcast.close()
            broadcast = publisher.create_broadcast("recovery/live")
            info = moq.TrackInfo(max_age_us=PUBLISHER_MAX_AGE_US)
            for name, data in [
                (CATALOG, json.dumps(manifest).encode()),
                (complete.name, complete_bytes),
            ]:
                track = broadcast.publish_track(name, info)
                group = track.append_group()
                group.write_frame(data)
                group.finish()
                handles.append(track)
            slow_track = broadcast.publish_track(slow.name, info)
            group = slow_track.append_group()
            group.write_frame(slow_bytes[:65536])
            broadcast.announce()
            old_epoch = publisher.session.epoch()
            async with ReconnectingBroadcast(
                relay.url, "recovery/live", tls_verify=False, timeout=5, max_reconnects=3
            ) as connection:
                receiver = ObservedReceiver(connection, timeout=8, retries=2)
                try:
                    cached = await receiver.receive_file(complete, tmp_path)
                    assert await receiver.receive(complete) == complete_bytes
                    cached_stat = cached.stat()
                    receiver.first_chunk.clear()
                    transfer = asyncio.create_task(receiver.receive_file(slow, tmp_path))
                    await asyncio.wait_for(receiver.first_chunk.wait(), 3)
                    assert list(tmp_path.glob("*.part"))
                    await relay.stop()
                    await relay.start(relay.bind)

                    async def reconnected():
                        while publisher.session.epoch() == old_epoch:
                            await publisher.session.status()

                    await asyncio.wait_for(reconnected(), 5)
                    group.write_frame(slow_bytes[65536:])
                    group.finish()
                    recovered = await asyncio.wait_for(transfer, 12)
                    assert recovered.read_bytes() == slow_bytes
                    assert connection.reconnects >= 1
                    assert receiver.requests >= 3  # completed + interrupted + retried
                    assert await receiver.receive_file(complete, tmp_path) == cached
                    requests = receiver.requests
                    assert await receiver.receive(complete) == complete_bytes
                    assert receiver.requests == requests
                    assert cached.stat().st_mtime_ns == cached_stat.st_mtime_ns
                    assert not list(tmp_path.glob("*.part"))
                    assert slow.name not in receiver.cache  # file API never retains full bytes
                    async with ReconnectingBroadcast(
                        relay.url, "recovery/live", tls_verify=False, timeout=5
                    ) as fresh:
                        newcomer = LayerReceiver(fresh, timeout=5)
                        assert await newcomer.receive(slow) == slow_bytes
                finally:
                    await receiver.close()
            broadcast.close()
    finally:
        await relay.stop()


@pytest.mark.asyncio
async def test_reconnect_refuses_changed_catalog(tmp_path):
    relay = await Relay().start()
    first_bytes, second_bytes = b"first scene", b"changed scene"
    first = Layer.from_payload("scene", 0, 0, first_bytes, 0)
    second = Layer.from_payload("scene", 0, 0, second_bytes, 0)
    publisher = quic_client(relay.url, tls_verify=False)
    await publisher.__aenter__()
    broadcast = publisher.create_broadcast("identity/scene")
    _, handles = publish_layers(broadcast, [(first, first_bytes)], metadata={})
    assert handles
    try:
        async with ReconnectingBroadcast(
            relay.url, "identity/scene", tls_verify=False, timeout=4
        ) as connection:
            receiver = LayerReceiver(connection)
            cached = await receiver.receive_file(first, tmp_path)
            await publisher.__aexit__(None, None, None)
            await relay.stop()
            await relay.start(relay.bind)
            async with quic_client(relay.url, tls_verify=False) as changed_publisher:
                changed = changed_publisher.create_broadcast("identity/scene")
                _, changed_handles = publish_layers(changed, [(second, second_bytes)], metadata={})
                assert changed_handles
                with pytest.raises(CatalogIdentityError):
                    await connection.recover(connection.generation)
                assert cached.read_bytes() == first_bytes
                with pytest.raises(CatalogIdentityError):
                    await receiver.receive_file(first, tmp_path)
                changed.close()
    finally:
        await publisher.__aexit__(None, None, None)
        await relay.stop()


@pytest.mark.asyncio
async def test_file_cache_new_receiver_and_cancel_partial(tmp_path):
    payload = b"disk-cached-exact" * 100
    layer = Layer.from_payload("scene", 0, 0, payload, 0)
    receiver = LayerReceiver(None, retries=0)
    started = asyncio.Event()

    async def chunks(*_args):
        yield payload[:10]
        started.set()
        await asyncio.Event().wait()

    receiver._chunks = chunks
    task = asyncio.create_task(receiver.receive_file(layer, tmp_path))
    await started.wait()
    await receiver.close()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not receiver.file_cache
    assert not list(tmp_path.glob("*.part"))
    final = tmp_path / f"{hashlib.sha256(payload).hexdigest()}.bin"
    final.write_bytes(payload)
    other = LayerReceiver(None)
    assert await other.receive_file(layer, tmp_path) == final
    assert other.requests == 0


@pytest.mark.asyncio
async def test_file_transfer_respects_configured_native_subscriber_cache(tmp_path):
    relay = await Relay().start()
    payload = (bytes(range(256)) * 4096) * 160
    layer = Layer.from_payload("scene", 0, 0, payload, 0)
    try:
        async with quic_client(relay.url, tls_verify=False) as publisher:
            broadcast = publisher.create_broadcast("bounded/scene")
            _, handles = publish_layers(broadcast, [(layer, payload)], metadata={})
            assert handles
            async with ReconnectingBroadcast(
                relay.url,
                "bounded/scene",
                tls_verify=False,
                cache_capacity_bytes=128 << 20,
                timeout=15,
            ) as connection:
                receiver = LayerReceiver(connection, timeout=15, retries=0)
                with pytest.raises(ValueError, match="native cache target"):
                    await receiver.receive_file(layer, tmp_path)
                assert receiver.requests == 0
                await receiver.close()
            async with ReconnectingBroadcast(
                relay.url,
                "bounded/scene",
                tls_verify=False,
                cache_capacity_bytes=320 << 20,
                timeout=15,
            ) as connection:
                receiver = LayerReceiver(connection, timeout=15, retries=0)
                paths = await asyncio.gather(
                    receiver.receive_file(layer, tmp_path), receiver.receive_file(layer, tmp_path)
                )
                assert paths[0] == paths[1]
                assert paths[0].stat().st_size == len(payload) < connection.cache_capacity_bytes
                assert await asyncio.to_thread(receiver._file_verified, paths[0], layer)
                assert receiver.requests == 1
                assert receiver.cache == {}
                assert await receiver.receive_file(layer, tmp_path) == paths[0]
                assert receiver.requests == 1
                await receiver.close()
            broadcast.close()
    finally:
        await relay.stop()


@pytest.mark.asyncio
async def test_reconnect_budget_exhaustion_and_cancellation(tmp_path):
    relay = await Relay().start()
    payload = b"completed-data-survives-exhaustion"
    layer = Layer.from_payload("scene", 0, 0, payload, 0)
    try:
        async with quic_client(relay.url, tls_verify=False) as publisher:
            broadcast = publisher.create_broadcast("budget/scene")
            _, handles = publish_layers(broadcast, [(layer, payload)], metadata={})
            assert handles
            async with ReconnectingBroadcast(
                relay.url,
                "budget/scene",
                tls_verify=False,
                timeout=0.25,
                max_reconnects=1,
                backoff_initial=0,
                backoff_max=0,
            ) as connection:
                receiver = LayerReceiver(connection)
                cached = await receiver.receive_file(layer, tmp_path)
                await relay.stop()
                with pytest.raises(RuntimeError, match="budget exhausted"):
                    await asyncio.wait_for(connection.recover(connection.generation), 2)
                assert connection.reconnects == 1
                assert cached.read_bytes() == payload
            # Recovery during backoff must stop immediately when the owner closes.
            await relay.start(relay.bind)
            # The original non-reconnecting publisher died with the relay: use a new one.
            async with quic_client(relay.url, tls_verify=False) as fresh_publisher:
                fresh = fresh_publisher.create_broadcast("budget/close")
                _, fresh_handles = publish_layers(fresh, [(layer, payload)], metadata={})
                assert fresh_handles
                connection = ReconnectingBroadcast(
                    relay.url,
                    "budget/close",
                    tls_verify=False,
                    timeout=2,
                    backoff_initial=1,
                    backoff_max=1,
                )
                await connection.__aenter__()
                recovery = asyncio.create_task(connection.recover(connection.generation))
                await asyncio.sleep(0.01)
                await connection.__aexit__(None, None, None)
                with pytest.raises(RuntimeError, match="closed"):
                    await asyncio.wait_for(recovery, 0.5)
                assert connection.client is None
                fresh.close()
            broadcast.close()
    finally:
        await relay.stop()
