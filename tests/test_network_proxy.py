"""Actual native QUIC through isolated, bounded packet impairment."""

import asyncio
from contextlib import suppress

import pytest

from moq3dgs.transport.moq_backend import (
    ORIGIN_CACHE_BYTES,
    Layer,
    LayerReceiver,
    ReconnectingBroadcast,
    native_module,
    publish_layers,
    quic_client,
)
from moq3dgs.transport.udp_proxy import UDPProxy


@pytest.mark.asyncio
async def test_native_quic_over_delayed_lossy_paced_udp(tmp_path):
    moq = native_module()
    payload = b"actual-quic-impairment" * 100000
    layer = Layer.from_payload("network", 0, 0, payload, 0)
    origin = moq.OriginProducer(cache_capacity_bytes=ORIGIN_CACHE_BYTES)
    sessions = []
    async with moq.Server(
        "127.0.0.1:0", tls_generate=["localhost"], publish=origin, subscribe=origin
    ) as relay:

        async def accept():
            async for request in relay:
                assert request.transport == moq.Transport.QUIC
                sessions.append(await request.accept())

        accepting = asyncio.create_task(accept())
        host, port = relay.local_addr.rsplit(":", 1)
        proxy = await UDPProxy.start((host, int(port)), mbps=40, delay_ms=2, loss=0.002, seed=1)
        try:
            async with quic_client("https://" + relay.local_addr, tls_verify=False) as publisher:
                broadcast = publisher.create_broadcast("test/impaired")
                _, handles = publish_layers(broadcast, [(layer, payload)], metadata={})
                assert handles
                url = f"https://127.0.0.1:{proxy.address[1]}"
                async with ReconnectingBroadcast(
                    url, "test/impaired", tls_verify=False, timeout=15
                ) as connection:
                    receiver = LayerReceiver(connection, timeout=15)
                    path = await receiver.receive_file(layer, tmp_path)
                    assert path.read_bytes() == payload
                    assert proxy.counters["dropped_packets"] > 0
                    assert proxy.counters["forwarded_bytes"] > len(payload)
                    await receiver.close()
                broadcast.close()
        finally:
            await proxy.close()
            accepting.cancel()
            with suppress(asyncio.CancelledError):
                await accepting
            for session in sessions:
                session.shutdown()
