"""Task-local UDP impairment for QUIC tests; never alters a shared host interface.

This is controlled application-level packet impairment, not a claim about an
actual internet path or equivalence to a kernel network emulator.
"""

from __future__ import annotations

import asyncio
import random
import time


class UDPProxy(asyncio.DatagramProtocol):
    """Pace relay-to-client traffic; bounded queues, delay/loss and one outage."""

    def __init__(
        self,
        upstream,
        *,
        mbps=0,
        delay_ms=0,
        loss=0,
        seed=1,
        outage_after=0,
        outage_seconds=0,
        queue_bytes=4 << 20,
    ):
        if mbps < 0 or delay_ms < 0 or not 0 <= loss < 1 or queue_bytes <= 0:
            raise ValueError("invalid bounded UDP impairment")
        self.upstream = upstream
        self.rate = mbps * 1_000_000 / 8
        self.delay = delay_ms / 1000
        self.loss = loss
        self.random = random.Random(seed)
        self.outage_after, self.outage_seconds = outage_after, outage_seconds
        self.limit = queue_bytes
        self.transport = None
        self.flows = {}
        self.creating = {}
        self.tasks = set()
        self.handles = set()
        self.queued = 0
        self.next_delivery = 0
        self.started = time.monotonic()
        self.counters = {
            "forwarded_packets": 0,
            "forwarded_bytes": 0,
            "dropped_packets": 0,
            "queue_drops": 0,
        }

    @classmethod
    async def start(cls, upstream, **options):
        proxy = cls(upstream, **options)
        await asyncio.get_running_loop().create_datagram_endpoint(
            lambda: proxy, local_addr=("127.0.0.1", 0)
        )
        return proxy

    def connection_made(self, transport):
        self.transport = transport

    @property
    def address(self):
        return self.transport.get_extra_info("sockname")

    def _send(self, transport, data, address=None, paced=False):
        now = time.monotonic()
        elapsed = now - self.started
        if (
            self.outage_seconds
            and self.outage_after <= elapsed < self.outage_after + self.outage_seconds
        ) or self.random.random() < self.loss:
            self.counters["dropped_packets"] += 1
            return
        when = now + self.delay
        if paced and self.rate:
            when = max(when, self.next_delivery)
        if self.queued + len(data) > self.limit or when - now > 3:
            self.counters["queue_drops"] += 1
            return
        if paced and self.rate:
            self.next_delivery = when + len(data) / self.rate
        self.queued += len(data)
        holder = []

        def deliver():
            self.handles.discard(holder[0])
            self.queued -= len(data)
            if not transport.is_closing():
                transport.sendto(data, address)
                self.counters["forwarded_packets"] += 1
                self.counters["forwarded_bytes"] += len(data)

        handle = asyncio.get_running_loop().call_later(max(0, when - now), deliver)
        holder.append(handle)
        self.handles.add(handle)

    def datagram_received(self, data, address):
        if address in self.flows:
            self._send(self.flows[address], data)
            return
        if address in self.creating:
            queued = self.creating[address]
            if sum(map(len, queued)) + len(data) <= 65536:
                queued.append(data)
            else:
                self.counters["queue_drops"] += 1
            return
        self.creating[address] = [data]
        parent = self

        class ReturnPath(asyncio.DatagramProtocol):
            def datagram_received(self, payload, _):
                parent._send(parent.transport, payload, address, paced=True)

        async def create():
            transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
                ReturnPath, remote_addr=self.upstream
            )
            self.flows[address] = transport
            for packet in self.creating.pop(address):
                self._send(transport, packet)

        task = asyncio.create_task(create())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def close(self):
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for handle in self.handles:
            handle.cancel()
        self.handles.clear()
        for transport in self.flows.values():
            transport.close()
        self.transport.close()
