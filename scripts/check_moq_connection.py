#!/usr/bin/env python3
"""CPU-only external-network check; no Torch, CUDA or rendering required."""

import argparse
import asyncio
import json
from pathlib import Path

from moq3dgs.transport.array_checks import verify_arrays
from moq3dgs.transport.moq_backend import Layer, LayerReceiver, ReconnectingBroadcast, quic_stats


async def check(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    async with ReconnectingBroadcast(
        args.relay,
        args.broadcast,
        tls_verify=not args.insecure,
        timeout=args.timeout,
        max_reconnects=3,
        cache_capacity_bytes=args.native_cache_mib * 1024 * 1024,
    ) as connection:
        receiver = LayerReceiver(connection, timeout=args.timeout)
        catalog = await receiver.catalog()
        layers = sorted((Layer(**item) for item in catalog["layers"]), key=lambda x: x.priority)
        if args.max_layers:
            layers = layers[: args.max_layers]

        async def receive(layer):
            path = await receiver.receive_file(layer, output / "cache")
            await asyncio.to_thread(
                verify_arrays, path, catalog["metadata"]["source_array_hashes"][layer.name]
            )

        try:
            await asyncio.gather(*(receive(layer) for layer in layers))
            initial_requests = receiver.requests
            await asyncio.gather(
                *(receiver.receive_file(layer, output / "cache") for layer in layers)
            )
            result = {
                "valid": True,
                "transport": "QUIC",
                "rendering_tested": False,
                "catalog_identity": connection.identity,
                "layers_received": len(layers),
                "bytes_received_payload": sum(layer.size for layer in layers),
                "source_attributes_exact": True,
                "reconnects": connection.reconnects,
                "cache_revisit_requests": receiver.requests - initial_requests,
                "quic_stats": quic_stats(connection.client),
            }
            (output / "summary.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
        finally:
            await receiver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relay", required=True)
    parser.add_argument("--broadcast", default="moqsplat/scene")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--native-cache-mib", type=int, default=128)
    parser.add_argument(
        "--max-layers",
        type=int,
        default=8,
        help="bounded first test; 0 checks the complete catalog",
    )
    parser.add_argument(
        "--insecure", action="store_true", help="explicit trusted self-signed endpoint"
    )
    args = parser.parse_args()
    if args.max_layers < 0 or args.timeout <= 0 or args.native_cache_mib < 17:
        parser.error("nonnegative layer count and positive timeout required")
    asyncio.run(check(args))


if __name__ == "__main__":
    main()
