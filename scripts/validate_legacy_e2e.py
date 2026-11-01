"""Exercise the corrected legacy TCP client, culling and renderer on real assets."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from moq3dgs.transport.client import MoQClient
from moq3dgs.transport.server import create_server
from moq3dgs.viewport.trace import load_trace, replay_trace


async def validate(args):
    updates = list(replay_trace(load_trace(args.trace)))
    if not 1 <= args.frames <= len(updates):
        raise ValueError(f"--frames must be between 1 and {len(updates)}")
    args.output.mkdir(parents=True, exist_ok=False)
    server = create_server(
        str(args.scene), num_clusters=4, host="127.0.0.1", port=0, device=args.device
    )
    client = None
    try:
        await server.start()
        port = server._server.sockets[0].getsockname()[1]
        client = MoQClient(
            host="127.0.0.1",
            port=port,
            output_dir=str(args.output),
            device=args.device,
            image_width=320,
            image_height=240,
        )
        await client.connect()
        await client.start_receiving()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and client.cache.num_entries < 24:
            await client.send_viewport_update(updates[0])
            await asyncio.sleep(0.05)
        paths = []
        for index, update in enumerate(updates[: args.frames]):
            await client.send_viewport_update(update)
            await asyncio.sleep(0.05)
            path = client.render_frame(
                index, update.view_matrix, update.fov, update.camera_position,
                fov_angles=update.fov_angles,
            )
            if path is None:
                raise AssertionError("Legacy client failed to render received real-scene data")
            paths.append(Path(path))
        cache_entries = client.cache.num_entries
        clusters_received = client._clusters_received
    finally:
        try:
            if client is not None:
                await client.disconnect()
        finally:
            await server.stop()

    if len(set(paths)) != 1:
        raise AssertionError("Legacy client returned inconsistent video paths")
    images = []
    with imageio.get_reader(str(paths[0])) as reader:
        for index, image in enumerate(reader):
            if index >= len(paths):
                raise AssertionError("Video contains more frames than requested")
            if not np.any(image) or image.min() == image.max():
                raise AssertionError(
                    f"Legacy client produced a black or constant image at frame {index}"
                )
            images.append(
                {
                    "path": str(paths[index]),
                    "frame_index": index,
                    "mean": float(image.mean()),
                    "nonblack_pixels": int(np.any(image != 0, axis=-1).sum()),
                }
            )
    if len(images) != len(paths):
        raise AssertionError(f"Video contains {len(images)} frames, expected {len(paths)}")
    result = {
        "valid": True,
        "transport": "TCP simulator",
        "frames_rendered": len(images),
        "cache_entries": cache_entries,
        "clusters_received": clusters_received,
        "images": images,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    asyncio.run(validate(parser.parse_args()))


if __name__ == "__main__":
    main()
