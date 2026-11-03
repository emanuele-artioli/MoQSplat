#!/usr/bin/env python3
"""Real-scene modern MoQ reproduction (layer tracks, not custom subgroup wire).

Run three processes: relay --bind HOST:PORT; publish --relay https://HOST:PORT
--scene REAL.ply; subscribe --relay ... --trace REAL.json --output NEW_DIRECTORY.
The native relay role uses official moq.Server, not the moq-relay executable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import time
from pathlib import Path

from moq3dgs.transport.moq_backend import (
    ORIGIN_CACHE_BYTES,
    PRIORITY_POLICY,
    VERSIONS,
    Layer,
    LayerReceiver,
    ReconnectingBroadcast,
    native_module,
    publish_layers,
    quic_client,
    quic_stats,
    viewport_priority,
)


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


async def relay(args):
    moq = native_module()
    sessions = []
    origin = moq.OriginProducer(cache_capacity_bytes=ORIGIN_CACHE_BYTES)
    async with moq.Server(
        args.bind,
        tls_generate=["localhost"],
        publish=origin,
        subscribe=origin,
    ) as server:
        print(
            json.dumps(
                {
                    "ready": True,
                    "bind": server.local_addr,
                    "implementation": "official moq.Server shared-origin relay",
                }
            ),
            flush=True,
        )

        async def accept():
            async for request in server:
                if request.transport != moq.Transport.QUIC:
                    await request.reject(403)
                    raise RuntimeError("relay rejected non-QUIC transport")
                sessions.append(await request.accept())
                print(
                    json.dumps({"accepted_transport": "QUIC", "sessions": len(sessions)}),
                    flush=True,
                )

        try:
            await asyncio.wait_for(accept(), args.duration)
        except asyncio.TimeoutError:
            pass
        finally:
            for session in sessions:
                session.shutdown()


async def publish(args):

    if args.prepared_manifest:
        layers, metadata = load_prepared(args.prepared_manifest)
    else:
        layers, metadata = load_scene_layers(args)
    async with quic_client(
        args.relay,
        tls_verify=not args.insecure,
        reconnect=True,
        reconnect_timeout=args.reconnect_timeout,
    ) as client:
        broadcast = client.create_broadcast(args.broadcast)
        manifest, handles = publish_layers(broadcast, layers, metadata=metadata)
        print(
            json.dumps({"ready": True, "manifest": manifest, "tracks_retained": len(handles)}),
            flush=True,
        )
        try:
            await asyncio.wait_for(client.session.closed(), args.duration)
        except TimeoutError:
            pass
        finally:
            broadcast.close()


def scene_arrays(scene):
    """Serialize the loader's actual attributes without tensor pickling."""
    return {
        key: getattr(scene, key).numpy().astype("<f4")
        for key in (
            "means",
            "opacities",
            "scales",
            "rotations",
            "sh_dc",
            "sh_rest",
        )
    }


def array_evidence(arrays):
    """Hash original arrays independently of their NPZ serialization."""
    return {
        name: {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
        for name, array in arrays.items()
    }


def load_prepared(manifest_path):
    """Consume original publisher DELTA PLYs; never regenerate missing inputs."""
    import numpy as np

    from moq3dgs.scene.loader import load_ply

    manifest_file = Path(manifest_path).resolve(strict=True)
    original = json.loads(manifest_file.read_text())
    root = manifest_file.parent
    clustered = original.get("clustering", {}).get("enabled", False)
    groups = original["clustering"]["clusters"] if clustered else original["objects"]
    if not groups:
        raise ValueError("prepared manifest contains no groups")
    layers, inputs, array_hashes, centroids = [], [], {}, {}
    for entry in groups:
        group_id = int(entry["cluster_id"] if clustered else entry["object_id"])
        # Original publisher reserves object -1 for background; preserve it via name.
        track_id = "scene" if group_id >= 0 else "scene-background"
        wire_group = abs(group_id)
        directory = entry["dir_relative"] if clustered else entry["dir_relative_to_outdir"]
        expected_total = int(entry["n_gaussians"] if clustered else entry["n_total"])
        if expected_total <= 0:
            raise ValueError(f"prepared group {group_id} must contain Gaussians")
        if sorted(int(item["layer"]) for item in entry["layers"]) != list(range(5)):
            raise ValueError(f"prepared group {group_id} must contain exactly layers 0..4")
        total = 0
        position_sum = np.zeros(3, dtype=np.float64)
        for item in entry["layers"]:
            source = (root / directory / item["file"]).resolve(strict=True)
            if not source.is_relative_to(root):
                raise ValueError("prepared layer path escapes manifest directory")
            scene = load_ply(source)
            if scene.num_gaussians != int(item["delta_gaussians"]):
                raise ValueError(f"delta Gaussian count mismatch: {source}")
            total += scene.num_gaussians
            original_arrays = scene_arrays(scene)
            position_sum += original_arrays["means"].sum(axis=0, dtype=np.float64)
            subgroup = int(item["layer"])
            if not 0 <= subgroup <= 4:
                raise ValueError("prepared reproduction requires five quality levels 0..4")
            buffer = io.BytesIO()
            np.savez(buffer, **original_arrays)
            payload = buffer.getvalue()
            layers.append(
                (
                    Layer.from_payload(track_id, wire_group, subgroup, payload, subgroup * 40),
                    payload,
                )
            )
            array_hashes[layers[-1][0].name] = array_evidence(original_arrays)
            inputs.append(
                {
                    "path": str(source.relative_to(root)),
                    "sha256": digest(str(source)),
                    "group_id": group_id,
                    "subgroup_id": subgroup,
                    "gaussians": scene.num_gaussians,
                }
            )
        if total != expected_total:
            raise ValueError(f"prepared group {group_id} total count {total} != {expected_total}")
        centroid = position_sum / total
        if not np.isfinite(centroid).all():
            raise ValueError(f"non-finite prepared group {group_id} centroid")
        centroids[f"{track_id}/{wire_group}"] = centroid.tolist()
    return layers, {
        "prepared_manifest_sha256": digest(str(manifest_file)),
        "grouping": "original publisher clusters" if clustered else "original publisher objects",
        "selected_splats": sum(item["gaussians"] for item in inputs),
        "source_layer_inputs": inputs,
        "attributes": "raw float32 npz containers",
        "source_array_hashes": array_hashes,
        "group_centroids": centroids,
        "priority_policy": PRIORITY_POLICY,
    }


def load_scene_layers(args):
    import numpy as np

    from moq3dgs.scene.loader import load_ply
    from moq3dgs.scene.partition import spatial_groups

    scene = load_ply(args.scene)
    count = (
        scene.num_gaussians
        if args.max_splats is None
        else min(args.max_splats, scene.num_gaussians)
    )
    if count <= 0:
        raise ValueError("scene selection is empty")
    arrays = {
        key: getattr(scene, key)[:count].numpy().astype("<f4")
        for key in (
            "means",
            "opacities",
            "scales",
            "rotations",
            "sh_dc",
            "sh_rest",
        )
    }
    score = (
        arrays["opacities"].reshape(-1)
        if args.pruning == "opacity"
        else arrays["scales"].sum(axis=1)
    )
    groups = spatial_groups(arrays["means"], args.group_splats)
    layers, array_hashes, centroids = [], {}, {}
    for group, members in enumerate(groups):
        order = members[np.argsort(-score[members], kind="stable")]
        centroids[f"scene/{group}"] = (
            arrays["means"][members].mean(axis=0, dtype=np.float64).tolist()
        )
        for index, indices in enumerate(np.array_split(order, 5)):
            buffer = io.BytesIO()
            original_arrays = {key: value[indices] for key, value in arrays.items()}
            np.savez(buffer, **original_arrays)
            payload = buffer.getvalue()
            layers.append((Layer.from_payload("scene", group, index, payload, index * 40), payload))
            array_hashes[layers[-1][0].name] = array_evidence(original_arrays)
    metadata = {
        "scene_sha256": digest(args.scene),
        "scene_splats": scene.num_gaussians,
        "selected_splats": count,
        "pruning": args.pruning,
        "grouping": "median spatial groups; original semantic grouping is separate",
        "maximum_group_splats": args.group_splats,
        "attributes": "raw float32 npz containers",
        "sh_degree": 3,
        "source_array_hashes": array_hashes,
        "group_centroids": centroids,
        "priority_policy": PRIORITY_POLICY,
    }
    return layers, metadata


async def subscribe(args):
    if args.live:
        from moq3dgs.client_player import play

        return await play(args)
    import numpy as np
    import torch
    from PIL import Image

    from moq3dgs.render.rasterizer import render_gaussians
    from moq3dgs.viewport.frustum import projection_matrix_from_fov
    from moq3dgs.viewport.trace import load_trace

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("real-scene validation requires CUDA rendering")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    trace = load_trace(args.trace)
    if not trace:
        raise ValueError("real camera trace is empty")
    if not 1 <= args.frames <= len(trace):
        raise ValueError(f"requested frames must be in 1..{len(trace)}")
    if args.camera_convention == "opencv":
        axes = np.diag([1.0, -1.0, -1.0, 1.0])
        trace = [
            {**pose, "view_matrix": (axes @ np.asarray(pose["view_matrix"])).tolist()}
            for pose in trace
        ]
    started = time.monotonic()
    async with ReconnectingBroadcast(
        args.relay,
        args.broadcast,
        tls_verify=not args.insecure,
        timeout=args.timeout,
        max_reconnects=args.reconnect_attempts,
        cache_capacity_bytes=ORIGIN_CACHE_BYTES,
    ) as connection:
        receiver = LayerReceiver(connection, timeout=args.timeout)
        manifest = await receiver.catalog()
        for centroid in manifest["metadata"]["group_centroids"].values():
            if len(centroid) != 3 or not np.isfinite(centroid).all():
                raise ValueError("catalog contains invalid group centroid")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        layers = [Layer(**item) for item in manifest["layers"]]
        tasks = {asyncio.create_task(receiver.receive(layer)): layer for layer in layers}
        decoded = {}
        priority_counts = {"changed": 0, "unchanged": 0, "pose_samples": 0}
        last_priority = {layer.name: layer.priority for layer in layers}
        progressive = []
        control_started = time.monotonic()

        async def control():
            while True:
                elapsed_ms = (time.monotonic() - control_started) * 1000
                pose = trace[0]
                for candidate in trace:
                    if candidate["timestamp_ms"] - trace[0]["timestamp_ms"] > elapsed_ms:
                        break
                    pose = candidate
                priority_counts["pose_samples"] += 1
                for layer in layers:
                    if layer.name not in receiver.active:
                        continue
                    centroid = manifest["metadata"]["group_centroids"][
                        f"{layer.track}/{layer.group}"
                    ]
                    priority = viewport_priority(
                        centroid, pose["camera_position"], pose["view_matrix"], layer.subgroup
                    )
                    if priority == last_priority[layer.name]:
                        priority_counts["unchanged"] += 1
                    elif receiver.update(layer, priority):
                        last_priority[layer.name] = priority
                        priority_counts["changed"] += 1
                await asyncio.sleep(0.1)

        def render_snapshot(selected, pose, filename, require_nonblack=True):
            # Called in a worker thread so QUIC reception/control continues on the loop.
            parts = {}
            for item in selected:
                for name, array in item.items():
                    parts.setdefault(name, []).append(array)
            tensors = {
                name: torch.from_numpy(np.concatenate(values)) for name, values in parts.items()
            }
            sh = torch.cat((tensors["sh_dc"], tensors["sh_rest"]), dim=1).reshape(
                len(tensors["means"]), -1
            )
            view = torch.tensor(pose["view_matrix"], dtype=torch.float32)
            projection = torch.from_numpy(
                projection_matrix_from_fov(
                    pose["fov"],
                    args.width / args.height,
                    fov_angles=pose.get("fov_angles"),
                )
            ).float()
            image = render_gaussians(
                tensors["means"],
                tensors["opacities"],
                sh,
                tensors["scales"],
                tensors["rotations"],
                view,
                projection,
                torch.tensor(pose["camera_position"], dtype=torch.float32),
                image_width=args.width,
                image_height=args.height,
                fov_y_deg=pose["fov"],
                fov_angles=pose.get("fov_angles"),
                device=args.device,
            ).numpy()
            if not np.isfinite(image).all() or (require_nonblack and not np.any(image)):
                raise RuntimeError(f"non-finite or black rendering: {filename}")
            Image.fromarray(image).save(output / filename)
            return {
                "file": filename,
                "splats": len(tensors["means"]),
                "nonblack_pixels": int(np.any(image != 0, axis=-1).sum()),
                "mean_pixel": float(image.mean()),
            }

        controller = asyncio.create_task(control())
        pending = set(tasks)
        try:
            while pending:
                if controller.done():
                    controller.result()
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda item: tasks[item].subgroup):
                    layer = tasks[task]
                    payload = task.result()
                    with np.load(io.BytesIO(payload), allow_pickle=False) as part:
                        arrays = {name: part[name] for name in part.files}
                    expected = manifest["metadata"]["source_array_hashes"][layer.name]
                    if array_evidence(arrays) != expected:
                        raise ValueError(
                            f"decoded arrays differ from original scene arrays: {layer.name}"
                        )
                    if not all(np.isfinite(array).all() for array in arrays.values()):
                        raise ValueError(f"non-finite original attributes: {layer.name}")
                    decoded[layer.name] = arrays
                    # Protect layer dependencies: render only contiguous base-first prefixes.
                    eligible = [
                        decoded[item.name]
                        for item in layers
                        if item.name in decoded
                        and all(
                            any(
                                lower.track == item.track
                                and lower.group == item.group
                                and lower.subgroup == level
                                and lower.name in decoded
                                for lower in layers
                            )
                            for level in range(item.subgroup + 1)
                        )
                    ]
                    if eligible and (not progressive or pending) and len(progressive) < 5:
                        in_flight = sum(not item.done() for item in tasks)
                        report = await asyncio.to_thread(
                            render_snapshot,
                            eligible,
                            trace[0],
                            f"progressive-{len(progressive):05d}.png",
                            False,
                        )
                        report.update(
                            {
                                "transfers_pending_at_start": in_flight,
                                "seconds_since_connect": time.monotonic() - started,
                                "completed_layers": len(decoded),
                            }
                        )
                        progressive.append(report)
        finally:
            control_failure = controller.exception() if controller.done() else None
            controller.cancel()
            await asyncio.gather(controller, return_exceptions=True)
            await receiver.close()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if control_failure is not None:
                raise control_failure
        payloads = [task.result() for task in tasks]
        if (
            sum(len(part["means"]) for part in decoded.values())
            != manifest["metadata"]["selected_splats"]
        ):
            raise ValueError("received Gaussian count differs from original selected scene")
        initial_requests = receiver.requests
        for layer in layers:
            await receiver.receive(layer)
        stats = quic_stats(connection.client)
        frames = 0
        final_images = []
        for index, frame in enumerate(trace[: args.frames]):
            final_images.append(
                await asyncio.to_thread(
                    render_snapshot, list(decoded.values()), frame, f"frame-{index:05d}.png"
                )
            )
            frames += 1
        summary = {
            "valid": frames > 0,
            "frames_rendered": frames + len(progressive),
            "device_type": "cuda",
            "transport": "QUIC",
            "versions": VERSIONS,
            "wire_roundtrip_exact": True,
            "original_arrays_exact": True,
            "progressive_images": progressive,
            "final_images": final_images,
            "progressive_rendering_overlapped_transfer": any(
                item["transfers_pending_at_start"] > 0 and item["nonblack_pixels"] > 0
                for item in progressive
            ),
            "memory_policy": (
                "verified completed payloads retained in RAM; GPU watermark swapping absent"
            ),
            "mapping": "modern MoQ layer tracks; not recovered custom subgroup protocol",
            "relay_validation": "external URL; inspect relay transport log separately",
            "trace_sha256": digest(args.trace),
            "input_camera_convention": args.camera_convention,
            "scene": manifest["metadata"],
            "bytes_received_payload": sum(map(len, payloads)),
            "layers_received": len(layers),
            "payload_sha256": [item.sha256 for item in layers],
            "cache_revisit_requests": receiver.requests - initial_requests,
            "reconnects": connection.reconnects,
            "priority_updates": priority_counts,
            "quic_stats": stats,
            "wall_seconds": time.monotonic() - started,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("relay", "publish", "subscribe"))
    parser.add_argument("--relay", default="https://127.0.0.1:4433")
    parser.add_argument("--broadcast", default="moqsplat/scene")
    parser.add_argument("--bind", default="127.0.0.1:4433")
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--insecure", action="store_true", help="explicit local self-signed TLS")
    parser.add_argument("--scene")
    parser.add_argument(
        "--prepared-manifest", help="original publisher manifest with existing DELTA PLYs"
    )
    parser.add_argument("--trace")
    parser.add_argument("--output")
    parser.add_argument(
        "--max-splats", type=int, help="explicit smoke subset; default entire scene"
    )
    parser.add_argument("--pruning", choices=("opacity", "scale"), default="opacity")
    parser.add_argument(
        "--group-splats",
        type=int,
        default=65536,
        help="maximum splats per spatial group before five quality layers",
    )
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--live", action="store_true", help="continuous received-data rendering")
    parser.add_argument("--viewer-port", type=int, help="local browser controls; 0 selects a port")
    parser.add_argument("--cache", help="verified disk cache; reusable between live sessions")
    parser.add_argument("--gpu-memory-mib", type=int, default=2048)
    parser.add_argument("--cpu-cache-mib", type=int, default=128)
    parser.add_argument("--native-cache-mib", type=int, default=128)
    parser.add_argument("--parallel-downloads", type=int, default=4)
    parser.add_argument(
        "--max-scene-gib",
        type=float,
        default=4,
        help="maximum complete cached payload size, not a RAM allocation",
    )
    parser.add_argument(
        "--fps", type=float, default=20, help="target rate, not a performance promise"
    )
    parser.add_argument(
        "--live-frames", type=int, default=0, help="optional continuous frame limit"
    )
    parser.add_argument("--reconnect-attempts", type=int, default=3)
    parser.add_argument("--reconnect-timeout", type=float, default=30)
    parser.add_argument("--camera-convention", choices=("opengl", "opencv"), default="opengl")
    args = parser.parse_args()
    if not all(
        math.isfinite(v) and v > 0
        for v in (args.duration, args.timeout, args.fps, args.reconnect_timeout, args.max_scene_gib)
    ):
        parser.error("duration, timeout, fps and cache limits must be finite and positive")
    if (
        args.parallel_downloads <= 0
        or args.native_cache_mib <= 0
        or args.gpu_memory_mib <= 0
        or args.cpu_cache_mib < 0
    ):
        parser.error("positive transfer/GPU limits and nonnegative CPU cache required")
    if args.frames <= 0 or args.width <= 0 or args.height <= 0 or args.duration <= 0:
        parser.error("frames, image dimensions and duration must be positive")
    if (
        args.fps <= 0
        or args.live_frames < 0
        or args.reconnect_attempts < 0
        or args.group_splats < 5
    ):
        parser.error("positive fps and nonnegative live frames/reconnect attempts required")
    if args.role == "publish" and bool(args.scene) == bool(args.prepared_manifest):
        parser.error("publish requires exactly one of --scene or --prepared-manifest")
    if args.prepared_manifest and args.max_splats is not None:
        parser.error("--max-splats cannot truncate original prepared layers")
    if args.role == "subscribe" and (not args.trace or not args.output):
        parser.error("subscribe requires explicit --trace and --output")
    asyncio.run({"relay": relay, "publish": publish, "subscribe": subscribe}[args.role](args))


if __name__ == "__main__":
    main()
