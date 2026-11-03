"""Continuously render verified received layers while QUIC and controls progress."""

from __future__ import annotations

import asyncio
import hashlib
import json
import resource
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from moq3dgs.client_runtime import LayerStore
from moq3dgs.transport.array_checks import missing_cache_bytes, verify_arrays
from moq3dgs.transport.moq_backend import (
    VERSIONS,
    Layer,
    LayerReceiver,
    ReconnectingBroadcast,
    quic_stats,
    viewport_priority,
)
from moq3dgs.viewer import FrameHub, PoseController, start_viewer
from moq3dgs.viewport.trace import load_trace


async def play(args):
    """One bounded session; browser controls stay local to the receiving machine."""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = Path(args.cache).resolve() if args.cache else output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    trace = load_trace(args.trace)
    if args.camera_convention == "opencv":
        axes = np.diag([1.0, -1.0, -1.0, 1.0])
        trace = [
            {**p, "view_matrix": (axes @ np.asarray(p["view_matrix"])).tolist()} for p in trace
        ]
    controller = PoseController(trace)
    hub = FrameHub(controller)
    server = start_viewer(hub, args.viewer_port) if args.viewer_port is not None else None
    store = LayerStore(args.gpu_memory_mib, args.cpu_cache_mib, args.device)
    renderer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moqsplat-cuda")
    event_loop = asyncio.get_running_loop()
    lock = asyncio.Lock()
    started = time.monotonic()
    frames = 0
    nonblack_frames = 0
    received = set()
    priority_updates = 0
    peak = 0
    frame_times = []
    input_latencies = []
    rendered_input_events = 0
    first_nonblack_seconds = None
    last_metrics = {}
    reader = None
    downloads = []
    control = None
    result = {"valid": False}
    failure = None
    identity = None
    manifest = None
    connection = ReconnectingBroadcast(
        args.relay,
        args.broadcast,
        tls_verify=not args.insecure,
        timeout=args.timeout,
        max_reconnects=args.reconnect_attempts,
        cache_capacity_bytes=args.native_cache_mib * 1024 * 1024,
    )
    try:
        async with connection:
            reader = LayerReceiver(
                connection,
                timeout=args.timeout,
                max_total_payload=int(args.max_scene_gib * 1024**3),
                max_active_files=args.parallel_downloads,
            )
            manifest = await reader.catalog()
            identity = connection.identity
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            layers = [Layer(**item) for item in manifest["layers"]]
            required_disk_bytes = await asyncio.to_thread(missing_cache_bytes, layers, cache)
            if shutil.disk_usage(cache).free < required_disk_bytes:
                raise RuntimeError("insufficient free space for missing scene cache files")
            centroids = manifest["metadata"]["group_centroids"]
            expected = manifest["metadata"]["source_array_hashes"]

            async def fetch(layer):
                path = await reader.receive_file(layer, cache)
                await asyncio.to_thread(verify_arrays, path, expected[layer.name])
                async with lock:
                    await asyncio.to_thread(
                        store.add_completed, layer, path, centroids[f"{layer.track}/{layer.group}"]
                    )
                    received.add(layer.name)

            queued = list(layers)

            async def download_worker():
                while queued:
                    pose = controller.sample((time.monotonic() - started) * 1000)
                    layer = min(
                        queued,
                        key=lambda item: viewport_priority(
                            centroids[f"{item.track}/{item.group}"],
                            pose["camera_position"],
                            pose["view_matrix"],
                            item.subgroup,
                        ),
                    )
                    queued.remove(layer)
                    await fetch(layer)

            downloads = [
                asyncio.create_task(download_worker()) for _ in range(args.parallel_downloads)
            ]
            latest_priority = {layer.name: layer.priority for layer in layers}

            async def update():
                nonlocal priority_updates
                while True:
                    pose = controller.sample((time.monotonic() - started) * 1000)
                    for layer in layers:
                        if layer.name not in reader.active:
                            continue
                        priority = viewport_priority(
                            centroids[f"{layer.track}/{layer.group}"],
                            pose["camera_position"],
                            pose["view_matrix"],
                            layer.subgroup,
                        )
                        if priority != latest_priority[layer.name] and reader.update(
                            layer, priority
                        ):
                            latest_priority[layer.name] = priority
                            priority_updates += 1
                    await asyncio.sleep(0.1)

            control = asyncio.create_task(update())
            print(
                json.dumps(
                    {
                        "ready": True,
                        "viewer_url": (
                            f"http://127.0.0.1:{server.server_port}" if server else None
                        ),
                        "mode": "continuous received-data CUDA playback",
                    }
                ),
                flush=True,
            )
            with (output / "frames.jsonl").open("w") as telemetry:
                while time.monotonic() - started < args.duration:
                    tick = time.monotonic()
                    for task in downloads:
                        if task.done():
                            task.result()
                    if control.done():
                        control.result()
                    pose = controller.sample((tick - started) * 1000)
                    with hub.condition:
                        budget, hub.pending_budget = hub.pending_budget, None
                    async with lock:
                        if budget is not None:
                            await event_loop.run_in_executor(renderer, store.set_gpu_budget, budget)
                        if not any(e.layer.subgroup == 0 for e in store.entries.values()):
                            image = None
                        else:
                            image, metrics = await event_loop.run_in_executor(
                                renderer, store.render, pose, args.width, args.height
                            )
                    if image is not None:
                        nonblack_frames += bool(np.any(image))
                        now = time.monotonic()
                        frames += 1
                        peak = max(peak, metrics["cuda_attempt_peak_allocated_bytes"])
                        if first_nonblack_seconds is None and np.any(image):
                            first_nonblack_seconds = now - started
                        pose_events = pose.get("_input_events", 0)
                        input_latency = None
                        if (
                            pose_events > rendered_input_events
                            and pose.get("_last_input") is not None
                        ):
                            input_latency = (now - pose["_last_input"]) * 1000
                            input_latencies.append(input_latency)
                            rendered_input_events = pose_events
                        frame_times.append(now)
                        metrics.update(
                            seconds=now - started,
                            received_layers=len(received),
                            total_layers=len(layers),
                            reconnects=connection.reconnects,
                            priority_updates=priority_updates,
                            input_events=pose_events,
                            input_to_render_ms=input_latency,
                            camera_position=pose["camera_position"],
                            nonblack_pixels=int(np.any(image != 0, axis=-1).sum()),
                            frame_ms=(now - tick) * 1000,
                            process_peak_rss_bytes=(
                                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                * (1 if sys.platform == "darwin" else 1024)
                            ),
                        )
                        telemetry.write(json.dumps(metrics) + "\n")
                        telemetry.flush()
                        await asyncio.to_thread(hub.publish, image, metrics)
                        if frames <= 3:
                            Image.fromarray(image).save(output / f"frame-{frames:05d}.png")
                        last_metrics = metrics
                        if args.live_frames and frames >= args.live_frames:
                            break
                    await asyncio.sleep(max(0.001, 1 / args.fps - (time.monotonic() - tick)))
            if frames:
                Image.fromarray(image).save(output / "last-frame.png")
            revisit_requests = None
            if len(received) == len(layers):
                before = reader.requests
                await asyncio.gather(*(reader.receive_file(layer, cache) for layer in layers))
                revisit_requests = reader.requests - before
            try:
                stats = quic_stats(connection.client)
            except Exception as exc:
                stats = {"unavailable_after_disconnect": str(exc)}
            result = {
                "valid": frames > 0,
                "device_type": "cuda",
                "transport": "QUIC",
                "mode": "continuous",
                "frames_rendered": frames,
                "nonblack_frames": nonblack_frames,
                "versions": VERSIONS,
                "torch_version": str(torch.__version__),
                "gpu": {
                    "name": torch.cuda.get_device_name(args.device),
                    "capability": list(torch.cuda.get_device_capability(args.device)),
                    "uuid": str(torch.cuda.get_device_properties(args.device).uuid),
                },
                "trace_sha256": hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),
                "layers_received": len(received),
                "layers_expected": len(layers),
                "all_layers_received": len(received) == len(layers),
                "payload_requests": reader.requests,
                "cache_revisit_requests": revisit_requests,
                "original_arrays_exact": True,
                "wire_roundtrip_exact": True,
                "catalog_identity": identity,
                "reconnects": connection.reconnects,
                "priority_updates": priority_updates,
                "input_events": controller.input_events,
                "cuda_peak_allocated_bytes": peak,
                "last_frame": last_metrics,
                "first_frame_seconds": frame_times[0] - started if frame_times else None,
                "first_nonblack_frame_seconds": first_nonblack_seconds,
                "max_frame_interval_ms": (
                    max(b - a for a, b in zip(frame_times, frame_times[1:])) * 1000
                    if len(frame_times) > 1
                    else None
                ),
                "input_to_render_ms": input_latencies,
                "input_latency_scope": (
                    "server control receipt to completed render; excludes display"
                ),
                "rendered_interval_fps": (
                    (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                    if len(frame_times) > 1
                    else 0
                ),
                "quic_stats": stats,
                "wall_seconds": time.monotonic() - started,
                "cache_directory": str(cache),
                "cache_policy": "verified disk layers; bounded residency",
                "native_cache_target_bytes": args.native_cache_mib * 1024 * 1024,
                "parallel_downloads": args.parallel_downloads,
            }
            if not result["valid"]:
                raise RuntimeError("session ended without a complete renderable base layer")
    except BaseException as exc:
        failure = exc
        result.update(
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
            frames_rendered=frames,
            last_frame=last_metrics,
            reconnects=connection.reconnects,
            layers_received=len(received),
            cuda_peak_allocated_bytes=peak,
        )
    finally:
        if control:
            control.cancel()
        for task in downloads:
            task.cancel()
        await asyncio.gather(*downloads, *([control] if control else []), return_exceptions=True)
        if reader:
            await reader.close()
        await asyncio.to_thread(renderer.shutdown, wait=True)
        with hub.condition:
            hub.stopped = True
            hub.condition.notify_all()
        if server:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
        (output / "summary.json").write_text(json.dumps(result, indent=2))
    if failure:
        raise failure
    print(json.dumps(result), flush=True)
