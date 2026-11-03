#!/usr/bin/env python3
"""Bounded real-scene qualification of preparation, playback and network recovery."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from moq3dgs.transport.udp_proxy import UDPProxy

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return stream.getsockname()[1]


def request(url, payload=None):
    if payload is not None:
        url = urllib.request.Request(
            url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
        )
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def command(argv, log, timeout, env=None):
    with log.open("w") as output:
        try:
            subprocess.run(
                argv,
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            output.flush()
            raise RuntimeError(f"{exc}: {log.read_text()[-4000:]}") from exc


class Children:
    def __init__(self, output):
        self.output = output
        self.processes = []
        self.commands = []

    def start(self, name, argv):
        path = self.output / f"{name}.log"
        with path.open("w") as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(process)
        self.commands.append({"name": name, "argv": argv})
        return process, path

    async def ready(self, child, seconds=120):
        process, path = child
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"child exited {process.returncode}: {path.read_text()[-2000:]}")
            if '"ready": true' in path.read_text():
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"child not ready: {path}")

    async def stop_one(self, process):
        if process.poll() is None:
            process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), 10)
            except asyncio.TimeoutError:
                process.kill()
                await asyncio.to_thread(process.wait)

    async def close(self):
        for process in reversed(self.processes):
            await self.stop_one(process)
        (self.output / "commands.json").write_text(json.dumps(self.commands, indent=2))


async def semantic(args, output, children, relay):
    """Use the released publisher/VMAF path, then render its actually delivered layers."""
    config = output / "publisher-config.json"
    config.write_text(
        json.dumps(
            [
                "--views",
                "4",
                "--lateral_offsets",
                "0",
                "--fov_scale",
                ".4",
                "--yolo_conf",
                ".55",
                "--mask_pad",
                "2",
                "--depth_band_frac",
                "1",
                "--no_dilate",
                "--layer_percentages",
                "20,40,60,80,100",
                "--layer_rank",
                "opacity",
                "--clustering",
                "--no_object_previews",
                "--no_layer_viz",
                "--no_combined_figure",
                "--no_layer_full_ply",
            ]
        )
    )
    work = output / "semantic"
    work.mkdir()
    prepared = work / "bicycle_opacity" / "seg"
    prepared.parent.mkdir()
    argv = [
        sys.executable,
        str(ROOT / "scripts/run_publisher_reproduction.py"),
        "--scene",
        args.scene,
        "--camera-json",
        args.publisher_camera,
        "--weights",
        args.weights,
        "--config",
        str(config),
        "--output",
        str(prepared),
        "--cache",
        str(output / "publisher-cache"),
        "--device",
        "cuda:0",
    ]
    await asyncio.to_thread(command, argv, output / "publisher.log", 420)
    manifest = json.loads((prepared / "manifest.json").read_text())
    if not any(item["object_id"] >= 0 for item in manifest["objects"]):
        raise ValueError("semantic publisher detected no real objects")
    environment = dict(
        os.environ,
        BICYCLE_PLY=args.scene,
        BICYCLE_TRAJ=args.publisher_camera,
        WORK_DIR=str(work),
        FFMPEG_BIN=args.ffmpeg,
        SEG_SCRIPT=str(ROOT / "publisher/gsplat_clustering_vmaf.py"),
        RENDER_SCRIPT=str(ROOT / "publisher/render_layer_videos.py"),
        ORBIT_SCRIPT=str(ROOT / "publisher/make_cluster_orbit_trajectory.py"),
    )
    argv = [
        "bash",
        str(ROOT / "publisher/compute_rd_vmaf.sh"),
        "--skip-seg",
        "--scene",
        "bicycle",
        "--rank",
        "opacity",
        "--duration-s",
        "1",
        "--fps",
        "4",
        "--device",
        "cuda:0",
    ]
    await asyncio.to_thread(command, argv, output / "publisher-vmaf.log", 300, environment)
    curves = list(work.rglob("rd_curve_bicycle_opacity.csv"))
    if len(curves) != 1:
        raise ValueError("semantic VMAF pipeline did not produce its curve")
    with curves[0].open() as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 5:
        raise ValueError("VMAF curve must contain five levels")
    runner = [sys.executable, str(ROOT / "scripts/run_moq_reproduction.py")]
    publisher = children.start(
        "semantic-publisher",
        runner
        + [
            "publish",
            "--relay",
            relay,
            "--insecure",
            "--prepared-manifest",
            str(prepared / "manifest.json"),
            "--broadcast",
            "qualification/semantic",
            "--duration",
            "900",
        ],
    )
    await children.ready(publisher, 120)
    await asyncio.to_thread(
        command,
        runner
        + [
            "subscribe",
            "--relay",
            relay,
            "--insecure",
            "--broadcast",
            "qualification/semantic",
            "--trace",
            args.trace,
            "--frames",
            "1",
            "--width",
            "320",
            "--height",
            "240",
            "--device",
            "cuda:0",
            "--output",
            str(output / "semantic-received"),
            "--timeout",
            "120",
        ],
        output / "semantic-subscriber.log",
        180,
    )
    await children.stop_one(publisher[0])
    received = json.loads((output / "semantic-received/summary.json").read_text())
    if not received["valid"] or not received["original_arrays_exact"]:
        raise ValueError("prepared publisher output failed native received-data rendering")
    return {
        "valid": True,
        "objects": len(manifest["objects"]),
        "vmaf_levels": len(rows),
        "received": received,
    }


async def experience(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    children = Children(output)
    runner = [sys.executable, str(ROOT / "scripts/run_moq_reproduction.py")]
    port = free_port()
    relay_url = f"https://127.0.0.1:{port}"
    relay_command = runner + ["relay", "--bind", f"127.0.0.1:{port}", "--duration", "1800"]
    result = {"valid": False, "cases": [], "stage": args.stage}
    try:
        relay = children.start("relay", relay_command)
        await children.ready(relay)
        if args.stage == "smoke":
            await asyncio.to_thread(
                command,
                [
                    sys.executable,
                    str(ROOT / "scripts/validate_reproduction.py"),
                    "--scene",
                    args.scene,
                    "--trace",
                    args.trace,
                    "--frames",
                    "1",
                    "--width",
                    "320",
                    "--height",
                    "240",
                    "--camera-convention",
                    "opengl",
                    "--device",
                    "cuda:0",
                    "--output",
                    str(output / "oracle"),
                ],
                output / "oracle.log",
                180,
            )
            result["semantic"] = await semantic(args, output, children, relay_url)
        publisher = children.start(
            "publisher",
            runner
            + [
                "publish",
                "--relay",
                relay_url,
                "--insecure",
                "--scene",
                args.scene,
                "--broadcast",
                "qualification/live",
                "--group-splats",
                "65536",
                "--duration",
                "1800",
            ],
        )
        await children.ready(publisher)
        profiles = [
            ("direct", 0, 0, 0, 45 if args.stage == "smoke" else 60),
            ("constrained", 120, 80, 0.001, 40 if args.stage == "smoke" else 90),
            ("recovery-soak", 200, 10, 0.001, args.soak_seconds),
        ]
        for label, mbps, delay, loss, duration in profiles:
            proxy = await UDPProxy.start(
                ("127.0.0.1", port), mbps=mbps, delay_ms=delay, loss=loss, seed=19
            )
            clients = []
            try:
                count = 1 if args.stage == "smoke" else 2
                for number in range(count):
                    viewer = free_port()
                    folder = output / f"{label}-{number}"
                    argv = runner + [
                        "subscribe",
                        "--relay",
                        f"https://127.0.0.1:{proxy.address[1]}",
                        "--insecure",
                        "--broadcast",
                        "qualification/live",
                        "--trace",
                        args.trace,
                        "--device",
                        "cuda:0",
                        "--live",
                        "--viewer-port",
                        str(viewer),
                        "--duration",
                        str(duration),
                        "--fps",
                        "12",
                        "--width",
                        "320",
                        "--height",
                        "240",
                        "--gpu-memory-mib",
                        "6144",
                        "--cpu-cache-mib",
                        "128",
                        "--native-cache-mib",
                        "128",
                        "--output",
                        str(folder),
                        "--timeout",
                        "60",
                    ]
                    child = children.start(f"{label}-client-{number}", argv)
                    await children.ready(child)
                    clients.append((child, folder, f"http://127.0.0.1:{viewer}"))
                first_frame = False
                restart = False
                lowered = False
                raised = False
                returned_to_trace = False
                lowered_at = 0
                started = time.monotonic()

                async def post_clients(path, payload):
                    for child, _, viewer in clients:
                        if child[0].poll() is None:
                            try:
                                await asyncio.to_thread(request, viewer + path, payload)
                            except OSError:
                                if child[0].poll() is None:
                                    raise

                while any(item[0][0].poll() is None for item in clients):
                    elapsed = time.monotonic() - started
                    if elapsed > duration + 120:
                        raise TimeoutError(f"client case exceeded bound: {label}")
                    for child, _, viewer in clients:
                        if child[0].poll() is not None:
                            continue
                        try:
                            status = await asyncio.to_thread(request, viewer + "/status")
                        except OSError:
                            continue
                        if status.get("nonblack_pixels", 0):
                            first_frame = True
                    if (
                        first_frame
                        and not lowered
                        and (
                            label != "direct"
                            or status.get("received_layers") == status.get("total_layers")
                            or elapsed > duration / 2
                        )
                    ):
                        await post_clients("/input", {"yaw": 0.08, "right": 0.05})
                        await post_clients("/budget", {"mib": 1536})
                        lowered = True
                        lowered_at = elapsed
                    if lowered and elapsed > lowered_at + 1 and not returned_to_trace:
                        await post_clients("/input", {"trace": True})
                        returned_to_trace = True
                    if lowered and elapsed > lowered_at + 3 and not raised:
                        await post_clients("/budget", {"mib": 6144})
                        raised = True
                    if label == "recovery-soak" and first_frame and not restart:
                        await children.stop_one(relay[0])
                        await asyncio.sleep(0.3)
                        relay = children.start("relay-restarted", relay_command)
                        await children.ready(relay)
                        restart = True
                    await asyncio.sleep(0.2)
                summaries = []
                for child, folder, _ in clients:
                    if child[0].returncode:
                        raise RuntimeError(f"client failed: {child[1]}")
                    summary = json.loads((folder / "summary.json").read_text())
                    if (
                        not summary["valid"]
                        or summary["frames_rendered"] < 3
                        or summary["nonblack_frames"] < 1
                        or summary["input_events"] < 1
                    ):
                        raise ValueError(
                            f"continuous client did not satisfy controls/rendering: {label}"
                        )
                    if summary["cache_revisit_requests"] not in (None, 0):
                        raise ValueError("cache revisit repeated payload requests")
                    if label == "direct" and not summary["all_layers_received"]:
                        raise ValueError("unrestricted client did not finish the real scene")
                    if label == "direct" and summary["last_frame"]["evictions"] < 1:
                        raise ValueError("direct client did not exercise real GPU layer eviction")
                    summaries.append(summary)
                if label == "recovery-soak" and not any(s["reconnects"] for s in summaries):
                    raise ValueError("relay restart did not exercise a recovered subscriber")
                result["cases"].append(
                    {
                        "name": label,
                        "configured_mbps": mbps,
                        "one_way_delay_ms": delay,
                        "loss_probability": loss,
                        "proxy_counters": proxy.counters,
                        "clients": summaries,
                    }
                )
                (output / "summary.json").write_text(json.dumps(result, indent=2))
            finally:
                await proxy.close()
        result.update(
            valid=True,
            device_type="cuda",
            transport="QUIC",
            wire_roundtrip_exact=True,
            frames_rendered=sum(
                s["frames_rendered"] for c in result["cases"] for s in c["clients"]
            ),
        )
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        await children.close()
        (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"valid": result["valid"], "cases": len(result["cases"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "trace", "output", "publisher-camera", "weights"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--ffmpeg", default="/opt/local/bin/ffmpeg")
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--soak-seconds", type=int, required=True)
    asyncio.run(experience(parser.parse_args()))


if __name__ == "__main__":
    main()
