#!/usr/bin/env python3
"""Run the original publisher with explicit, locally available inputs.

The output directory must be new and outside the checkout. Configuration is a
JSON array of additional publisher arguments; protected arguments cannot be
replaced. This wrapper does not download model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / "publisher" / "gsplat_clustering_vmaf.py"
RESERVED = {"--input", "--camera_json", "--yolo_model", "--outdir", "--device"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_input(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    return path


def outside_checkout(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    path = path.resolve()
    if path == ROOT or path.is_relative_to(ROOT):
        raise ValueError(f"{label} must be outside the checkout: {path}")
    return path


def write_record(output: Path, record: dict) -> None:
    temp = output / "run.json.tmp"
    temp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temp, output / "run.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    scene = file_input(args.scene, "Scene")
    camera = file_input(args.camera_json, "Camera JSON")
    weights = file_input(args.weights, "YOLO weights")
    config = file_input(args.config, "Config")
    publisher = file_input(PUBLISHER, "Publisher script")
    output = outside_checkout(args.output, "Output")
    cache = outside_checkout(args.cache, "Cache")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}")
    if cache == output or cache.is_relative_to(output) or output.is_relative_to(cache):
        raise ValueError("Output and cache must be separate directories")
    tokens = json.loads(config.read_text())
    if not isinstance(tokens, list) or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError("Config must be a JSON list of nonempty strings")
    for token in tokens:
        option = token.split("=", 1)[0]
        if option.startswith("--") and any(flag.startswith(option) for flag in RESERVED):
            raise ValueError(f"Config cannot override reserved argument: {token}")

    argv = [
        sys.executable,
        str(publisher),
        "--input",
        str(scene),
        "--camera_json",
        str(camera),
        "--yolo_model",
        str(weights),
        "--outdir",
        str(output),
        "--device",
        args.device,
        *tokens,
    ]
    env = os.environ.copy()
    cache_dirs = {
        "TMPDIR": "tmp",
        "TMP": "tmp",
        "TEMP": "tmp",
        "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "PYTHONPYCACHEPREFIX": "pycache",
        "XDG_CACHE_HOME": "xdg",
        "YOLO_CONFIG_DIR": "yolo",
        "MPLCONFIGDIR": "matplotlib",
    }
    for name, suffix in cache_dirs.items():
        directory = cache / suffix
        directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
    output.mkdir(parents=True, exist_ok=False)
    files = {
        "scene": scene,
        "camera_json": camera,
        "weights": weights,
        "config": config,
        "publisher_script": publisher,
    }
    record = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {k: {"path": str(v), "sha256": digest(v)} for k, v in files.items()},
        "argv": argv,
        "environment": {key: env[key] for key in cache_dirs},
        "packages": sorted(
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        ),
    }
    write_record(output, record)
    try:
        with (output / "run.log").open("xb") as log:
            record["status"] = "running"
            write_record(output, record)
            proc = subprocess.run(
                argv, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        record["returncode"] = proc.returncode
        if proc.returncode != 0:
            raise RuntimeError(
                f"Publisher failed with exit code {proc.returncode}; see {output / 'run.log'}"
            )
        if not (output / "manifest.json").is_file():
            raise RuntimeError("Publisher exited successfully without output/manifest.json")
        record["status"] = "complete"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_record(output, record)


if __name__ == "__main__":
    main()
