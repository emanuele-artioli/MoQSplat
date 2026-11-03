"""Prepared artifacts must be complete; the bridge never invents missing layers."""

import json
import runpy
from pathlib import Path

import pytest


def test_prepared_layers_require_five_unique_ids_and_existing_files(tmp_path):
    load = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_moq_reproduction.py"))[
        "load_prepared"
    ]
    layers = [{"layer": i, "file": f"layer{i}.ply", "delta_gaussians": 1} for i in range(5)]
    manifest = {
        "clustering": {
            "enabled": True,
            "clusters": [
                {
                    "cluster_id": 0,
                    "n_gaussians": 5,
                    "dir_relative": "layers",
                    "layers": layers,
                }
            ],
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError):
        load(path)
    layers[-1]["layer"] = 3
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly layers 0..4"):
        load(path)
