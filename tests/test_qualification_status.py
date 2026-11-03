"""Prevent startup statuses from firing expensive memory qualification too early."""

import importlib.util
from pathlib import Path


def test_all_received_requires_every_client_to_have_a_nonempty_catalog():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_client_experience.py"
    spec = importlib.util.spec_from_file_location("client_qualification_status", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check = module.all_clients_received
    complete = {"received_layers": 640, "total_layers": 640}
    assert not check([complete, {"state": "waiting for base layer"}], 2)
    assert not check([complete], 2)
    assert not check([{"received_layers": 0, "total_layers": 0}], 1)
    assert not check([complete, {"received_layers": 639, "total_layers": 640}], 2)
    assert check([complete, complete], 2)
