"""The bounded publisher should finish normally on Python 3.10 as well as 3.12."""

import asyncio
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_publisher_duration_closes_broadcast_without_timeout_error(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/run_moq_reproduction.py"
    spec = importlib.util.spec_from_file_location("native_runner_lifecycle", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    closed = []

    async def session_closed():
        await asyncio.sleep(3600)

    broadcast = SimpleNamespace(close=lambda: closed.append(True))
    client = SimpleNamespace(
        session=SimpleNamespace(closed=session_closed), create_broadcast=lambda _: broadcast
    )

    @asynccontextmanager
    async def connection(*_, **__):
        yield client

    monkeypatch.setattr(runner, "quic_client", connection)
    monkeypatch.setattr(runner, "load_scene_layers", lambda _: ([], {}))
    monkeypatch.setattr(runner, "publish_layers", lambda *_, **__: ({}, []))
    args = SimpleNamespace(
        prepared_manifest=None,
        relay="https://localhost:4433",
        insecure=True,
        reconnect_timeout=1,
        broadcast="test",
        duration=0.001,
    )
    await runner.publish(args)
    assert closed == [True]
