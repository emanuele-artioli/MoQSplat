"""Budget and disk-residency behavior without requiring a GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from moq3dgs.client_runtime import LayerStore


def layer(name, subgroup, group=0):
    return SimpleNamespace(name=name, subgroup=subgroup, track="t", group=group)


def save(path):
    np.savez(
        path,
        means=np.zeros((2, 3), np.float32),
        opacities=np.zeros((2, 1), np.float32),
        scales=np.zeros((2, 3), np.float32),
        rotations=np.zeros((2, 4), np.float32),
        sh_dc=np.zeros((2, 1, 3), np.float32),
        sh_rest=np.zeros((2, 15, 3), np.float32),
    )


def pose():
    return {"camera_position": [0, 0, 0], "view_matrix": np.eye(4).tolist(), "fov": 90}


def setup_runtime(tmp_path, monkeypatch):
    store = LayerStore(100, 0)
    for name, level in [("base", 0), ("enh1", 1), ("enh2", 2)]:
        file = tmp_path / f"{name}.npz"
        save(file)
        store.add_completed(layer(name, level), file, [0, 0, -1])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(store, "_limit_allocator", lambda: None)

    def promote(entry):
        arrays = store._arrays(entry)
        size = sum(x.nbytes for x in arrays.values())
        store.gpu[entry.layer.name] = (arrays, size)
        store.gpu_bytes += size
        store.promotions += 1

    monkeypatch.setattr(store, "_promote", promote)
    monkeypatch.setattr(store, "_draw", lambda *_: torch.ones((2, 2, 3), dtype=torch.uint8))
    return store


def test_bounded_cpu_and_gpu_eviction_restore_from_disk(tmp_path, monkeypatch):
    store = setup_runtime(tmp_path, monkeypatch)
    _, metrics = store.render(pose(), 2, 2)
    assert metrics["resident_layers"] == ["base", "enh1", "enh2"]
    assert metrics["cpu_cache_bytes"] == 0
    store.entries["enh2"].size = store.gpu_budget
    _, metrics = store.render(pose(), 2, 2)
    assert metrics["resident_layers"] == ["base", "enh1"]
    assert metrics["evictions"] == 1
    store.entries["enh2"].size = store.entries["enh1"].size
    _, metrics = store.render(pose(), 2, 2)
    assert metrics["resident_layers"] == ["base", "enh1", "enh2"]
    assert metrics["promotions"] == 4


def test_oom_sheds_enhancements_but_preserves_base(tmp_path, monkeypatch):
    store = setup_runtime(tmp_path, monkeypatch)
    calls = []

    def draw(selected, *_):
        calls.append([e.layer.name for e in selected])
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return torch.ones((2, 2, 3), dtype=torch.uint8)

    monkeypatch.setattr(store, "_draw", draw)
    _, metrics = store.render(pose(), 2, 2)
    assert calls == [["base", "enh1", "enh2"], ["base", "enh1"]]
    assert metrics["resident_layers"] == ["base", "enh1"]
    assert metrics["evictions"] == 1


def test_base_budget_and_oom_fail_clearly(tmp_path, monkeypatch):
    store = setup_runtime(tmp_path, monkeypatch)
    store.entries["base"].size = store.gpu_budget
    with pytest.raises(RuntimeError, match="base layers cannot fit"):
        store.render(pose(), 2, 2)
    store.entries["base"].size = 0
    store.entries = {"base": store.entries["base"]}

    def fail(*_):
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(store, "_draw", fail)
    with pytest.raises(RuntimeError, match="OOM.*base"):
        store.render(pose(), 2, 2)


def test_missing_base_cannot_render_enhancement(tmp_path, monkeypatch):
    store = setup_runtime(tmp_path, monkeypatch)
    del store.entries["base"]
    with pytest.raises(RuntimeError, match="no complete base"):
        store.render(pose(), 2, 2)


def test_priority_prefers_visible_group_and_contiguous_levels(tmp_path, monkeypatch):
    store = setup_runtime(tmp_path, monkeypatch)
    second = tmp_path / "other.npz"
    save(second)
    store.add_completed(layer("other-base", 0, 1), second, [0, 0, 1])
    store.add_completed(layer("other-enh1", 1, 1), second, [0, 0, 1])
    # Admission can hold both bases and just one enhancement, with target at low watermark.
    for entry in store.entries.values():
        entry.size = 1024 * 1024
    store.high_watermark = 75 * 1024 * 1024
    store.low_watermark = 74 * 1024 * 1024
    names = [entry.layer.name for entry in store._select(pose())]
    assert "base" in names and "other-base" in names
    assert "enh1" in names and "other-enh1" not in names
    assert "enh2" not in names


def test_cpu_lru_eviction_and_disk_reload(tmp_path, monkeypatch):
    store = setup_runtime(tmp_path, monkeypatch)
    base = store.entries["base"]
    first = store._arrays(base)
    size = sum(x.nbytes for x in first.values())
    store.cpu_budget = size
    store._arrays(base)
    assert list(store.cpu) == ["base"]
    store._arrays(store.entries["enh1"])
    assert list(store.cpu) == ["enh1"]
    assert store.cpu_bytes == size
    first["means"][0, 0] = 123
    reloaded = store._arrays(base)
    assert reloaded["means"][0, 0] == 0
    assert list(store.cpu) == ["base"]
    assert store.cpu_bytes == size


def test_measured_peak_sheds_enhancements_and_fails_for_base(tmp_path, monkeypatch):
    store = setup_runtime(tmp_path, monkeypatch)
    peaks = iter([store.gpu_budget + 1, 1024])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: next(peaks))
    _, metrics = store.render(pose(), 2, 2)
    assert metrics["resident_layers"] == ["base", "enh1"]
    assert metrics["splat_count"] == 4
    assert metrics["cuda_peak_allocated_bytes"] == 1024
    store.entries = {"base": store.entries["base"]}
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: store.gpu_budget + 1)
    with pytest.raises(RuntimeError, match="peak exceeds GPU budget.*base"):
        store.render(pose(), 2, 2)


def test_invalid_shapes_and_budget_setter(tmp_path, monkeypatch):
    store = setup_runtime(tmp_path, monkeypatch)
    store.set_gpu_budget(120)
    assert store.gpu_budget == 120 * 1024 * 1024
    with pytest.raises(ValueError, match="positive"):
        store.set_gpu_budget(0)
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        means=np.zeros((2, 3), np.float32),
        opacities=np.zeros((1, 1), np.float32),
        scales=np.zeros((2, 3), np.float32),
        rotations=np.zeros((2, 4), np.float32),
        sh_dc=np.zeros((2, 1, 3), np.float32),
        sh_rest=np.zeros((2, 15, 3), np.float32),
    )
    store.add_completed(layer("bad", 0, 9), path, [0, 0, -1])
    with pytest.raises(ValueError, match="shapes"):
        store._arrays(store.entries["bad"])
