"""Warm caches should not require a second complete scene's free disk space."""

from types import SimpleNamespace

from moq3dgs.transport.array_checks import missing_cache_bytes


def test_warm_and_partial_disk_cache_account_only_missing_files(tmp_path):
    layers = [SimpleNamespace(sha256="a", size=5), SimpleNamespace(sha256="b", size=7)]
    assert missing_cache_bytes(layers, tmp_path) == 12
    (tmp_path / "a.bin").write_bytes(b"12345")
    assert missing_cache_bytes(layers, tmp_path) == 7
    (tmp_path / "b.bin").write_bytes(b"1234567")
    assert missing_cache_bytes(layers, tmp_path) == 0
    (tmp_path / "a.bin").write_bytes(b"short")
    # Same-size content is verified by LayerReceiver, not this disk-size precheck.
    assert missing_cache_bytes(layers, tmp_path) == 0
    (tmp_path / "a.bin").write_bytes(b"x")
    assert missing_cache_bytes(layers, tmp_path) == 5
