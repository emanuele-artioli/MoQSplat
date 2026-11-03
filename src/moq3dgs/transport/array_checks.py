"""Source-attribute verification usable by a CPU-only network test client."""

import hashlib

import numpy as np


def verify_arrays(path, expected):
    """Check decoded source attributes independently of any renderer."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(expected):
            raise ValueError("received attributes differ from source catalog")
        for name in data.files:
            array = data[name]
            contiguous = np.ascontiguousarray(array)
            evidence = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(memoryview(contiguous)).hexdigest(),
            }
            if evidence != expected[name] or not np.isfinite(array).all():
                raise ValueError(f"received source attribute mismatch: {name}")


def missing_cache_bytes(layers, directory):
    """Estimate new disk space without charging existing same-size content twice.

    This is a size precheck only. LayerReceiver still verifies every cached
    file's SHA256 before use; replacing a corrupt file can need extra space.
    """
    required = 0
    for layer in layers:
        path = directory / f"{layer.sha256}.bin"
        if not path.is_file() or path.stat().st_size != layer.size:
            required += layer.size
    return required
