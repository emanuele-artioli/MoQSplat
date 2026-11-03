# MoQSplat

Viewport-aware delivery of pretrained 3D Gaussian Splatting scenes over
Media-over-QUIC (MoQ). The native Python interface uses pinned Rust MoQ/QUIC
libraries; the renderer uses CUDA gsplat.

Start with [REPRODUCTION.md](REPRODUCTION.md) for setup, input provenance and
commands. [CLIENT.md](CLIENT.md) covers the live client, memory limits, recovery
and an independent-network test recipe. The original semantic publisher and
five-level VMAF pipeline are in [publisher/](publisher/README.md).

The modern native path is an independent functional reproduction. The paper's
customized Rust subgroup implementation and exact end-to-end experiment
artifacts have not been recovered. It is not an exact Table-I reproduction.

## Native live client

Install into an isolated environment with a compatible Torch/CUDA toolchain:

```sh
python -m pip install -e '.[dev,reproduce,publisher]'
python -m pip check
python -m pytest tests -q
```

In separate terminals, with an actual trained scene and corrected camera trace:

```sh
python scripts/run_moq_reproduction.py relay --bind 127.0.0.1:4433 --duration 900

python scripts/run_moq_reproduction.py publish \
  --relay https://127.0.0.1:4433 --insecure \
  --scene /absolute/path/bicycle.ply --duration 900

python scripts/run_moq_reproduction.py subscribe \
  --relay https://127.0.0.1:4433 --insecure \
  --trace /absolute/path/corrected-bicycle-trace.json --device cuda:0 \
  --live --viewer-port 8765 --gpu-memory-mib 2048 \
  --cache /absolute/path/reusable-layer-cache \
  --output /absolute/path/new-client-run --duration 300
```

Open `http://127.0.0.1:8765`. Drag to look, use W/A/S/D and Q/E to move,
and return to the recorded trace with the button. Rendering runs on the native
client's GPU; the local browser displays its JPEG stream. A requested frame
rate is a target, not a guarantee. This is not a WebGPU or VR renderer.

MoQ priorities decide which useful data arrives first at the available
bandwidth. Filling the link is expected. The client retains verified completed
layers on disk, controls CPU/GPU residency and retries interrupted delivery.
It may eventually fetch the complete static scene. Quality at a fixed
bandwidth needs measured comparisons; minimum transferred bytes is not the
objective.

## Other paths and evidence

The older `moq3dgs-server`, `moq3dgs-client` and `scripts/run_e2e.py` entrypoints
remain TCP simulators. Use the native commands above to test MoQ/QUIC.

[The November 1 GPU report](reproduction/VALIDATION-20261101.md) records exact
full-scene transport and source-render comparison, including failed attempts.
New live-client qualification is recorded separately and does not rewrite that
historical evidence.

MIT license.
