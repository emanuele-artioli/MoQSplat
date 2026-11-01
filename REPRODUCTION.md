# Reproducing MoQSplat

The repository contains three distinct implementations. Select the one needed
for the result you are trying to reproduce.

| Path | Implemented behavior | Reproduction scope |
|---|---|---|
| `publisher/` | Semantic detection, back-projection, grouping, five incremental opacity/scale quality levels, offline rendering and VMAF | Original released publisher pipeline; see its README and saved configuration |
| `moq3dgs-server`, `moq3dgs-client`, `scripts/run_e2e.py` | Python TCP simulator, WFPS tiers and SH refinements | Legacy simulator; these commands do not establish MoQ or QUIC behavior |
| `scripts/run_moq_reproduction.py` | Rust-backed MoQ over QUIC, independent layer tracks, relay/publisher/subscriber roles, verified payloads and received-data CUDA rendering | Independent functional reproduction using pinned current MoQ libraries |

The paper describes a customized Rust `moq-lite` implementation with an explicit
subgroup wire layer. That implementation has not been recovered from the public
repositories during this audit. The modern functional reproduction maps each
application `(track, group, subgroup)` to an independently subscribable native
MoQ track. It must not be described as the original customized wire protocol or
as an exact reproduction of the paper's end-to-end results.

The repaired path passed full-scene GPU validation (6,131,954 Gaussians), real
three-process QUIC delivery, and comparison against direct source rendering.
See [the validation record](reproduction/VALIDATION-20261101.md) for exact
inputs, code/environment identities, results, failed attempts and limits.

## Current native client capabilities

The native CLI is a transport/rendering correctness harness. It saves progressive
and final PNG snapshots; it does not provide an interactive browser/VR viewer or
a continuously paced playback loop. It requests every catalog layer. The trace
changes active stream priorities, but does not restrict fetching to the current
viewport. This distinction matters when evaluating bandwidth savings.

Completed payloads are cached in process RAM. Connection reconnection and
GPU/RAM watermark swapping are absent. A connection interruption can fail the
session; layer retries only apply while a usable connection remains. The 1 GiB
per-layer and 4 GiB aggregate payload guards are not peak-memory guarantees:
serialization, decoded arrays and rendering tensors can require additional RAM.

Validation covered one scene on one GPU class, short loopback sessions and
sequential subscribers. Continuous playback, WAN loss/congestion, simultaneous
audiences, long sessions and a clean-machine installation are not qualified.
The original semantic publisher-to-native-client path still needs a successful
run with complete original publisher artifacts; its bridge guards alone do not
establish that end-to-end compatibility.

## Environment

Use a separate environment; do not change another project's pinned installation.
The original `environment.yaml` describes Python 3.10, PyTorch 2.5.1 and CUDA 12.1;
that historical combination has not been requalified here. The real-scene repair
was tested separately with Python 3.12.14, PyTorch 2.10.0+cu128, gsplat 1.5.3
and an RTX 6000 Ada (compute capability 8.9). Select a compatible CUDA compiler
and headers explicitly before installing this checkout and its reproduction extras:

```sh
# Activate an isolated Python environment with the recorded Torch/CUDA build.
python -m pip install -e '.[dev,reproduce]'
python -m pip freeze --all > environment-resolved.txt
python -m pytest tests -q
```

For a Conda CUDA toolkit whose headers live under `targets/x86_64-linux`, set
`CUDA_HOME`, prepend `$CUDA_HOME/bin` to `PATH`, and add its target `include` and
`lib` directories to `CPATH` and `LIBRARY_PATH`. Keep `TORCH_EXTENSIONS_DIR` on
local storage. The recorded GPU run used CUDA compiler 12.4.131 with the above
Torch binary; this establishes that tested configuration only. The renderer
composes the returned premultiplied RGB and alpha with the background because
the pinned gsplat packed wrapper rejects an explicit background tensor.

The native bindings are pinned together: `moq-rs==0.5.2`,
`moq-ffi==0.4.20`; the renderer is `gsplat==1.5.3`. Moving-main documentation
uses different priority types; do not substitute an untested release. In this
release the native API uses unsigned higher-first priorities. The adapter maps
paper priority `p` (0 is highest) to `255-p` and tests both endpoints.

## Inputs

Obtain a real trained Bicycle 3DGS PLY and the original EyeNavGS camera trace.
They are external research assets, not downloaded automatically. Pass both
paths explicitly. The historical local links are `assets/bicycle.ply` and
`assets/bicycle_trace.json`; the latter must not be replaced by the nonexistent
`assets/eval_trace.json`. Record the source, license and SHA256 of each asset.

The real-scene renderer uses world-to-camera OpenGL matrices: negative Z is
forward and positive Y is up. It converts camera axes to gsplat's OpenCV
convention. `scripts/validate_reproduction.py` accepts an explicit input camera
convention; do not select a convention by choosing the nicer output image.
For EyeNav CSV inputs, regenerate a new trace instead of reusing a historical
JSON from the old converter:

```sh
python scripts/convert_trace.py /absolute/path/user101_bicycle.csv \
  /absolute/path/new-bicycle-scene-trace.json --eye 0
```

The converter follows the reference recording/replay software's world-axis
change: scene position is `diag(1,-1,-1) @ raw_position`; the GL view matrix is
the raw inverse pose right-multiplied by `diag(1,-1,-1,1)`. The renderer's later
left multiplication changes camera axes to CUDA/OpenCV. These are separate
operations. Scene initialization/scale must not be applied again.
The reference software records signed angles in **left, right, down, up** order;
vertical FOV is `up-down`. The first pinned Bicycle row gives 95.9704°, whereas
the old converter produced 81.9903°. All four angles and the shifted principal
point now propagate through culling and rendering. See the pinned
[recording/conversion code](https://github.com/symmru/EyeNavGS_Software/blob/88cbd793e806e1a9ae23fa7d5b93d3b3b3ac78d2/src/core/openxr/OpenXRRdrMode.cpp#L395-L420)
and [FOV construction](https://github.com/symmru/EyeNavGS_Software/blob/88cbd793e806e1a9ae23fa7d5b93d3b3b3ac78d2/src/core/openxr/OpenXRHMD.cpp#L241-L262).
The original recording binary's revision is unknown; the software revision in
converted metadata identifies the reference semantics audited here. The pinned
CSV's full Git blob and SHA256 were checked, not just its first row.

The corrected trace/input identities and historical identities are saved
separately under `reproduction/`. Use `corrected-input-identities-20261101.json`
for the repaired EyeNav conversion; the earlier identity file describes the
preserved historical trace.

## Validate the renderer before streaming

```sh
PYTHONPATH=src python scripts/validate_reproduction.py \
  --scene /absolute/path/bicycle.ply \
  --trace /absolute/path/bicycle_trace.json \
  --camera-convention opengl --device cuda:0 --frames 3 \
  --output /absolute/path/new-render-check
```

The check uses every Gaussian in the supplied scene. It requires CUDA gsplat,
rejects silent CPU fallback, validates real-data serialization, and preserves
PNG images, per-frame diagnostics, input/code checksums and `summary.json`.
Shortening the trace is a correctness smoke; it does not reproduce a complete
camera session or the published quality curves.

## Three-process MoQ/QUIC run

Start each command in a separate terminal, in the same environment. Use a new
output directory for every run. The self-signed certificate option below is
explicit and intended for the local reproduction only.

```sh
PYTHONPATH=src python scripts/run_moq_reproduction.py relay \
  --bind 127.0.0.1:4433 --duration 600

PYTHONPATH=src python scripts/run_moq_reproduction.py publish \
  --relay https://127.0.0.1:4433 --insecure \
  --scene /absolute/path/bicycle.ply --pruning opacity --duration 600

PYTHONPATH=src python scripts/run_moq_reproduction.py subscribe \
  --relay https://127.0.0.1:4433 --insecure \
  --trace /absolute/path/bicycle_trace.json --device cuda:0 --frames 3 \
  --output /absolute/path/new-moq-check
```

The relay role uses the official Rust-backed `moq.Server` shared origin in a
separate process. Validation against the standalone `moq-relay` executable is a
separate interoperability test. WebSocket fallback and client reconnection are
disabled. The relay rejects non-QUIC connections; the subscriber records native
QUIC counters rather than inferring transport from a URL or class name.

The pinned library caps each transport group at 32 MiB. Schema v2 splits each
complete application layer into consecutive groups of at most 16 MiB; receivers
verify sequence, completeness and the complete-layer hash before caching it.
Publisher, subscriber and relay origins have explicit 4 GiB cache capacities.
An actual QUIC regression verified exact recovery by a second subscriber after
33 seconds idle. The library's separate idle-cache expiry is not configurable
through this pinned FFI; this test does not establish indefinite relay retention.

The publisher's `--scene` mode forms five disjoint 20% opacity/scale layers in
one group. This validates transport on the real scene; it does not replace the
original semantic grouping experiment. For original prepared groups, use the
bridge below.

## Connect the original publisher output

Run the existing offline publisher pipeline following `publisher/README.md`.
Keep its complete output directory, including every DELTA PLY referenced by
`manifest.json`. The JSON and images committed in the repository alone are not
the binary layer assets.

Install `.[publisher,reproduce]` for that pipeline. The wrapper below requires
an existing local YOLO weight file, an original-format camera JSON and a JSON
array of additional original publisher arguments. It refuses an existing output
directory, records hashes and the complete command before running, and keeps
compiler/model caches outside the checkout:

```sh
python scripts/run_publisher_reproduction.py \
  --scene /absolute/path/bicycle.ply --camera-json /absolute/path/cameras.json \
  --weights /absolute/path/Models/YOLO/yolo26x.pt \
  --config /absolute/path/publisher-arguments.json \
  --output /absolute/path/new-publisher-run --cache /absolute/path/compiler-cache
```

The wrapper's guards are tested; a new complete semantic-publisher/VMAF run has
not been performed as part of the transport repair. Preserve the original
publisher configuration to compare with its published figures.

```sh
PYTHONPATH=src python scripts/run_moq_reproduction.py publish \
  --relay https://127.0.0.1:4433 --insecure \
  --prepared-manifest /absolute/path/publisher-output/manifest.json --duration 600
```

The bridge selects semantic clusters when enabled, otherwise object groups.
It preserves group/layer identities, loads the existing DELTA PLYs, verifies
their Gaussian counts and records their hashes. Missing assets, escaping paths
or count mismatches are errors; the bridge never silently regenerates layers.

## Evidence and limits

Keep the relay/publisher/subscriber logs together with the output manifest,
summary, PNGs, source revision, selected patch checksums, complete environment
lock, original scene/trace/layer hashes, GPU model/UUID, compiler/native-library
versions and the exact command. Record timeouts, incomplete transfers and
negative results beside successful runs. Synthetic unit tests establish code
contracts; they do not establish paper performance.

For an exact Table-I reproduction, the original network impairment commands,
session duration, publisher artifacts and customized transport revision are
still needed. A loopback QUIC test cannot reproduce a 10/30/50/100 Mbps network
experiment, packet-loss behavior, relay scheduling or original timing by itself.
GPU/RAM watermark behavior must be independently tested before claiming it.
