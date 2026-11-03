# Live native client

## Data flow and scope

The publisher prepares a static scene once. The native `--scene` path splits
it deterministically into spatial groups (at most 65,536 Gaussians by default),
then five disjoint opacity/scale quality layers per group. Every Gaussian and
its original attributes are preserved. The prepared-manifest path instead
preserves original semantic publisher groups. Both use independently
subscribable native layer tracks and real QUIC.

A subscriber expresses interest in tracks; data is then pushed on those
subscriptions. Its latest viewport updates priorities of active tracks and
selects the next queued tracks. Four downloads run concurrently by default.
This is bounded demand admission, not per-packet HTTP polling. Priorities
influence useful arrival order on a saturated link; eventual complete
static-scene delivery is valid. No quality-at-bandwidth superiority claim is
made without an appropriate measured comparison.

`subscribe --live` continuously renders the currently usable received data.
The browser controls change the camera without waiting for all enhancements.
Base layers are protected; enhancements require preceding levels in the same
group. Whole-scene preparation remains on the publisher.

## Installation and assets

Use a separate Python environment and the extras in README. Linux CUDA is
required for rendering. Python 3.10 and 3.12 have been exercised; see dated
validation records for exact builds and GPU scope. A CUDA compiler, compatible
headers, C++ compiler and Python development headers may be needed for gsplat's
first JIT build. Put compiler caches and scratch on a disk with enough space.
A fresh virtual environment is not a fresh operating system qualification.

Use one OpenCV provider: the publisher extra pins `opencv-python==4.10.0.84`,
which also satisfies Ultralytics. Do not install a conflicting headless
provider in the same environment. Run `python -m pip check` and record
`python -m pip freeze --all` after installation.

Supply a trained PLY, corrected camera trace and, for semantic preparation,
local YOLO weights explicitly. They are external research assets. The
reproduction records identify exact tested hashes; a different trained PLY
can have a different count and quality. No model weights are downloaded by
the wrapper. Preserve original assets and write each run to a new directory.

Original publisher outputs include source-index sidecars. They disambiguate
Gaussians with identical XYZ coordinates when constructing level renders.
Old outputs without these sidecars use the legacy coordinate mapping, which
now fails clearly on ambiguous matches; regenerate those outputs from the
original scene rather than silently dropping duplicate positions.

## Memory and startup

Completed payloads stream to content-addressed disk files. Size and SHA256 must
match before atomic completion. Incomplete files are not usable cache entries.
`--cache` can reuse completed files across runs. Plan disk space for the whole
scene; `--max-scene-gib` is an aggregate payload safety limit, not an RSS limit.

`--native-cache-mib` (default 128) targets the native origin cache. Active
downloads reserve capacity; a cold layer larger than the available capacity
fails with a suggested minimum. Use smaller spatial groups or a larger native
cache for large semantic layers. For the tested Bicycle semantic outputs, use
`--native-cache-mib 512` for live playback; their background layers exceed
the default 128 MiB target. The finite diagnostic subscriber already uses
4 GiB. Large prepared semantic layers need a whole completed layer before
the first usable image; small spatial groups give earlier partial coverage.
This target excludes SDK/QUIC and process
overhead and is not an exact whole-process RAM cap.

`--cpu-cache-mib` (default 128) bounds the LRU of decoded arrays. Decoding the
current layer still needs temporary memory. `--gpu-memory-mib` (default 2048)
controls resident layers, a conservative workspace estimate and the Torch
allocator limit. Driver context and allocations outside Torch are excluded.
Measured Torch peak and process peak RSS are recorded.

The GPU high/low watermarks are 90%/80% of the selected budget: when estimated
use approaches the high watermark, discard lower-priority enhancements toward
the low watermark. This headroom avoids repeated load/evict oscillation.
Promotion reloads verified disk/CPU content without downloading it again.
Scene-dependent CUDA workspace can force additional enhancement eviction.
If mandatory base layers cannot fit, the client fails clearly; it cannot
render an arbitrarily large scene on arbitrarily small hardware.

The first spatial base layer can render before the complete scene arrives.
Startup still includes catalog reception, source preparation on a cold
publisher, CUDA context/JIT initialization and one received base layer.
No fixed startup latency or frame-rate guarantee is implied.

## Recovery

On connection or unfinished-track failure, a subscriber opens a fresh native
client and discovers the broadcast again. It verifies a canonical catalog
identity before reusing completed content. A changed scene is rejected.
Completed files survive; an unfinished layer retries from its beginning.
There is no byte-range resume. Retry count and timeouts are bounded.

The publisher enables the pinned SDK's bounded reconnect/reannouncement path.
A relay restart still causes an outage while QUIC notices the failure and
endpoints reconnect; it is not seamless migration. Retry exhaustion fails
with an explicit error. A publisher process crash needs a process restart.

## Browser and telemetry

The HTTP viewer binds to loopback. For a remote GPU client, forward its viewer
port over SSH (for example `ssh -L 8765:127.0.0.1:8765 YOUR_GPU_HOST`), then
open the local URL. Keep that tunnel alive. This tunnel is for controls and
images; it does not turn a loopback MoQ connection into an internet MoQ test.

`frames.jsonl` records selected layers, camera pose, completed-layer count,
GPU promotions/evictions, CPU cache size, measured CUDA allocation, controls
and reconnections. `summary.json` reports first usable frame, frame intervals
and control-receipt-to-render samples. That latency excludes browser/display
time and is not motion-to-photon latency. Early black partial views are valid;
a nonblack source scene and enough useful coverage are needed for a visible image.

## Independent-network test

First use the CPU-only checker on a separate Linux/macOS machine. The native
wheel must be available for its architecture; CUDA is not required by this
checker. For the lightweight network-only path in a source checkout, install
`numpy`, `moq-rs==0.5.2` and `moq-ffi==0.4.20`, and prefix the command with
`PYTHONPATH=src`. The complete project installation also includes Torch.

```sh
python scripts/check_moq_connection.py \
  --relay https://YOUR_REACHABLE_RELAY:4433 \
  --broadcast moqsplat/scene --max-layers 8 \
  --output /absolute/path/new-remote-network-check
```

The relay must listen on an externally reachable address, UDP must pass the
host/firewall/NAT, and its TLS certificate must be trusted. The bundled relay
uses a generated self-signed certificate: `--insecure` is an explicit option
for an agreed trusted test endpoint. Do not expose other services for this
test. Its successful result verifies exact received attributes and zero
extra requests on a cache revisit; it does not qualify rendering.

For visual qualification, run the live subscriber on a colleague's CUDA
machine with the same source trace. Ask them to keep logs, summary, frames
and a short screen recording, then try movement before completion, revisit
a previous view, lower/raise the GPU budget and interrupt the network briefly.
Record their OS, GPU, installation steps, relay location, duration and
observed link conditions. Repeat with two independent subscribers for
audience behavior. No human is needed for the automated controlled tests;
a human is useful for actual browser responsiveness and a new machine/network.

`scripts/validate_client_experience.py` automates semantic publisher/VMAF,
local received-data rendering, three user-space UDP impairment profiles,
controls, GPU eviction and relay restart. Smoke uses one client and full
uses two; sessions are deliberately bounded. These are not a substitute for
physical WAN testing, long-duration soak qualification or the original paper's
network settings. Preserve negative results beside successful cases.
