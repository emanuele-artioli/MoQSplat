# Client and reproduction qualification — 3 November 2026

The release now has a continuous native CUDA client, browser controls, disk-backed
verified caching, bounded CPU/GPU residency and reconnection. This is an independent
modern MoQ/QUIC implementation. It is not the paper's unrecovered custom Rust
subgroup implementation or an exact reproduction of Table I.

## Verified behavior

| Check | Observed result |
|---|---|
| Independent installation | New Linux Python 3.10.12 venv, system-site-packages=false; pip check clean; **119 tests passed** in the final gate |
| Complete semantic pipeline | Real YOLO, full Bicycle scene, five-level VMAF, original DELTA output consumed over native QUIC and rendered on CUDA |
| Semantic source integrity | All **6,131,954 Gaussians**, ten semantic layers, exact original render attributes; source/received image max RGB difference **1/255** |
| Single live subscriber | Direct and constrained playback; all 640 spatial layers received in direct/recovery cases; actual GPU eviction and zero new payload requests on cache revisits |
| Two direct subscribers | **990 / 960 frames**, each received all 640 layers within a 90-second session; that job's separate eviction-controller assertion failed, as explained below |
| Two constrained subscribers | Actual **40-second pilot**, aggregate 120 Mbps, 80 ms one-way delay, 0.1% loss: **419 / 411 frames**, 128 / 90 layers received; source attributes verified |
| Two recovery subscribers | **180-second sessions**, aggregate 200 Mbps, 10 ms one-way delay, 0.1% loss and abrupt relay restart: **1,989 / 1,993 frames**, both received all 640 layers and reconnected once |
| Residency and cache, final recovery | **589 / 593 GPU evictions**, 1,229 / 1,233 promotions; cache revisits issued **0 / 0 new payload requests** |
| CUDA thread and memory | One render thread per client; no recorded allocated, attempted, reserved or resident frame exceeded its selected GPU budget |

The semantic setting is a new bounded functional configuration: four views,
no dilation, opacity layers 20/40/60/80/100 and four orbit frames. It is not the
original paper operating point. The preserved earlier semantic curve is
70.9639, 92.1280, 97.7351, 99.3894, 99.3570 VMAF; all metrics/counts/sizes were
checked for validity. Quality is not assumed monotonic. The original renderer
was tracked upstream; its absence from an earlier partial checkout was not a
public missing-file defect. Actual broken module imports were repaired.

MoQ pushes data after subscription. Bounded concurrent interests and viewport
priorities select useful arrival order on an occupied link. Eventual full-scene
delivery is valid. These checks establish integrity and functionality, not
bandwidth savings or superiority in viewport quality at a fixed bandwidth.

## Visible behavior and limits

The first useful final-case frames appeared in approximately 2–5.5 seconds.
Early control receipt-to-completed-render samples were approximately 34–109 ms.
These are **diagnostic observations under shared GPU contention**, not benchmark
or motion-to-photon claims. Requested frame rate is a target.

The final recovery clients produced 45 / 36 raw black partial renders while
viewport data was unavailable. All were displayed using the last useful JPEG,
with an explicit waiting message and stale displayed-pose/age metadata. Maximum
held-image age was approximately 2.70 / 3.04 seconds. Raw evidence remains black;
held images are not counted as newly rendered nonblack views. Fully received
empty views are displayed normally.

Changing GPU memory budgets caused frame gaps up to approximately 2.39 seconds.
Uniformly smooth playback is therefore not established. CPU cache limits exclude
transient decoding and the rest of the process; GPU limits exclude driver context
and allocations outside Torch. Final recovery peak Torch allocated memory was
about 4.52 GB per client, reserved about 6.43 GB, and process peak RSS about 1.58 GB.
Mandatory base layers can still be too large for a device. Large prepared semantic
layers need a larger native cache target (512 MiB for the tested Bicycle outputs).
Server whole-scene preparation remains offline; the live client receives layer
files and does not load the source PLY or prepare the whole scene.

The two-subscriber recovery test lasts three minutes. It is not an hours-long
soak or an audience-capacity benchmark. Its recorded trace can reach its final
pose before the session ends; controls and memory changes are explicit events.
The constrained pilot actually ran 40 seconds because the existing smoke profile
selects that duration; the planned focused 90-second constrained pilot was not run.

## Environment, identities and evidence

Final job: **`20261103T162943Z-5642b945`**, COMPLETE. Fleet smoke validator passed;
the full command's substantive checks passed and the same adapter validator was
run separately against exported full artifacts: passed, 3,982 frames. The fleet
itself invokes the external validator for smoke only.

Final production/qualification source: `c96339fe8fb9eaf6c887b5446fbe059e6b4d8ba6`;
source archive SHA256
`8b6b4c31ef22705184960c66ac9081d76cea6aa87564d4ae2686254bdce94375`.
Focused runs explicitly reuse the unchanged original-publisher/source-render and
single-client evidence from job `20261103T154539Z-3d7be786` smoke, summary SHA256
`40bd034989e9d980c68925b769be3b42faba5a071b9544a3db1979222adc2ae2`.
That smoke passed all three network cases, including 1,039 recovery frames.

GPU: gpu6 RTX 6000 Ada, capability 8.9, UUID
`GPU-ef96fc4f-134e-4c01-290e-9a6c8bbef553` (the earlier pipeline also used gpu6's
other Ada). Torch 2.10.0+cu128, gsplat 1.5.3, moq-rs 0.5.2 / moq-ffi 0.4.20,
OpenCV 4.10.0.84, Ultralytics 8.4.6. CUDA compiler 12.4.131 with explicit toolkit
headers/libraries; FFmpeg with libvmaf at `/opt/local/bin/ffmpeg`.
Native library `moq_ffi/_uniffi/libmoq_ffi.so` SHA256:
`aa313c69b57e287234cd2d925bdeeaa27bdb9609c3bf6c059a9d5700f4b8a36a`.
The [dependency freeze](environment-clean-20261103.txt),
[input/source identities and selected results](results-20261103/),
[operating guide](../CLIENT.md), and [November 1 record](VALIDATION-20261101.md)
provide commands, camera provenance and older evidence.

Canonical raw logs, frame journals, complete manifests, staging/execution records
and `published.tar` / `partial.tar` archives remain under the fleet inbox job IDs
in the experiment record. Compact journal aggregates identify their raw hashes.
The aggregate received-layer count may advance after the last render before a
session stops; final summary counts and journal counts are reported separately.
PointStream adapter checks: Ruff, pinned mypy (382 files), pytest 1,450 passed /
30 skipped. Legacy global MoQSplat Ruff still has existing diagnostics; focused
new/native files pass. No new operating-system installation was tested.

## Preserved failed gates

| Job | Failure or scope |
|---|---|
| `20261103T150639Z-1ceedad4` | Publication rejected staged Models path; inspected unpublished, never executed |
| `20261103T150751Z-2f6bb31b` | Qualification passed unsupported size arguments to the fixed-size oracle |
| `20261103T151437Z-d50b3f16` | Source CUDA and semantic preparation passed; upstream orbit helper imported an absent module |
| `20261103T152241Z-0989d383` | Semantic/VMAF/native and direct/constrained live cases passed; recovery reconnected but later cuSolver handle creation failed; dedicated CUDA worker introduced and requalified |
| `20261103T153511Z-10f1d7a3` | Pending request cancelled without execution before correcting its resource reservation |
| `20261103T154539Z-3d7be786` | Complete single-client smoke passed; both concurrent clients remained healthy but had 597 / 614 layers at the 60-second completion cutoff |
| `20261103T161225Z-f5903422` | Both clients received all layers in 90 seconds; driver treated empty startup status as complete and tested eviction too early. Controller fixed; actual concurrent eviction verified in the final recovery run |

All failed outputs and source revisions are retained. Conservative GPU accounting
is **3,600 / 3,600 seconds** including the earlier November 1 work and staging /
supervision allowances. No further GPU run, training or paid API was launched.
A bounded environment-pack attempt timed out; its incomplete archive is marked
invalid, retained, and was never used as an environment.

## External tests and original artifacts

Two CPU-only Mac-to-gpu6 attempts did not establish direct QUIC reachability.
The first retry overlapped endpoint availability too briefly; a synchronized
retry also timed out. The relay accepted its local publisher, but no external
Mac session. The specific firewall/VPN/network-policy cause is unknown. Temporary
relays expired and stopped; no firewall change or persistent deployment was made.
A permitted UDP route / trusted reachable endpoint is needed before outside-network
success can be claimed. CLIENT.md includes a lightweight separate-machine checker.

A colleague with a separate CUDA machine should check new-OS setup, browser
responsiveness, visible pauses, longer sessions and an actual network outage.
The browser displays the native client's GPU JPEG stream; it is not a WebGPU/VR
renderer. Controlled local impairment does not establish real WAN behavior.

The public fork/branch audit did not locate the original customized Rust transport
or the full original experiment bundle. Those original artifacts/configurations
remain necessary for exact paper-performance reproduction. Extended-paper work
is outside this task.
