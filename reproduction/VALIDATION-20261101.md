# Reproduction repair validation — 1 November 2026

This record concerns MoQSplat correctness and release completeness. It is not
PointStream codec evidence or a reproduction of the paper's performance table.

## Source audit and changes

The public upstream, three public forks, five branches and their releases were
checked at the revisions in `fork-audit-20261101.json`. None supplied the paper's
custom Rust subgroup transport. The corresponding server blob was identical
across those branches and explicitly implemented a Python TCP simulator.

PR #2 correctly identified rendering/subscription problems. The repair loads
all RGB spherical-harmonic coefficients in their correct order, renders them
with gsplat, converts camera axes explicitly, derives forward from the view
matrix's third row, and separates requested from received/cached layers. The
proposed frustum plane negations were rejected by translated six-plane tests.

The original EyeNav conversion also missed the recorded world-axis change and
misread signed FOV field order. The repaired converter follows pinned primary
recording/replay source, keeps asymmetric angles, and records raw CSV identity.
The first Bicycle row's vertical FOV is 95.9704 degrees; the historical converter
gave 81.9903 degrees. Historical traces remain unchanged.

The independent native path pins moq-rs 0.5.2 / moq-ffi 0.4.20 and uses actual
Rust QUIC with fallback disabled. Application layers have separate native tracks;
large layers use bounded transport groups and a complete-layer SHA256. This is
an explicit modern mapping, not recovery of the customized subgroup wire format.

## Inputs and device

The Bicycle scene contains 6,131,954 Gaussians; its SHA256 is
`64d357cb25bd85f710f8551a18d830f8497277fbd8c5805adfd72ffe9ca78227`.
The corrected 2,893-frame trace SHA256 is
`4f3567862ddf4e56ab439a262bd7535980422dd05fe360c1b92b762f3fd97499`.
`corrected-input-identities-20261101.json` records the raw CSV hash and both
reference commits. The recording binary's actual revision remains unknown.

Runs use gpu6's RTX 6000 Ada, UUID
`GPU-dbbb295f-6156-1133-8658-841f3f5b17d0`, compute capability 8.9, with Python
3.12.14, Torch 2.10.0+cu128, gsplat 1.5.3 and CUDA compiler 12.4.131. The separate
environment inherits a read-only local PointStream base; unrelated packages in
that base are not required for MoQSplat. Each fleet stage saves its full freeze.

## Preserved experiments

| Fleet job | Result |
|---|---|
| `20261101T131954Z-e5f936c7` | Import timeout on NFS; no CUDA result |
| `20261101T134114Z-eeb9ec59` | 50/52 CPU tests; missing FFmpeg plugin and assertion mismatch |
| `20261101T135740Z-6f49d789` | Interrupted upload; inspected unpublished and unable to execute; never replayed |
| `20261101T144900Z-dc0598d9` | 62 CPU tests; CUDA compile header-path failure |
| `20261101T153034Z-1dbe7112` | 76 CPU tests; CUDA compiled, then packed gsplat background-argument failure |
| `20261101T154427Z-9679149a` | Snapshot omitted adapter module; failed before experiment |
| `20261101T160216Z-31d696de` | 77 CPU tests, full-scene CUDA oracle and legacy TCP client passed; native publisher hit hard group-size limit |
| `20261101T161843Z-425ac71f` | **Passed smoke and full:** 82 CPU tests, full-scene CUDA oracle, legacy TCP client and three-process native QUIC; smoke fleet validator and post-run full-artifact validation passed |

All failed attempts and their source archives are retained. The group-limit run
`20261101T160216Z-31d696de` saved a full-scene source PNG and a nonblack decoded
legacy-client MP4. Both were
visually inspected. The source render used all Gaussians; the legacy viewport
selected 4,939,558 and received all 24 expected cache entries. These results alone
do not establish successful native streaming. Conservative charges through that
run were 1,080 of the 3,600 GPU-second ceiling. The final successful run is
conservatively charged 360 seconds, including staging/supervision: **1,440 / 3,600
GPU-seconds total**. Stop after the correctness decision; no performance sweep
or training was launched.

## Successful native GPU validation

Job `20261101T161843Z-425ac71f` ran the one-frame gate, then the same path at
three trace frames. Source code matches commit `42b6bbc`; external source archive
SHA256 is `d64052cbee3e821d6dc83fb0fd767f883ec119c358165ce9bb011706a108d2c8`.
The PointStream adapter snapshot is
`793af4d68595848369d8b3bc2d99866c03cc953245a41424710e14c47f100ff7`
at `71153358`; the job specification SHA256 is
`044fda1b18c3905c9f328008c9f55c1bb94d81a7e4ba43a58484074c74ea54bb`.

The full run received **all 6,131,954 original Gaussians** in five disjoint
opacity layers: 1,447,148,624 payload bytes, with every decoded attribute checked
against its original source-array hash. The native client recorded QUIC packet
and byte counters, 45 changed priority updates over 72 pose samples, and zero
additional payload requests when revisiting cached layers. Three progressive
images were rendered, including rendering while transfers were still pending;
three final CUDA frames used the complete received scene.

The received-data final frames were visually inspected and compared to direct
full-source CUDA renders at the same corrected poses, 320×240 resolution. Their
largest absolute RGB channel differences were 1, 2 and 2 on the 0–255 scale;
mean absolute channel errors were 0.0000434, 0.0001389 and 0.0000781. These are
observed numerical agreement, not bit-identical PNGs or a performance claim.
Partitioning changes Gaussian ordering. Source attributes were byte-exact.

The smoke and full execution durations were 100.31 and 87.68 seconds. They
include preparation, serialization, rendering and validation and must not be
reported as streaming latency or network benchmarks. The fleet applies its
substantive validator to the smoke gate. The identical adapter validator was
also run locally on the exported full summary and passed; this was post-run
verification, not a second fleet promotion gate. `results-20261101/` preserves
the summaries, validation/execution JSON,
the exact spec, frozen source identities, native library hashes and image
comparison. The complete environment freeze and command logs remain alongside
the canonical fleet artifacts. The native codec library is moq-ffi 0.4.20;
legacy MP4 encode/decode used imageio-ffmpeg 0.7.0's FFmpeg 7.0.2-static, not the
host's separate FFmpeg executable.

Ten focused native regressions also passed locally, including actual QUIC
delivery larger than the pinned library's 32 MiB group cap, malformed-group
refusal, and cache deduplication. A separate bounded probe verified complete
late recovery after 33 seconds idle; indefinite relay retention is unproven.

Canonical evidence is stored under
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/JOB_ID/`.
Completed stages use `published.tar`; failed stages use `partial.tar`. Preserve
the logs, execution/validation JSON, environment freeze and archive together.

## Remaining limits

The native runner is a finite snapshot-producing harness, not a qualified
interactive viewer. It requests all catalog layers and changes their priorities;
viewport-selective fetching, continuous playback and automatic reconnection are
absent. Cache revisits concern this process's RAM cache. Payload-budget guards
do not bound peak RAM/VRAM. Short loopback runs with sequential subscribers do
not qualify WAN behavior, concurrent audiences or long-session cache behavior.
The tested environment inherited an existing local base; a fresh installation
on an independent machine has not been exercised.

The offline publisher wrapper and prepared DELTA-PLY bridge have regression
tests; the semantic detection/VMAF pipeline has not been rerun. Original binary
publisher artifacts, network-impairment commands, session configuration and
customized transport revision are still needed for an exact paper comparison.
Standalone `moq-relay` interoperability, GPU/RAM watermark swapping and the
published bandwidth/packet-loss results have not been established. Loopback
transport correctness must not be reported as those results.
