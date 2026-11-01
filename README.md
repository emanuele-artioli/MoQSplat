# 3DGS-MoQ: Viewport-Aware 3D Gaussian Splatting over Media-over-QUIC

Streams pre-trained 3D Gaussian Splatting scenes to clients over Media-over-QUIC
(MoQ), with viewport-aware spatial partitioning and dynamic priority scheduling.

**Reproduction status:** read [REPRODUCTION.md](REPRODUCTION.md) before running
experiments. The legacy quick-start commands below use a TCP simulator. The
native MoQ/QUIC path and original-publisher input bridge are provided separately;
the paper's customized subgroup transport has not been recovered.

## Architecture

```
Server                              Client
┌──────────────────────┐            ┌──────────────────────┐
│  Scene Loader (PLY)  │            │  Trace Replayer      │
│  Spatial Partitioner │──manifest─>│  Frustum Calculator  │
│  MoQ Publisher       │──clusters─>│  Subscription Mgr    │
│  Priority Scheduler  │<─viewport──│  Splat Cache         │
└──────────────────────┘            │  GPU Renderer        │
                                    │  Disk Writer         │
                                    └──────────────────────┘
```

### MoQ Transport Mapping

| MoQ Concept   | 3DGS Mapping                                      |
|---------------|----------------------------------------------------|
| **Broadcast** | Entire 3DGS scene                                  |
| **Track**     | Spatial volume (Octree leaf / e.g., Room)          |
| **Group**     | Density cluster within a Track (K-Means / Item)    |
| **Subgroup**  | Importance Tier (LoD based on splat scale/volume)  |
| **Object**    | Chunked payload (SH bands split into Objects)      |

**Subgroup (Importance Tier) Breakdown:**
*   **Subgroup 0**: Large Splats (Geometry + Opacity + SH0)
*   **Subgroup 1**: Medium Splats (Geometry + Opacity + SH0)
*   **Subgroup 2**: Small/Fine Splats (Geometry + Opacity + SH0)
*   **Subgroup 3**: Large Splats (SH1-SH3)
*   **Subgroup 4**: Medium/Small Splats (SH1-SH3)

*Note: Base geometry (Subgroups 0-2) always holds priority over reflections (Subgroups 3-4).*

## Setup

```bash
# Create conda environment (CUDA + PyTorch)
conda env create -f environment.yaml
conda activate 3dgs_moq

# Or install directly via pip (requires PyTorch pre-installed)
pip install -e ".[dev]"
```

## Quick Start

The server loads the PLY, partitions it into 3D clusters (using K-means), and starts an MoQ/QUIC publisher.

> [!NOTE]
> On this machine, GPU 0 is often occupied by other processes. It is recommended to use `cuda:1` for rendering.

```bash
# Start server
moq3dgs-server --scene assets/bicycle.ply --clusters 64 --port 4433

# Start client (points to GPU 1 by default)
moq3dgs-client --trace assets/bicycle_trace.json --device cuda:1
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
src/moq3dgs/
├── models.py           # Pydantic data contracts
├── decorators.py       # @network_bound, @gpu_bound
├── scene/
│   ├── loader.py       # PLY scene loading
│   ├── clustering.py   # K-means + octree partitioning
│   └── lod.py          # LoD splitting (base / enhancement)
├── viewport/
│   ├── frustum.py      # View frustum culling
│   ├── priority.py     # Dynamic priority (0-255)
│   └── trace.py        # Camera trace replay
├── transport/
│   ├── manifest.py     # MoQ manifest generation
│   ├── protocol.py     # Binary wire protocol
│   ├── server.py       # Async MoQ publisher
│   └── client.py       # Async MoQ subscriber
├── render/
│   ├── cache.py        # Persistent splat cache
│   ├── rasterizer.py   # GPU rasterisation wrapper
│   └── writer.py       # Frame-to-disk writer
├── server_app.py       # Server entry point
└── client_app.py       # Client entry point
```

## Adaptive Bitrate & Budget Management

The system implements a multi-layer adaptive strategy to ensure visual fidelity and real-time performance:

### 1. Spatial-Aware LoD (Server-Side)
Gaussians are partitioned into 3 **Subsets** (Large, Medium, Small) using **Weighted Farthest-Point Sampling (WFPS)**. Each subset is available in 2 **Quality Levels**:
- **Quality 0 (Base)**: Essential geometry + DC color (SH0). Self-contained.
- **Quality 1 (Full)**: Adds high-frequency detail (SH1-3).

### 2. Breadth-First ABR (Client-Side)
The client follows a "fill-then-upgrade" strategy:
- **Base Coverage**: Subscribes to Quality 0 for all visible subsets (Large → Medium → Small).
- **Detail Enhancement**: Once Quality 0 is received for visible clusters, it upgrades to Quality 1.
- **Persistence**: Subscriptions are maintained as long as clusters are near the viewport to prevent "popping".

### 3. Adaptive Rendering Budget
To maintain a target of **24 FPS (40ms/frame)**, the client dynamically manages a **Gaussian Budget**:
- **Halo Culling**: Visibility testing uses a 20% margin (halo) to prevent edge popping.
- **Priority Ranking**: Visible clusters are ranked by Distance, Centeredness, and Layer type.
- **Dynamic Feedback**: If render time exceeds 40ms, the budget (max splats) is reduced. If under 30ms, it is increased to allow higher detail.

## Priority Scheme

Priority `0` (highest) → `255` (lowest), computed dynamically:

1. **Distance to camera** — closer = higher priority
2. **Frustum position** — centre = highest, periphery = medium, behind = lowest
3. **Importance Tier (Subgroup)** — Subgroups 0-2 (Base Geometry) > Subgroups 3-4 (SH Reflections). Priorities ebb and flow based on the viewport, but geometry always maintains a baseline priority over reflections.

## Next Steps

- [ ] Implement occlusion culling for indoor scenes (Track-level visibility based on portals).
- [ ] Transition from TCP placeholder to full QUIC transport using `aioquic`.
- [ ] Add support for dynamic LoD transitions using alpha-blending to prevent "popping" between tiers.

## License

MIT
