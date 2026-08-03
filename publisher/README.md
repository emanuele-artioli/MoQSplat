# 🚀 Publisher-side Progressive 3D Gaussian Splatting Streaming Evaluation

[![3DGS](https://img.shields.io/badge/3DGS-Gaussian%20Splatting-blue)]()
[![VMAF](https://img.shields.io/badge/Quality-VMAF-green)]()
[![MoQSplat](https://img.shields.io/badge/Based%20on-MoQSplat-orange)]()

This repository extends **MoQSplat** with a publisher-side pipeline for **progressive 3D Gaussian Splatting (3DGS) preparation and layer-distortion evaluation**.

The framework enables object-aware Gaussian selection, progressive layer generation, view-consistent rendering, and **VMAF-based quality analysis** to study the trade-off between transmitted Gaussian data and visual quality under bandwidth constraints.

---

# ✨ Overview

The publisher pipeline provides an end-to-end workflow for generating and evaluating progressive 3DGS representations:

| Module | Description |
|:--|:--|
| 🎯 **Gaussian Decomposition** | Object-aware Gaussian clustering from multi-view scene analysis |
| 🧩 **Progressive Layering** | Hierarchical Gaussian transmission levels (20%, 40%, 60%, 80%, 100%) |
| 🎥 **Trajectory Generation** | Smooth camera orbit generation around selected objects |
| 🖥️ **Progressive Rendering** | Rendering each Gaussian transmission layer |
| 📈 **RD Evaluation** | VMAF-based layer-distortion analysis |

---

# 🔄 Pipeline

    🌐 3DGS Scene (.ply)
             |
             v
    🎯 Object-aware Gaussian Selection
             |
             v
    🧩 Progressive Gaussian Layers
             |
             v
    🎥 Camera Trajectory Generation
             |
             v
    🖥️ Layer Rendering
             |
             v
    🎞️ Video Encoding
             |
             v
    📊 VMAF Layer-Distortion Analysis


---

# 📁 Publisher Module Structure

publisher/

├── compute_rd_vmaf.sh
│ 🚀 End-to-end RD evaluation pipeline
│
├── gsplat_clustering_vmaf.py
│ 🎯 Object detection, Gaussian clustering,
│ and progressive layer generation
│
├── make_cluster_orbit_trajectory.py
│ 🎥 Camera orbit trajectory generation
│
└── render_layer_videos.py
🖥️ Progressive Gaussian layer rendering


---

# 🧠 Gaussian Pruning Stlayergies

The framework supports different Gaussian ordering stlayergies for progressive transmission.

## 🔵 Opacity-based Pruning

Gaussian primitives are prioritized according to opacity contribution.

Higher-opacity Gaussians are transmitted earlier, favoring primitives with stronger visual impact.

---

## 🟣 Scale-based Pruning

Gaussian primitives are prioritized according to spatial scale.

Large-scale Gaussians are transmitted earlier, favoring primitives covering larger scene regions.


---

# 📦 Requirements

Python dependencies:
```bash
torch | gsplat | ultralytics | plyfile | opencv-python | scipy
```
Additionally, FFmpeg with **libvmaf** support is required:

```bash
ffmpeg -filters | grep libvmaf
```
# 📊 Evaluation

1- Run the complete RD evaluation:
./compute_rd_vmaf.sh

2-Evaluate a specific ranking stlayergy:
./compute_rd_vmaf.sh --rank opacity 

3- Evaluate a specific scene:
./compute_rd_vmaf.sh --scene bicycle

4- Reuse existing segmentation results:
./compute_rd_vmaf.sh --skip-seg

5- Results will be:
rd_workdir/

├── rd_curve_<scene>_<rank>.csv
│   Scene-level RD measurements
│
├── rd_curve_<scene>_<rank>_per_frame.csv
│   Frame-level VMAF results
│
└── rd_curve_average_<rank>.csv
    Aggregated RD curves
