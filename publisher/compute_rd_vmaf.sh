#!/usr/bin/env bash
# ============================================================================
#  compute_rd_vmaf.sh  --  cluster-orbit RD curve, EXACT USER FLAGS
#  --------------------------------------------------------------------------
#  Rate-Distortion curve via VMAF for progressive 3DGS streaming.
#
#  This script reproduces the user's four validated segmentation commands
#  (Bicycle x {opacity, scale}, Room x {opacity, scale}) byte-for-byte, then
#  for each of those four segmentation outputs:
#    - synthesises a smooth 5-second orbit around the chosen cluster (first
#      frame == cluster preview camera from the paper figure),
#    - renders the orbit at each progressive layer level (base=20%, +EL1=40%,
#      +EL2=60%, +EL3=80%, +EL4=100% = reference),
#    - encodes each PNG sequence to LOSSLESS H.264 MP4 (yuv420p, -crf 0),
#    - runs libvmaf per layer level vs the reference,
#    - writes per-frame and per-(scene,rank) CSVs.
#
#  Finally, aggregates across scenes per rank method into rd_curve_average.csv.
#
#  IMPORTANT FOR REPRODUCIBILITY:
#    The four segmentation commands below mirror EXACTLY the user's working
#    invocations. The only difference is --outdir, --layer_rank, and the
#    visualization-skip flags removed (so previews are kept exactly as the
#    user runs them outside this pipeline). Do NOT casually edit these.
#
#  Requirements:
#    - Python env with: torch, gsplat, ultralytics, plyfile, opencv, scipy
#    - ffmpeg compiled with libvmaf (verify: ffmpeg -filters | grep libvmaf)
#      If your system ffmpeg lacks libvmaf, set FFMPEG_BIN to a static build:
#        cd /tmp && wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
#        tar xf ffmpeg-release-amd64-static.tar.xz
#        FFMPEG_BIN=/tmp/ffmpeg-*-amd64-static/ffmpeg ./compute_rd_vmaf.sh
#
#  Usage:
#    ./compute_rd_vmaf.sh
#    ./compute_rd_vmaf.sh --skip-seg          # reuse existing segmentation outputs
#    ./compute_rd_vmaf.sh --rank opacity      # only the opacity rank method
#    ./compute_rd_vmaf.sh --scene bicycle     # only the bicycle scene
# ============================================================================

set -euo pipefail

# -------- Paths to inputs (override via env vars) ---------------------------
BICYCLE_PLY="${BICYCLE_PLY:-../GaussianAdaptiveStreamer/static/models/bicycle/Bicycle.ply}"
BICYCLE_TRAJ="${BICYCLE_TRAJ:-../GaussianAdaptiveStreamer/TestMovements/NTHU/bicycle/user1_bicycle.json}"
BICYCLE_TARGET_CLASS="${BICYCLE_TARGET_CLASS:-bicycle}"   # cluster must contain this class

ROOM_PLY="${ROOM_PLY:-../GaussianAdaptiveStreamer/static/models/room/Room.ply}"
ROOM_TRAJ="${ROOM_TRAJ:-../GaussianAdaptiveStreamer/TestMovements/NTHU/room/user1_room.json}"
ROOM_TARGET_CLASS="${ROOM_TARGET_CLASS:-chair}"

WORK_DIR="${WORK_DIR:-./rd_workdir}"

# The user's segmentation script is named gsplat_clustering.py locally; their
# working copy in this repo is gsplat_clustering_vmaf.py. Override
# SEG_SCRIPT if your filename differs.
SEG_SCRIPT="${SEG_SCRIPT:-./gsplat_clustering_vmaf.py}"
RENDER_SCRIPT="${RENDER_SCRIPT:-./render_layer_videos.py}"
ORBIT_SCRIPT="${ORBIT_SCRIPT:-./make_cluster_orbit_trajectory.py}"

# -------- Orbit knobs -------------------------------------------------------
DURATION_S="${DURATION_S:-10}"
FPS="${FPS:-30}"
ORBIT_AMPLITUDE_DEG="${ORBIT_AMPLITUDE_DEG:-90}"

DEVICE="${DEVICE:-cuda}"

# -------- Encoding & VMAF ---------------------------------------------------
FFMPEG="${FFMPEG_BIN:-${FFMPEG:-../ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg}}"

# Lossless x264, yuv420p chroma subsampling (required by libvmaf default model).
FFMPEG_X264_FLAGS="-c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv420p"
VMAF_MODEL_FLAG="${VMAF_MODEL_FLAG:-}"   # e.g. "model=version=vmaf_v0.6.1"

# -------- Selection toggles -------------------------------------------------
SKIP_SEG=0
ONLY_RANK=""           # "" = both opacity and scale; "opacity" or "scale" to filter
ONLY_SCENE=""          # "" = both; "bicycle" or "room" to filter

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-seg)    SKIP_SEG=1; shift;;
        --work-dir)    WORK_DIR="$2"; shift 2;;
        --device)      DEVICE="$2"; shift 2;;
        --duration-s)  DURATION_S="$2"; shift 2;;
        --fps)         FPS="$2"; shift 2;;
        --amplitude)   ORBIT_AMPLITUDE_DEG="$2"; shift 2;;
        --rank)        ONLY_RANK="$2"; shift 2;;
        --scene)       ONLY_SCENE="$2"; shift 2;;
        -h|--help)
            head -50 "$0" | tail -49
            exit 0;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$WORK_DIR"

# -------- Sanity checks -----------------------------------------------------
echo "===== Sanity checks ====="
for f in "$BICYCLE_PLY" "$BICYCLE_TRAJ" "$ROOM_PLY" "$ROOM_TRAJ" \
         "$SEG_SCRIPT" "$RENDER_SCRIPT" "$ORBIT_SCRIPT"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing file: $f" >&2
        exit 1
    fi
done
echo "  inputs OK"

if ! command -v "$FFMPEG" >/dev/null 2>&1 && [[ ! -x "$FFMPEG" ]]; then
    echo "ERROR: ffmpeg binary '$FFMPEG' not found." >&2
    exit 1
fi
if ! "$FFMPEG" -hide_banner -filters 2>/dev/null | grep -q libvmaf; then
    echo "ERROR: '$FFMPEG' is installed but lacks libvmaf support." >&2
    echo "  Quick fix (no compilation):" >&2
    echo "    cd /tmp" >&2
    echo "    wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz" >&2
    echo "    tar xf ffmpeg-release-amd64-static.tar.xz" >&2
    echo "    FFMPEG_BIN=\"\$(pwd)/ffmpeg-*-amd64-static/ffmpeg\" $(basename "$0")" >&2
    exit 1
fi
echo "  ffmpeg+libvmaf OK ($FFMPEG)"

PYTHON="$(command -v python || command -v python3)"
echo "  python: $PYTHON"
echo "  duration: ${DURATION_S}s @ ${FPS} fps  =  $((DURATION_S * FPS)) frames"
echo "  orbit amplitude: ±${ORBIT_AMPLITUDE_DEG}°"
echo "  ranks: ${ONLY_RANK:-opacity,scale}"
echo "  scenes: ${ONLY_SCENE:-bicycle,room}"
echo ""

# ============================================================================
#  Segmentation runners -- one per scene, EXACTLY matching the user's
#  validated commands. The ONLY parameter that varies between runs of the
#  same scene is --layer_rank and --outdir.
# ============================================================================

run_segmentation_bicycle () {
    local outdir="$1"
    local rank="$2"
    "$PYTHON" "$SEG_SCRIPT" \
        --input "$BICYCLE_PLY" \
        --camera_json "$BICYCLE_TRAJ" \
        --frame_index 0 \
        --outdir "$outdir" \
        --views 8 \
        --lateral_offsets "0" \
        --fov_scale 0.4 \
        --detector yolo \
        --yolo_model "yolo26x.pt" \
        --iou_threshold 0.2 \
        --yolo_conf 0.55 \
        --mask_pad 2 \
        --depth_band_frac 1 \
        --dilate_radius_abs 1 \
        --layer_percentages "20,40,60,80,100" \
        --layer_rank "$rank" \
        --object_preview_views all \
        --clustering \
        --device "$DEVICE"
}

run_segmentation_room () {
    local outdir="$1"
    local rank="$2"
    "$PYTHON" "$SEG_SCRIPT" \
        --input "$ROOM_PLY" \
        --camera_json "$ROOM_TRAJ" \
        --frame_index 0 \
        --outdir "$outdir" \
        --views 24 \
        --lateral_offsets "0" \
        --fov_scale 0.7 \
        --detector yolo \
        --iou_threshold 0.2 \
        --yolo_conf 0.75 \
        --mask_pad 3 \
        --depth_band_frac 2 \
        --dilate_radius_abs 1 \
        --layer_percentages "20,40,60,80,100" \
        --layer_rank "$rank" \
        --object_preview_views all \
        --clustering \
        --device "$DEVICE"
}

# ============================================================================
#  One pipeline run: process (scene, rank_method) into CSVs.
# ============================================================================

process_run () {
    local scene_name="$1"          # "bicycle" or "room"
    local rank="$2"                # "opacity" or "scale"
    local target_class="$3"
    local ply_path="$4"

    local tag="${scene_name}_${rank}"
    local seg_dir="$WORK_DIR/${tag}/seg"
    local render_dir="$WORK_DIR/${tag}/render"
    local mp4_dir="$WORK_DIR/${tag}/mp4"
    local vmaf_dir="$WORK_DIR/${tag}/vmaf"
    mkdir -p "$seg_dir" "$render_dir" "$mp4_dir" "$vmaf_dir"

    echo "============================================================"
    echo " RUN: ${tag}    target_class=${target_class}"
    echo "============================================================"

    # ---- Step 1: segmentation (idempotent) -------------------------------
    if [[ $SKIP_SEG -eq 1 && -f "$seg_dir/clusters_catalog.json" ]]; then
        echo "[$tag] step 1: SKIPPED (clusters_catalog.json exists)"
    else
        echo "[$tag] step 1: running segmentation (rank=$rank)..."
        if [[ "$scene_name" == "bicycle" ]]; then
            run_segmentation_bicycle "$seg_dir" "$rank"
        else
            run_segmentation_room "$seg_dir" "$rank"
        fi
        if [[ ! -f "$seg_dir/clusters_catalog.json" ]]; then
            echo "ERROR: segmentation did not produce clusters_catalog.json" >&2
            exit 1
        fi
    fi

    # ---- Step 2: build smooth orbit trajectory around target cluster -----
    local orbit_json="$seg_dir/cluster_orbit_traj.json"
    if [[ -f "$orbit_json" ]]; then
        echo "[$tag] step 2: SKIPPED (orbit trajectory exists)"
    else
        echo "[$tag] step 2: building orbit trajectory..."
        "$PYTHON" "$ORBIT_SCRIPT" \
            --manifest "$seg_dir/manifest.json" \
            --out_json "$orbit_json" \
            --duration_s "$DURATION_S" \
            --fps "$FPS" \
            --target_class "$target_class" \
            --orbit_amplitude_deg "$ORBIT_AMPLITUDE_DEG"
        if [[ ! -f "$orbit_json" ]]; then
            echo "ERROR: orbit generator did not produce $orbit_json" >&2
            exit 1
        fi
    fi

    # ---- Step 3: render orbit at each layer level ------------------------
    if [[ -f "$render_dir/rates.json" ]]; then
        echo "[$tag] step 3: SKIPPED (rates.json exists)"
    else
        echo "[$tag] step 3: rendering orbit at each layer level..."
        "$PYTHON" "$RENDER_SCRIPT" \
            --input "$ply_path" \
            --camera_json "$orbit_json" \
            --manifest "$seg_dir/manifest.json" \
            --out_dir "$render_dir" \
            --device "$DEVICE"
    fi

    # ---- Step 4: encode PNG sequences -> lossless MP4 --------------------
    echo "[$tag] step 4: encoding PNG sequences -> lossless MP4..."
    local rates_json="$render_dir/rates.json"
    mapfile -t LEVEL_DIRS < <("$PYTHON" -c "
import json
d = json.load(open('$rates_json'))
for n in d['level_dirs']: print(n)
")
    REF_DIR="$("$PYTHON" -c "
import json
print(json.load(open('$rates_json'))['reference_dir'])
")"

    local ref_mp4="$mp4_dir/reference.mp4"
    if [[ -f "$ref_mp4" ]]; then
        echo "  reference.mp4 exists, skipping"
    else
        "$FFMPEG" -y -hide_banner -loglevel error \
            -framerate "$FPS" \
            -i "$render_dir/$REF_DIR/frame_%06d.png" \
            $FFMPEG_X264_FLAGS \
            "$ref_mp4"
        echo "  -> $ref_mp4"
    fi

    for lvl in "${LEVEL_DIRS[@]}"; do
        local out_mp4="$mp4_dir/${lvl}.mp4"
        if [[ -f "$out_mp4" ]]; then
            echo "  ${lvl}.mp4 exists, skipping"
            continue
        fi
        "$FFMPEG" -y -hide_banner -loglevel error \
            -framerate "$FPS" \
            -i "$render_dir/$lvl/frame_%06d.png" \
            $FFMPEG_X264_FLAGS \
            "$out_mp4"
        echo "  -> $out_mp4"
    done

    # ---- Step 5: VMAF per layer level ------------------------------------
    echo "[$tag] step 5: running libvmaf..."
    for lvl in "${LEVEL_DIRS[@]}"; do
        local out_json="$vmaf_dir/${lvl}.json"
        if [[ -f "$out_json" ]]; then
            echo "  ${lvl}.json exists, skipping"
            continue
        fi
        local vmaf_filter="libvmaf=log_fmt=json:log_path=$out_json"
        if [[ -n "$VMAF_MODEL_FLAG" ]]; then
            vmaf_filter="${vmaf_filter}:${VMAF_MODEL_FLAG}"
        fi
        "$FFMPEG" -y -hide_banner -loglevel error \
            -i "$mp4_dir/${lvl}.mp4" \
            -i "$ref_mp4" \
            -lavfi "$vmaf_filter" \
            -f null -
        echo "  -> $out_json"
    done

    # ---- Step 6: per-(scene,rank) CSVs -----------------------------------
    echo "[$tag] step 6: assembling per-run CSVs..."
    "$PYTHON" - <<PYEOF
import json, os, csv

scene      = "$scene_name"
rank       = "$rank"
tag        = "$tag"
work_dir   = "$WORK_DIR"
seg_dir    = "$seg_dir"
vmaf_dir   = "$vmaf_dir"
render_dir = "$render_dir"

with open(os.path.join(render_dir, "rates.json")) as f:
    rates = json.load(f)
with open(os.path.join(seg_dir, "manifest.json")) as f:
    manifest = json.load(f)

level_dirs = rates["level_dirs"]
per_level  = rates["per_level"]

per_frame_rows = []
per_run_rows   = []

for lvl in level_dirs:
    info = per_level[lvl]
    k         = info["level"]
    cum_pct   = info["cumulative_pct"]
    cum_bytes = info["cumulative_bytes"]
    cum_kb    = info["cumulative_kb"]
    n_g       = info["n_gaussians"]

    vjson_path = os.path.join(vmaf_dir, f"{lvl}.json")
    if not os.path.exists(vjson_path):
        print(f"  WARN: no VMAF JSON for {lvl}, skipping")
        continue
    with open(vjson_path) as f:
        vj = json.load(f)

    frames = vj.get("frames", [])
    vmaf_vals = []
    for fr in frames:
        metrics = fr.get("metrics", fr)
        vmaf = metrics.get("vmaf", metrics.get("VMAF_score"))
        if vmaf is None:
            continue
        vmaf_vals.append(float(vmaf))
        per_frame_rows.append({
            "scene":            scene,
            "rank_method":      rank,
            "level":            k,
            "layer_name":       info["name"],
            "cumulative_pct":   cum_pct,
            "cumulative_bytes": cum_bytes,
            "cumulative_kb":    cum_kb,
            "frame":            fr.get("frameNum", len(vmaf_vals) - 1),
            "vmaf":             vmaf,
        })

    pooled = vj.get("pooled_metrics", {}).get("vmaf", {})
    mean_v = pooled.get("mean", (sum(vmaf_vals)/len(vmaf_vals)) if vmaf_vals else float("nan"))
    min_v  = pooled.get("min",  min(vmaf_vals) if vmaf_vals else float("nan"))
    hm_v   = pooled.get("harmonic_mean")
    if hm_v is None and vmaf_vals:
        nz = [v for v in vmaf_vals if v > 0]
        hm_v = len(nz) / sum(1.0/v for v in nz) if nz else float("nan")

    per_run_rows.append({
        "scene":              scene,
        "rank_method":        rank,
        "level":              k,
        "layer_name":         info["name"],
        "cumulative_pct":     cum_pct,
        "cumulative_bytes":   cum_bytes,
        "cumulative_kb":      cum_kb,
        "n_gaussians":        n_g,
        "n_frames":           len(vmaf_vals),
        "vmaf_mean":          round(mean_v, 4),
        "vmaf_min":           round(min_v,  4),
        "vmaf_harmonic_mean": round(hm_v,   4) if hm_v == hm_v else hm_v,
    })

pf_path = os.path.join(work_dir, f"rd_curve_{tag}_per_frame.csv")
with open(pf_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys())
                                      if per_frame_rows else
                                      ["scene","rank_method","level","layer_name",
                                       "cumulative_pct","cumulative_bytes",
                                       "cumulative_kb","frame","vmaf"])
    w.writeheader(); w.writerows(per_frame_rows)
print(f"  per-frame -> {pf_path}  ({len(per_frame_rows)} rows)")

ps_path = os.path.join(work_dir, f"rd_curve_{tag}.csv")
with open(ps_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys())
                                      if per_run_rows else
                                      ["scene","rank_method","level","layer_name",
                                       "cumulative_pct","cumulative_bytes",
                                       "cumulative_kb","n_gaussians","n_frames",
                                       "vmaf_mean","vmaf_min","vmaf_harmonic_mean"])
    w.writeheader(); w.writerows(per_run_rows)
print(f"  per-run   -> {ps_path}  ({len(per_run_rows)} rows)")
PYEOF

    echo "[$tag] DONE"
    echo ""
}

# ============================================================================
#  Run every (scene, rank) combination requested.
# ============================================================================
RUN_SCENES=()
[[ -z "$ONLY_SCENE" || "$ONLY_SCENE" == "bicycle" ]] && RUN_SCENES+=("bicycle")
[[ -z "$ONLY_SCENE" || "$ONLY_SCENE" == "room"    ]] && RUN_SCENES+=("room")

RUN_RANKS=()
[[ -z "$ONLY_RANK" || "$ONLY_RANK" == "opacity" ]] && RUN_RANKS+=("opacity")
[[ -z "$ONLY_RANK" || "$ONLY_RANK" == "scale"   ]] && RUN_RANKS+=("scale")

for scene in "${RUN_SCENES[@]}"; do
    case "$scene" in
        bicycle) ply="$BICYCLE_PLY"; tc="$BICYCLE_TARGET_CLASS";;
        room)    ply="$ROOM_PLY";    tc="$ROOM_TARGET_CLASS";;
    esac
    for rank in "${RUN_RANKS[@]}"; do
        process_run "$scene" "$rank" "$tc" "$ply"
    done
done

# ============================================================================
#  Aggregate RD curve across scenes, PER RANK METHOD. Produces:
#    rd_curve_average_opacity.csv  -- bicycle_opacity + room_opacity averaged
#    rd_curve_average_scale.csv    -- bicycle_scale   + room_scale   averaged
# ============================================================================
echo "===== Building per-rank aggregate RD curves ====="
"$PYTHON" - <<PYEOF
import csv, os
from collections import defaultdict

work_dir = "$WORK_DIR"
ranks_str = "${ONLY_RANK}"
scenes_str = "${ONLY_SCENE}"

ranks  = [r for r in ["opacity", "scale"]
          if (not ranks_str)  or (ranks_str  == r)]
scenes = [s for s in ["bicycle", "room"]
          if (not scenes_str) or (scenes_str == s)]

for rank in ranks:
    by_level_vmaf  = defaultdict(list)
    by_level_min   = defaultdict(list)
    by_level_hm    = defaultdict(list)
    by_level_bytes = defaultdict(list)
    by_level_name  = {}
    by_level_pct   = {}

    for scene in scenes:
        csv_path = os.path.join(work_dir, f"rd_curve_{scene}_{rank}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARN: missing {csv_path}, skipping")
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                k = int(row["level"])
                by_level_vmaf [k].append(float(row["vmaf_mean"]))
                try: by_level_min[k].append(float(row["vmaf_min"]))
                except: pass
                try: by_level_hm[k].append(float(row["vmaf_harmonic_mean"]))
                except: pass
                by_level_bytes[k].append(int(row["cumulative_bytes"]))
                by_level_name[k] = row["layer_name"]
                by_level_pct[k]  = float(row["cumulative_pct"])

    if not by_level_vmaf:
        print(f"  rank={rank}: nothing to aggregate, skipping")
        continue

    out_path = os.path.join(work_dir, f"rd_curve_average_{rank}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank_method", "level", "layer_name", "cumulative_pct",
                    "avg_cumulative_bytes_per_scene",
                    "sum_cumulative_bytes_both_scenes",
                    "avg_vmaf_mean", "avg_vmaf_min", "avg_vmaf_harmonic_mean",
                    "n_scenes"])
        for k in sorted(by_level_vmaf):
            avg_vmaf  = sum(by_level_vmaf [k]) / len(by_level_vmaf [k])
            avg_min   = (sum(by_level_min  [k]) / len(by_level_min  [k])) if by_level_min[k]  else ""
            avg_hm    = (sum(by_level_hm   [k]) / len(by_level_hm   [k])) if by_level_hm[k]   else ""
            avg_bytes = sum(by_level_bytes[k]) // len(by_level_bytes[k])
            sum_bytes = sum(by_level_bytes[k])
            w.writerow([rank, k, by_level_name[k], by_level_pct[k],
                        avg_bytes, sum_bytes,
                        round(avg_vmaf, 4),
                        round(avg_min, 4)  if avg_min  != "" else "",
                        round(avg_hm,  4)  if avg_hm   != "" else "",
                        len(by_level_vmaf[k])])
    print(f"  rank={rank} aggregate -> {out_path}  "
          f"({len(by_level_vmaf)} levels, {len(scenes)} scenes)")
PYEOF

echo ""
echo "===== ALL DONE ====="
echo "Per-(scene,rank) CSVs:"
for s in "${RUN_SCENES[@]}"; do
    for r in "${RUN_RANKS[@]}"; do
        echo "  $WORK_DIR/rd_curve_${s}_${r}.csv             (per-level VMAF)"
        echo "  $WORK_DIR/rd_curve_${s}_${r}_per_frame.csv   (per-frame VMAF)"
    done
done
echo "Per-rank aggregate CSVs:"
for r in "${RUN_RANKS[@]}"; do
    echo "  $WORK_DIR/rd_curve_average_${r}.csv"
done
echo ""
echo "Videos (first frame == cluster preview from paper figure):"
for s in "${RUN_SCENES[@]}"; do
    for r in "${RUN_RANKS[@]}"; do
        echo "  $WORK_DIR/${s}_${r}/mp4/reference.mp4"
    done
done