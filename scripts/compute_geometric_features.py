#!/usr/bin/env python3
"""
compute_geometric_features.py
=================================
Tính 8 geometric features per-point từ raw meshes và tạo ra enriched HDF5 files
với input shape [B, N, 11] cho PointNeXt.

Feature layout (11 dims total):
    cols 0-2 │ x, y, z          │ XYZ normalized to unit sphere (same as baseline)
    col  3   │ k1               │ Principal curvature maximum  (per-vertex → barycentric interp)
    col  4   │ k2               │ Principal curvature minimum  (per-vertex → barycentric interp)
    col  5   │ H                │ Mean curvature = (k1+k2)/2   (per-vertex → barycentric interp)
    col  6   │ K                │ Gaussian curvature = k1×k2   (per-vertex → barycentric interp)
    col  7   │ sa_v_ratio       │ Surface area / volume (global mesh metric, broadcast to all pts)
    col  8   │ dist_centroid    │ L2 distance from centroid (computed on normalized point cloud)
    col  9   │ local_density    │ Mean dist to 16-NN           (computed on normalized point cloud)
    col 10   │ boundary_dist    │ Min dist from each point to nearest boundary edge (topology-based)

Normalization strategy:
    - XYZ:         per-sample (center + unit sphere), same as baseline
    - Features 3–10: global z-score using TRAIN split statistics (mean/std stored in metadata)

Supported datasets:
    fantastic_breaks — PLY meshes in data/Fantastic_Breaks_v1/
    breaking_bad     — OBJ meshes in data/BreakingBad/

Usage:
    # Activate conda env with PyVista + trimesh
    source ~/anaconda3/bin/activate pointnet

    # Fantastic Breaks (with meta-based fracture_mask)
    python scripts/compute_geometric_features.py \\
        --dataset fantastic_breaks \\
        --data_root data/Fantastic_Breaks_v1 \\
        --output_dir data/fantastic-breaks-classification \\
        --num_points 8192 --seed 42

    # Breaking Bad (no meta → fracture_mask = 0)
    python scripts/compute_geometric_features.py \\
        --dataset breaking_bad \\
        --data_root data/BreakingBad \\
        --split_dir data/BreakingBad/data_split \\
        --output_dir data/breakingbad_classification \\
        --subsets artifact everyday/Vase everyday/Mug everyday/Cup everyday/Plate \\
        --num_points 8192 --balance undersample --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature constants ─────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "x", "y", "z",        # 0-2 : XYZ (unit sphere normalized)
    "k1",                  # 3   : max principal curvature
    "k2",                  # 4   : min principal curvature
    "H",                   # 5   : mean curvature = (k1+k2)/2
    "K",                   # 6   : Gaussian curvature = k1×k2
    "sa_v_ratio",          # 7   : surface area / volume
    "dist_centroid",       # 8   : L2 dist from centroid
    "local_density",       # 9   : mean dist to 16-NN
    "boundary_dist",       # 10  : min dist to nearest boundary edge
]
N_TOTAL_DIMS   = len(FEATURE_NAMES)  # 11
GEOM_SLICE     = slice(3, N_TOTAL_DIMS)  # features to z-score normalize (3..10)
CURV_CLIP      = 50.0    # symmetric clip for curvatures on unit-sphere mesh
SAV_MAX        = 500.0   # safety cap for SA/V before global normalization
KNN_K          = 16      # number of neighbours for local density
BOUNDARY_DEFAULT = 1.0   # default boundary_dist for watertight meshes (no boundary)


# ═════════════════════════════════════════════════════════════════════════════
# Core: single-mesh processing
# ═════════════════════════════════════════════════════════════════════════════

def _compute_boundary_distance(
    tm,
    pts_norm: np.ndarray,
    norm_verts: np.ndarray,
) -> np.ndarray:
    """
    Compute per-point minimum distance to the nearest boundary edge.

    Boundary edges are edges shared by exactly 1 face (open mesh boundary).
    For watertight meshes (e.g. complete objects), there are no boundary edges
    and all points get BOUNDARY_DEFAULT (1.0).

    Returns: float32 array of shape (N,).
    """
    from scipy.spatial import cKDTree

    # Find boundary edges: edges appearing in exactly 1 face
    edges = tm.edges_sorted  # (E, 2)
    # Use a fast counting approach via numpy
    # Convert edge pairs to a single uint64 for fast counting
    max_idx = int(edges.max()) + 1
    edge_keys = edges[:, 0].astype(np.int64) * max_idx + edges[:, 1].astype(np.int64)
    unique_keys, counts = np.unique(edge_keys, return_counts=True)
    boundary_keys = unique_keys[counts == 1]

    if len(boundary_keys) == 0:
        # Watertight mesh — no boundary edges → uniform default
        return np.full(len(pts_norm), BOUNDARY_DEFAULT, dtype=np.float32)

    # Decode boundary edge keys back to vertex indices
    bnd_v0 = (boundary_keys // max_idx).astype(np.int64)
    bnd_v1 = (boundary_keys % max_idx).astype(np.int64)

    # Compute midpoints of boundary edges in normalized space (fast approximation)
    bnd_midpoints = (norm_verts[bnd_v0] + norm_verts[bnd_v1]) / 2.0  # (B, 3)

    # Build KD-tree on boundary midpoints and query for each sampled point
    bnd_tree = cKDTree(bnd_midpoints.astype(np.float64))
    dists, _ = bnd_tree.query(pts_norm.astype(np.float64), k=1)

    return dists.astype(np.float32)


def process_mesh(
    mesh_path: str,
    num_points: int,
    seed: int,
) -> Optional[np.ndarray]:
    """
    Load one mesh and return a [num_points, 11] float32 feature array.

    Pipeline:
      1. Load mesh with trimesh.
      2. Sample N points + get face indices (for barycentric interp).
      3. Normalize XYZ to unit sphere (center + scale).
      4. Apply same transform to mesh vertices → build PyVista PolyData.
      5. Compute per-vertex curvatures (k1, k2, H, K) via PyVista.
      6. Compute SA/V ratio from the normalized mesh.
      7. Barycentric-interpolate vertex curvatures to sampled points.
      8. Compute dist_centroid and local_density on the normalized point cloud.
      9. Compute boundary_distance (topology-based, no meta needed).
     10. Stack into [N, 11] and return.

    Returns None on any failure (degenerate mesh, loading error, etc.).
    """
    # Lazy import to keep module-level startup fast
    import trimesh
    import pyvista as pv
    from scipy.spatial import cKDTree

    try:
        # ── 1. Load with trimesh ──────────────────────────────────────────────
        tm = trimesh.load(mesh_path, force="mesh", process=False)
        if tm.vertices.shape[0] < 4 or len(tm.faces) < 4:
            raise ValueError(
                f"Degenerate mesh: {tm.vertices.shape[0]} verts / {len(tm.faces)} faces"
            )

        # ── 2. Sample N surface points ────────────────────────────────────────
        # trimesh.sample.sample_surface returns (pts, face_ids) where face_ids[i]
        # is the triangle index that point i was sampled from.
        raw_pts, face_ids = trimesh.sample.sample_surface(
            tm, num_points, seed=seed
        )
        raw_pts = raw_pts.astype(np.float64)

        # ── 3. Per-sample XYZ normalization (center + unit sphere) ────────────
        centroid = raw_pts.mean(axis=0)            # (3,)
        shifted  = raw_pts - centroid              # (N, 3)
        scale    = np.max(np.linalg.norm(shifted, axis=1))
        if scale < 1e-8:
            raise ValueError(f"Near-zero bounding radius: {scale:.2e}")
        pts_norm = (shifted / scale).astype(np.float32)   # (N, 3), unit sphere

        # ── 4. Normalize mesh vertices with the same transform ────────────────
        norm_verts = ((tm.vertices - centroid) / scale).astype(np.float32)

        # Build PyVista PolyData from normalized vertices
        faces_pv = np.hstack([
            np.full((len(tm.faces), 1), 3, dtype=np.int32),
            tm.faces.astype(np.int32),
        ]).ravel()
        pv_mesh = pv.PolyData(norm_verts, faces_pv)

        # ── 5. Per-vertex curvatures via PyVista ──────────────────────────────
        def _safe_curv(curv_type: str) -> np.ndarray:
            """Compute curvature; clip and zero-fill NaN/Inf on failure."""
            try:
                vals = np.asarray(pv_mesh.curvature(curv_type), dtype=np.float32)
                vals = np.nan_to_num(vals, nan=0.0,
                                     posinf=CURV_CLIP, neginf=-CURV_CLIP)
                return np.clip(vals, -CURV_CLIP, CURV_CLIP)
            except Exception:
                return np.zeros(pv_mesh.n_points, dtype=np.float32)

        k1_vert = _safe_curv("maximum")   # (V,)
        k2_vert = _safe_curv("minimum")   # (V,)
        H_vert  = _safe_curv("mean")      # (V,)
        K_vert  = _safe_curv("gaussian")  # (V,)

        # ── 6. Global SA/V ratio (computed on normalized mesh) ────────────────
        sa  = float(pv_mesh.area)
        vol = abs(float(pv_mesh.volume))   # abs: open meshes can give signed vol
        # For broken fragments (open meshes), vol ≈ 0 → high SA/V → discriminative!
        sa_v = float(np.clip(sa / max(vol, 1e-4), 0.0, SAV_MAX))

        # ── 7. Barycentric interpolation: vertex curvatures → sampled points ──
        # face_verts_pos[i] = 3D positions of the 3 vertices of face face_ids[i]
        face_verts_pos = norm_verts[tm.faces[face_ids]]          # (N, 3, 3)
        bary = trimesh.triangles.points_to_barycentric(
            triangles=face_verts_pos, points=pts_norm
        ).astype(np.float64)                                      # (N, 3)
        # Numerical safety: bary coords should sum to 1 per row
        bary = np.clip(bary, 0.0, 1.0)
        bary /= bary.sum(axis=1, keepdims=True).clip(1e-8, None)

        def _bary_interp(attr_vert: np.ndarray) -> np.ndarray:
            """Weighted sum of vertex attribute using barycentric coords."""
            face_vals = attr_vert[tm.faces[face_ids]]    # (N, 3)
            return (bary * face_vals).sum(axis=1).astype(np.float32)

        k1_pts = _bary_interp(k1_vert)
        k2_pts = _bary_interp(k2_vert)
        H_pts  = _bary_interp(H_vert)
        K_pts  = _bary_interp(K_vert)

        # ── 8. Post-sampling point-cloud features ─────────────────────────────
        # dist_centroid: L2 distance from origin (= centroid, since pts_norm is centered)
        dist_centroid = np.linalg.norm(pts_norm, axis=1).astype(np.float32)  # (N,)

        # local_density: mean distance to 16 nearest neighbours
        tree = cKDTree(pts_norm.astype(np.float64))
        nn_dists, _ = tree.query(pts_norm, k=KNN_K + 1)   # +1: first is self (dist=0)
        local_density = nn_dists[:, 1:].mean(axis=1).astype(np.float32)      # (N,)

        # ── 9. Boundary distance (topology-based) ─────────────────────────────
        boundary_dist = _compute_boundary_distance(tm, pts_norm, norm_verts)  # (N,)

        # ── 10. Assemble [N, 11] feature matrix ──────────────────────────────
        sa_v_col = np.full(num_points, sa_v, dtype=np.float32)

        features = np.stack([
            pts_norm[:, 0],   # x
            pts_norm[:, 1],   # y
            pts_norm[:, 2],   # z
            k1_pts,           # k1
            k2_pts,           # k2
            H_pts,            # H
            K_pts,            # K
            sa_v_col,         # sa_v_ratio (broadcast)
            dist_centroid,    # dist_centroid
            local_density,    # local_density
            boundary_dist,    # boundary_dist
        ], axis=1)            # (N, 11)

        return features

    except Exception as exc:
        log.warning(f"  ✗  Failed [{Path(mesh_path).name}]: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Normalization
# ═════════════════════════════════════════════════════════════════════════════

def compute_train_stats(
    data: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Compute per-feature mean and std on the TRAIN split (features 3-9 only).
    data: float32 [B, N, 10]
    Returns: {feature_name: {mean: float, std: float, min: float, max: float}}
    """
    stats: dict[str, dict[str, float]] = {}
    # Flatten B×N → (B*N,) per feature dimension
    flat = data[:, :, 3:].reshape(-1, N_TOTAL_DIMS - 3)   # (B*N, 7)
    for i, name in enumerate(FEATURE_NAMES[3:]):
        col = flat[:, i]
        stats[name] = {
            "mean": float(col.mean()),
            "std":  float(col.std()) + 1e-8,
            "min":  float(col.min()),
            "max":  float(col.max()),
        }
    return stats


def apply_normalization(
    data: np.ndarray,
    stats: dict[str, dict[str, float]],
    clip_sigma: float = 5.0,
) -> np.ndarray:
    """
    Apply z-score normalization to features 3-9 using pre-computed train stats.
    Clips to ±clip_sigma after normalization.
    data: float32 [B, N, 10]  (modified in-place copy)
    """
    data = data.copy()
    for i, name in enumerate(FEATURE_NAMES[3:]):
        col_idx = 3 + i
        mu, sigma = stats[name]["mean"], stats[name]["std"]
        data[:, :, col_idx] = np.clip(
            (data[:, :, col_idx] - mu) / sigma,
            -clip_sigma, clip_sigma,
        ).astype(np.float32)
    return data


# ═════════════════════════════════════════════════════════════════════════════
# Dataset-specific sample discovery
# ═════════════════════════════════════════════════════════════════════════════

# ── Fantastic Breaks ──────────────────────────────────────────────────────────

def _discover_fb_samples_all(data_root: str) -> list[dict]:
    """Walk Fantastic_Breaks_v1 and return list of {path, label, id}."""
    samples = []
    for cat_dir in sorted(glob.glob(os.path.join(data_root, "*"))):
        if not os.path.isdir(cat_dir):
            continue
        for obj_dir in sorted(glob.glob(os.path.join(cat_dir, "*"))):
            if not os.path.isdir(obj_dir):
                continue
            obj_id = os.path.basename(obj_dir)
            c = os.path.join(obj_dir, "model_c.ply")
            b = os.path.join(obj_dir, "model_b_0.ply")
            if os.path.exists(c):
                samples.append({"path": c, "label": 0, "id": f"{obj_id}_c"})
            if os.path.exists(b):
                samples.append({"path": b, "label": 1, "id": f"{obj_id}_b"})
    return samples


def discover_fb_splits(
    data_root: str, seed: int = 42, test_ratio: float = 0.2
) -> dict[str, list[dict]]:
    """
    Reproduce the exact same stratified train/test split used by
    prepare_classification_data.py (seed=42, test_ratio=0.2).
    """
    samples = _discover_fb_samples_all(data_root)
    labels  = np.array([s["label"] for s in samples])
    rng     = np.random.default_rng(seed)

    c_idx = np.where(labels == 0)[0]
    b_idx = np.where(labels == 1)[0]
    rng.shuffle(c_idx)
    rng.shuffle(b_idx)

    nc_test = int(len(c_idx) * test_ratio)
    nb_test = int(len(b_idx) * test_ratio)

    test_idx  = np.concatenate([c_idx[:nc_test],  b_idx[:nb_test]])
    train_idx = np.concatenate([c_idx[nc_test:],  b_idx[nb_test:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return {
        "train": [samples[i] for i in train_idx],
        "test":  [samples[i] for i in test_idx],
    }


# ── Breaking Bad ──────────────────────────────────────────────────────────────

def _load_split_entries(split_dir: str, subset: str, split: str) -> list[str]:
    """Load object paths from official Breaking Bad split .txt files."""
    prefix   = subset.split("/")[0]
    filepath = os.path.join(split_dir, f"{prefix}.{split}.txt")
    if not os.path.exists(filepath):
        log.warning(f"Split file not found: {filepath}")
        return []
    with open(filepath) as f:
        entries = [l.strip() for l in f if l.strip()]
    if "/" in subset:
        entries = [e for e in entries if e.startswith(subset + "/")]
    return entries


def _discover_bb_for_entry(data_root: str, obj_entry: str) -> dict:
    """Return complete_path and broken_paths for one Breaking Bad object entry."""
    obj_dir     = os.path.join(data_root, obj_entry)
    complete    = os.path.join(obj_dir, "mode_0", "piece_0.obj")
    frac_dir    = os.path.join(obj_dir, "fractured_0")
    broken_list = []
    if os.path.isdir(frac_dir):
        broken_list = sorted(
            os.path.join(frac_dir, f)
            for f in os.listdir(frac_dir)
            if f.endswith(".obj")
        )
    return {
        "object_id":     obj_entry,
        "complete_path": complete if os.path.exists(complete) else None,
        "broken_paths":  broken_list,
    }


def _balance_bb(
    samples: list[dict], strategy: str, seed: int
) -> list[dict]:
    """Balance Breaking Bad classes (none / undersample / one_per_obj)."""
    if strategy == "none":
        return samples
    rng      = np.random.default_rng(seed)
    complete = [s for s in samples if s["label"] == 0]
    broken   = [s for s in samples if s["label"] == 1]

    if strategy == "one_per_obj":
        by_obj: dict[str, list[dict]] = defaultdict(list)
        for s in broken:
            parts  = s["id"].split("/")
            parent = "/".join(parts[:-2])
            by_obj[parent].append(s)
        bal_broken = [
            pieces[int(rng.integers(0, len(pieces)))]
            for pieces in by_obj.values()
        ]
        balanced = complete + bal_broken

    elif strategy == "undersample":
        nc, nb = len(complete), len(broken)
        if nb > nc:
            idx = rng.choice(nb, nc, replace=False)
            balanced = complete + [broken[i] for i in idx]
        elif nc > nb:
            idx = rng.choice(nc, nb, replace=False)
            balanced = [complete[i] for i in idx] + broken
        else:
            balanced = samples

    else:
        raise ValueError(f"Unknown balance strategy: {strategy!r}")

    rng.shuffle(balanced)
    return balanced


def discover_bb_splits(
    data_root: str,
    split_dir: str,
    subsets: list[str],
    seed: int = 42,
    balance: str = "undersample",
) -> dict[str, list[dict]]:
    """
    Build per-split sample lists for Breaking Bad, reproducing the same
    logic as prepare_breakingbad_cls.py so that object coverage is consistent.
    Split names in split files: 'train' and 'val'  →  mapped to 'train'/'test'.
    """
    result: dict[str, list[dict]] = {}

    for file_split, out_split in [("train", "train"), ("val", "test")]:
        raw: list[dict] = []
        for subset in subsets:
            for entry in _load_split_entries(split_dir, subset, file_split):
                info = _discover_bb_for_entry(data_root, entry)
                if info["complete_path"]:
                    raw.append({
                        "path": info["complete_path"],
                        "label": 0,
                        "id": f"{entry}/mode_0",
                    })
                for bp in info["broken_paths"]:
                    raw.append({
                        "path": bp,
                        "label": 1,
                        "id": f"{entry}/fractured_0/{os.path.basename(bp)}",
                    })

        result[out_split] = _balance_bb(raw, balance, seed)
        nc = sum(1 for s in result[out_split] if s["label"] == 0)
        nb = sum(1 for s in result[out_split] if s["label"] == 1)
        log.info(
            f"  {file_split!r} → {out_split!r}: "
            f"{len(result[out_split])} samples "
            f"(complete={nc}, broken={nb})"
        )

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Processing pipeline
# ═════════════════════════════════════════════════════════════════════════════

def process_split(
    samples: list[dict],
    num_points: int,
    seed: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Process all meshes in a split.
    Returns: points [B, N, 11] float32, labels [B, 1] int64, object_ids list.
    """
    all_feat:   list[np.ndarray] = []
    all_labels: list[int]        = []
    all_ids:    list[str]        = []
    failed                       = 0

    for i, s in enumerate(samples):
        label_str = "complete" if s["label"] == 0 else "broken"
        if (i % 20 == 0) or (i == len(samples) - 1):
            log.info(
                f"  [{i+1:4d}/{len(samples)}] "
                f"{label_str:8s}  {Path(s['path']).name}"
            )

        feat = process_mesh(s["path"], num_points, seed)
        if feat is None:
            failed += 1
            continue

        all_feat.append(feat)
        all_labels.append(s["label"])
        all_ids.append(s["id"])

    if failed:
        log.warning(f"  {failed} meshes failed to process in split={split_name!r}")

    points = np.stack(all_feat,   axis=0).astype(np.float32)  # (B, N, 11)
    labels = np.array(all_labels, dtype=np.int64).reshape(-1, 1)   # (B, 1)
    return points, labels, all_ids


def save_enriched_h5(
    filepath: str, data: np.ndarray, labels: np.ndarray
) -> None:
    """Save enriched feature array to compressed HDF5."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data",  data=data,   dtype="float32", compression="gzip")
        f.create_dataset("label", data=labels, dtype="int64",   compression="gzip")
    log.info(f"  Saved {filepath}  data={data.shape}  label={labels.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute geometric features for Breaking Bad / Fantastic Breaks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True,
                   choices=["fantastic_breaks", "breaking_bad"],
                   help="Which dataset to process.")
    p.add_argument("--data_root", required=True,
                   help="Root directory of the raw mesh dataset.")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for enriched HDF5 + metadata.")
    p.add_argument("--num_points", type=int, default=8192,
                   help="Points sampled per mesh. Default: 8192.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling + splitting. Default: 42.")

    # Breaking Bad specific
    p.add_argument("--split_dir", default=None,
                   help="[breaking_bad] Directory with *.train.txt / *.val.txt files.")
    p.add_argument("--subsets", nargs="+",
                   default=["artifact", "everyday/Vase",
                            "everyday/Mug", "everyday/Cup", "everyday/Plate"],
                   help="[breaking_bad] Subsets to include.")
    p.add_argument("--balance", default="undersample",
                   choices=["none", "undersample", "one_per_obj"],
                   help="[breaking_bad] Class balance strategy. Default: undersample.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths
    args.data_root  = os.path.abspath(args.data_root)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("=" * 70)
    log.info(f"  Dataset     : {args.dataset}")
    log.info(f"  Data root   : {args.data_root}")
    log.info(f"  Output dir  : {args.output_dir}")
    log.info(f"  Num points  : {args.num_points}")
    log.info(f"  Seed        : {args.seed}")
    log.info("=" * 70)

    # ── 1. Discover splits ────────────────────────────────────────────────────
    log.info("Discovering splits...")
    if args.dataset == "fantastic_breaks":
        splits = discover_fb_splits(args.data_root, seed=args.seed)
    else:
        if args.split_dir is None:
            raise ValueError("--split_dir is required for dataset=breaking_bad")
        args.split_dir = os.path.abspath(args.split_dir)
        splits = discover_bb_splits(
            args.data_root, args.split_dir, args.subsets,
            seed=args.seed, balance=args.balance,
        )

    for sp, slist in splits.items():
        nc = sum(1 for s in slist if s["label"] == 0)
        nb = sum(1 for s in slist if s["label"] == 1)
        log.info(f"  {sp:6s}: {len(slist):4d} samples  (complete={nc}, broken={nb})")

    # ── 2. Process TRAIN split (needed first for normalization stats) ──────────
    t0 = time.time()
    log.info("\nProcessing TRAIN split...")
    train_data, train_labels, train_ids = process_split(
        splits["train"], args.num_points, args.seed, "train"
    )
    log.info(f"  Done in {time.time()-t0:.1f}s — shape {train_data.shape}")

    # ── 3. Compute global normalization stats from train features ─────────────
    log.info("\nComputing normalization statistics from TRAIN split...")
    train_stats = compute_train_stats(train_data)
    for name, st in train_stats.items():
        log.info(
            f"  {name:15s}  mean={st['mean']:8.4f}  std={st['std']:8.4f}"
            f"  range=[{st['min']:.3f}, {st['max']:.3f}]"
        )

    # ── 4. Normalize TRAIN features ───────────────────────────────────────────
    train_data_norm = apply_normalization(train_data, train_stats)

    # ── 5. Process remaining splits ───────────────────────────────────────────
    other_splits: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {
        "train": (train_data_norm, train_labels, train_ids)
    }
    for split_name, sample_list in splits.items():
        if split_name == "train":
            continue
        log.info(f"\nProcessing {split_name.upper()} split...")
        t1 = time.time()
        data, labels, ids = process_split(
            sample_list, args.num_points, args.seed, split_name
        )
        data_norm = apply_normalization(data, train_stats)
        other_splits[split_name] = (data_norm, labels, ids)
        log.info(f"  Done in {time.time()-t1:.1f}s — shape {data.shape}")

    # ── 6. Save HDF5 files ────────────────────────────────────────────────────
    log.info("\nSaving enriched HDF5 files...")
    split_meta: dict[str, dict] = {}
    for split_name, (data_norm, labels, ids) in other_splits.items():
        # Use 'test' filename for non-train splits (FantasticBreaksCls expects
        # train_data.h5 / test_data.h5 naming)
        h5_name   = f"{split_name}_data_enriched.h5"
        h5_path   = os.path.join(args.output_dir, h5_name)
        save_enriched_h5(h5_path, data_norm, labels)

        # Also write a companion file list (mirrors baseline naming)
        fl_path = os.path.join(args.output_dir, f"{split_name}_files_enriched.txt")
        with open(fl_path, "w") as f:
            f.write(h5_path + "\n")

        # Save object IDs for traceability
        ids_path = os.path.join(
            args.output_dir, f"{split_name}_object_ids_enriched.txt"
        )
        with open(ids_path, "w") as f:
            f.write("\n".join(ids) + "\n")

        nc = int((labels == 0).sum())
        nb = int((labels == 1).sum())
        split_meta[split_name] = {
            "h5_file":      h5_path,
            "n_samples":    int(len(labels)),
            "n_complete":   nc,
            "n_broken":     nb,
            "points_shape": list(data_norm.shape),
        }

    # ── 7. Save feature_metadata.json ────────────────────────────────────────
    import datetime
    metadata = {
        "created":      datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset":      args.dataset,
        "num_points":   args.num_points,
        "seed":         args.seed,
        "n_total_dims": N_TOTAL_DIMS,
        "feature_names": FEATURE_NAMES,
        "normalization": {
            "type":         "z-score (mean/std from train split)",
            "clip_sigma":   5.0,
            "xyz_note":     "XYZ (cols 0-2) normalized per-sample to unit sphere; NOT z-scored.",
            "features":     train_stats,
        },
        "splits": split_meta,
        "processing": {
            "curvature_clip":   CURV_CLIP,
            "sav_max":          SAV_MAX,
            "knn_k":            KNN_K,
            "interp_method":    "barycentric (trimesh.triangles.points_to_barycentric)",
        },
    }

    meta_path = os.path.join(args.output_dir, "feature_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"\n  Metadata saved to: {meta_path}")

    log.info("\n" + "=" * 70)
    log.info("  DONE")
    log.info(f"  Output dir : {args.output_dir}")
    log.info(f"  Feature dim: {N_TOTAL_DIMS}  {FEATURE_NAMES}")
    log.info("=" * 70)

    log.info("\nNext steps:")
    log.info("  1. Update FantasticBreaksCls adapter to load *_data_enriched.h5")
    log.info("  2. Set in_channels: 11 in the YAML config files")
    log.info("  3. Retrain PointNeXt-B with enriched 11D features")


if __name__ == "__main__":
    main()
