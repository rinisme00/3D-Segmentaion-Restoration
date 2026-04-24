#!/usr/bin/env python3
"""
compute_geometric_features.py
=================================
Compute 8 geometric features per-point from raw meshes and generate enriched HDF5 files
with input shape [B, N, 11] for PointNeXt.

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
    # Step 0: always build manifests first (object-disjoint splits)
    python scripts/build_manifests.py

    # Fantastic Breaks
    python scripts/compute_geometric_features.py \\
        --dataset fantastic_breaks \\
        --data_root data/Fantastic_Breaks_v1 \\
        --output_dir data/fb_classification \\
        --num_points 8192 --seed 42

    # Breaking Bad
    python scripts/compute_geometric_features.py \\
        --dataset breaking_bad \\
        --data_root data/BreakingBad \\
        --output_dir data/bb_classification \\
        --num_points 8192 --balance undersample --seed 42

    # Override manifest path explicitly
    python scripts/compute_geometric_features.py \\
        --dataset fantastic_breaks \\
        --data_root data/Fantastic_Breaks_v1 \\
        --manifest data/manifests/fantastic_breaks_classification.csv \\
        --output_dir data/fb_classification \\
        --num_points 8192
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

# Make src/ importable so we can use the canonical manifest validator
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.manifests.builder import validate_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature constants — 9D default (Config B per report §6.6) ────────────────
#
#   cols 0-2 │ x, y, z            │ XYZ normalized to unit sphere
#   cols 3-5 │ nx, ny, nz         │ estimated surface normals (covariance eigenvector)
#   col  6   │ local_density      │ mean distance to k-NN
#   col  7   │ surface_variation  │ λ_min / (λ1+λ2+λ3+ε)  — smoothness measure
#   col  8   │ eigenentropy       │ −Σ λ̄ᵢ ln(λ̄ᵢ)  — neighbourhood disorder
#
FEATURE_NAMES = [
    "x", "y", "z",          # 0-2
    "nx", "ny", "nz",       # 3-5
    "local_density",         # 6
    "surface_variation",     # 7
    "eigenentropy",          # 8
]
N_TOTAL_DIMS = len(FEATURE_NAMES)   # 9
KNN_K        = 16                   # neighbours for local geometry


# ═════════════════════════════════════════════════════════════════════════════
# Core: single-mesh processing
# ═════════════════════════════════════════════════════════════════════════════

def _compute_local_geometry(
    pts: np.ndarray,   # (N, 3) float64, already normalized
    k: int = KNN_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single KNN pass + vectorized local covariance eigendecomposition.
    Returns (normals, local_density, surface_variation, eigenentropy) all float32.

    normals          (N, 3)  — eigenvector of smallest eigenvalue (raw, pre-orientation)
    local_density    (N,)    — mean distance to k-NN
    surface_variation(N,)    — λ_min / (λ1+λ2+λ3+ε)
    eigenentropy     (N,)    — −Σ λ̄ᵢ ln(λ̄ᵢ+ε)
    """
    from scipy.spatial import cKDTree

    N = len(pts)
    k_actual = min(k, N - 1)
    eps = 1e-8

    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=k_actual + 1)  # (N, k+1); col 0 = self
    dists = dists[:, 1:]   # (N, k)
    idxs  = idxs[:, 1:]   # (N, k)

    # local_density
    local_density = dists.mean(axis=1).astype(np.float32)  # (N,)

    # Gather neighbour coordinates: (N, k, 3)
    neighbors = pts[idxs]                                   # (N, k, 3)
    centroids = neighbors.mean(axis=1, keepdims=True)       # (N, 1, 3)
    centered  = neighbors - centroids                       # (N, k, 3)

    # Batch 3×3 covariance matrices: (N, 3, 3)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k_actual

    # Batch eigendecomposition — eigh guarantees ascending order, numerically stable
    evals, evecs = np.linalg.eigh(cov)   # evals (N,3), evecs (N,3,3)
    evals = np.clip(evals, 0.0, None)    # numerical safety

    # Normals = column 0 of evecs (eigenvector of smallest eigenvalue)
    normals = evecs[:, :, 0].astype(np.float32)  # (N, 3)

    # Surface variation: λ_min / (λ1+λ2+λ3+ε)
    lsum = evals.sum(axis=1) + eps          # (N,)
    surface_variation = (evals[:, 0] / lsum).astype(np.float32)   # (N,)

    # Eigenentropy: −Σ λ̄ᵢ ln(λ̄ᵢ+ε)
    lbar = np.clip(evals / lsum[:, None], eps, None)               # (N, 3)
    eigenentropy = (-np.sum(lbar * np.log(lbar), axis=1)).astype(np.float32)  # (N,)

    return normals, local_density, surface_variation, eigenentropy


def _orient_normals(
    normals: np.ndarray,   # (N, 3) float32 — raw covariance normals
    tm,                    # trimesh mesh
    face_ids: np.ndarray,  # (N,) face indices from surface sampling
) -> np.ndarray:
    """
    Flip each normal so it agrees with the sampled face normal.
    Uses face normals as reference (robust, always available from trimesh).
    """
    ref = tm.face_normals[face_ids].astype(np.float32)  # (N, 3)
    dot = (normals * ref).sum(axis=1)                    # (N,)
    signs = np.where(dot < 0, -1.0, 1.0).astype(np.float32)
    return normals * signs[:, None]


def process_mesh(
    mesh_path: str,
    num_points: int,
    seed: int,
) -> "np.ndarray | None":
    """
    Load one mesh and return a [num_points, 9] float32 feature array.

    Pipeline:
      1. Load mesh with trimesh.
      2. Sample N surface points (uniform).
      3. Normalize XYZ to unit sphere (center + max-radius scale).
      4. Run _compute_local_geometry() — single KNN + batch eigendecomp:
           → normals, local_density, surface_variation, eigenentropy
      5. Orient normals using face normals as reference.
      6. Stack into [N, 9] and return.

    Returns None on any failure (degenerate mesh, load error, etc.).
    """
    import trimesh

    try:
        # ── 1. Load ───────────────────────────────────────────────────────────
        tm = trimesh.load(mesh_path, force="mesh", process=False)
        if tm.vertices.shape[0] < 4 or len(tm.faces) < 4:
            raise ValueError(
                f"Degenerate mesh: {tm.vertices.shape[0]} verts / {len(tm.faces)} faces"
            )

        # ── 2. Sample N surface points ────────────────────────────────────────
        raw_pts, face_ids = trimesh.sample.sample_surface(
            tm, num_points, seed=seed
        )
        raw_pts = raw_pts.astype(np.float64)

        # ── 3. Per-sample XYZ normalization (center + unit sphere) ────────────
        centroid = raw_pts.mean(axis=0)          # (3,)
        shifted  = raw_pts - centroid             # (N, 3)
        scale    = np.max(np.linalg.norm(shifted, axis=1))
        if scale < 1e-8:
            raise ValueError(f"Near-zero bounding radius: {scale:.2e}")
        pts_norm = (shifted / scale)              # (N, 3) float64

        # ── 4. Local geometry ─────────────────────────────────────────────────
        normals, local_density, surface_variation, eigenentropy = \
            _compute_local_geometry(pts_norm, k=KNN_K)

        # ── 5. Orient normals ─────────────────────────────────────────────────
        normals = _orient_normals(normals, tm, face_ids)

        # ── 6. Assemble [N, 9] ────────────────────────────────────────────────
        pts_f32 = pts_norm.astype(np.float32)
        features = np.stack([
            pts_f32[:, 0],     # x
            pts_f32[:, 1],     # y
            pts_f32[:, 2],     # z
            normals[:, 0],     # nx
            normals[:, 1],     # ny
            normals[:, 2],     # nz
            local_density,     # local_density
            surface_variation, # surface_variation
            eigenentropy,      # eigenentropy
        ], axis=1)             # (N, 9)

        return features.astype(np.float32)

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

# ── Manifest-driven split loading (canonical path) ───────────────────────────

def _manifest_to_samples(
    manifest_path: str,
    dataset_name: str,
    balance: str = "none",
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Load a manifest CSV and return a dict of {split_name: [sample_dict]}.

    This is the canonical source of truth for split assignment.
    Running validate_manifest() ensures object-disjoint integrity before
    any feature extraction begins.

    Each sample dict has:
        path  : absolute path to the mesh file
        label : 0 = complete, 1 = broken
        id    : traceability string (base_object_id/variant_id)
    """
    df = pd.read_csv(manifest_path)
    log.info(f"  Loaded manifest: {manifest_path} ({len(df)} rows)")

    # Validate object-disjoint integrity — abort on leakage
    try:
        validate_manifest(df)
        log.info("  Manifest validation: ✅ object-disjoint splits confirmed")
    except (ValueError, FileNotFoundError) as exc:
        log.error(f"  Manifest validation FAILED: {exc}")
        raise

    result: dict[str, list[dict]] = {}
    for split_name, grp in df.groupby("split"):
        if split_name in ("unknown", "None", None):
            continue
        samples = [
            {
                "path":  row["file_path_mesh"],
                "label": int(row["label"]),
                "id":    f"{row['base_object_id']}/{row['variant_id']}",
            }
            for _, row in grp.iterrows()
            if row["file_path_mesh"] and isinstance(row["file_path_mesh"], str)
        ]
        if balance != "none" and dataset_name == "breaking_bad":
            samples = _balance_bb(samples, balance, seed)
        nc = sum(1 for s in samples if s["label"] == 0)
        nb = sum(1 for s in samples if s["label"] == 1)
        log.info(f"  {split_name:8s}: {len(samples):5d} samples  (complete={nc}, broken={nb})")
        result[str(split_name)] = samples

    return result


# ── Breaking Bad: balance helper (still used by manifest path) ────────────────

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

def _default_manifest_path(dataset: str, project_root: str) -> str:
    """Return the canonical manifest path for the given dataset."""
    name_map = {
        "fantastic_breaks": "fantastic_breaks_classification.csv",
        "breaking_bad":     "breaking_bad_classification.csv",
    }
    return os.path.join(project_root, "data", "manifests", name_map[dataset])


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
    p.add_argument(
        "--manifest", default=None,
        help="Path to manifest CSV (single source of truth for splits). "
             "If not provided, auto-detected from data/manifests/{dataset}_classification.csv. "
             "Run scripts/build_manifests.py first to generate the manifest.",
    )

    # Breaking Bad specific
    p.add_argument("--balance", default="undersample",
                   choices=["none", "undersample", "one_per_obj"],
                   help="Class balance strategy for Breaking Bad. Default: undersample.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths
    project_root    = str(Path(__file__).resolve().parents[1])
    args.data_root  = os.path.abspath(args.data_root)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve manifest path
    if args.manifest is None:
        args.manifest = _default_manifest_path(args.dataset, project_root)
    args.manifest = os.path.abspath(args.manifest)

    log.info("=" * 70)
    log.info(f"  Dataset     : {args.dataset}")
    log.info(f"  Data root   : {args.data_root}")
    log.info(f"  Manifest    : {args.manifest}")
    log.info(f"  Output dir  : {args.output_dir}")
    log.info(f"  Num points  : {args.num_points}")
    log.info(f"  Seed        : {args.seed}")
    log.info("=" * 70)

    # ── 1. Load splits from canonical manifest ────────────────────────────────
    log.info("Loading splits from manifest...")
    if not os.path.exists(args.manifest):
        raise FileNotFoundError(
            f"Manifest not found: {args.manifest}\n"
            "Run 'python scripts/build_manifests.py' first to generate it."
        )
    splits = _manifest_to_samples(
        args.manifest, args.dataset, balance=args.balance, seed=args.seed
    )

    if not splits:
        raise RuntimeError(
            "No usable splits found in manifest. "
            "Check that build_manifests.py was run and split column is not 'unknown'."
        )
    log.info(f"  Splits found: {sorted(splits.keys())}")

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
        "created":       datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset":       args.dataset,
        "manifest":      args.manifest,
        "num_points":    args.num_points,
        "seed":          args.seed,
        "feature_config": "default (9D: XYZ + normals + local_density + surface_variation + eigenentropy)",
        "n_total_dims":  N_TOTAL_DIMS,
        "feature_names": FEATURE_NAMES,
        "normalization": {
            "type":      "z-score (mean/std from train split)",
            "clip_sigma": 5.0,
            "xyz_note":  "XYZ (cols 0-2) normalized per-sample to unit sphere; NOT z-scored.",
            "normals_note": "Normals (cols 3-5) z-scored; effectively a direction normalization.",
            "features":  train_stats,
        },
        "splits": split_meta,
        "processing": {
            "knn_k":        KNN_K,
            "normal_method": "local covariance eigenvector (smallest eigenvalue), oriented by face normal",
            "surface_variation": "lambda_min / (lambda_sum + 1e-8)",
            "eigenentropy":      "-sum(lambda_bar * log(lambda_bar + 1e-8))",
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
    log.info("  1. Set in_channels: 9 in PointNeXt YAML configs")
    log.info("  2. Re-run training with the new 9D enriched HDF5 files")
    log.info("  3. Compare vs 3D baseline (--feature_preset xyz) in ablation")


if __name__ == "__main__":
    main()
