"""
Purpose:
This script generates a single Point Cloud segmentation sample from a 3D broken object and its corresponding restored fragment. 
It loads the broken mesh, restored mesh, transformation matrix, and crack labels to combine them into one unified point cloud 
that contains 3 classes: Intact (0), Crack/Fracture (1), and Fragment (2). It outputs `.txt`, `.pts`, and `.seg` files.

Usage:
    python scripts/generate_one_seg_sample.py \
        --metadata-csv /path/to/metadata.csv \
        --object-id <OBJECT_ID_OR_COMPOSITE_ID> \
        --out-dir /path/to/output_dir

Core Logic:
1. Load broken mesh (`_b`) and the specific fragment mesh (`_r`).
2. Apply the necessary 4x4 coordinate transform to BOTH the chunk (`_r`) and broken body (`_b`) to align them correctly if they aren't already.
3. Extract vertex labels for the broken piece: mask arrays in `.npz` take precedence. If unavailable, fall back to RGB color heuristics.
4. Downsample to a fixed maximum number of points, specifically prioritizing rare minority classes (like Crack/Fracture points) so they aren't lost to random subsampling, capping at 50% to ensure Intact points are also represented.
5. Combine the meshes and labels, save the raw point lists, and output an image plot for QA (Quality Assurance).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from utils import (
    INTACT_LABEL,
    CRACK_LABEL,
    FRAGMENT_LABEL,
    load_mesh,
    get_mesh_vertices,
    get_vertex_colors,
    labels_from_vertex_colors,
    labels_from_mask,
    find_transform,
    random_subsample,
    save_pts_seg_txt,
    save_qa_plot,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one headless-safe segmentation sample.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--object-id", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--transform-key", type=str, default=None)
    parser.add_argument("--mask-key", type=str, default=None)
    parser.add_argument("--max-broken-points", type=int, default=15000)
    parser.add_argument("--max-fragment-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Access metadata database to fetch file locations corresponding to `object_id`
    df = pd.read_csv(args.metadata_csv, dtype={"object_id": str})
    row = df[df["object_id"] == args.object_id]
    if row.empty:
        raise ValueError(f"object_id '{args.object_id}' not found in metadata.")
    row = row.iloc[0]

    path_b = Path(row["path_b"])
    path_r = Path(row["path_r"])
    path_npz = Path(row["path_npz"])

    # Load broken and restored fragment meshes
    mesh_b = load_mesh(path_b)
    mesh_r = load_mesh(path_r)

    # Note: Fantastic Breaks dataset meshes are ALREADY aligned in a bounding box 
    # of size [-0.5, 0.5]^3. The 'transform' matrix in the npz file is actually an 
    # un-normalizing transform (a uniform scaling factor + translation) to revert 
    # the normalized mesh back to its true original size from ShapeNet.
    # We must apply this transform to ALL meshes equally so they stay relative to each other!
    T = find_transform(path_npz, transform_key=args.transform_key)
    
    mesh_b_aligned = o3d.geometry.TriangleMesh(mesh_b)
    mesh_b_aligned.transform(T)
    mesh_b_aligned.compute_vertex_normals()
    
    mesh_r_aligned = o3d.geometry.TriangleMesh(mesh_r)
    mesh_r_aligned.transform(T)
    mesh_r_aligned.compute_vertex_normals()

    broken_points = get_mesh_vertices(mesh_b_aligned)
    fragment_points = get_mesh_vertices(mesh_r_aligned)

    # Classify crack vertices on the broken mesh 
    # Labels must be derived relative to the broken points array
    broken_labels = labels_from_mask(path_npz, args.mask_key, len(broken_points))
    if broken_labels is None:
        broken_colors = get_vertex_colors(mesh_b_aligned) # Colors pull from alignment load
        broken_labels = labels_from_vertex_colors(broken_colors, len(broken_points))

    # All fragment points are classified blindly as FRAGMENT_LABEL
    fragment_labels = np.full(len(fragment_points), FRAGMENT_LABEL, dtype=np.int64)

    # Subsample matrices for manageable point cloud counts
    # The broken mesh prioritization avoids downsampling our relatively scarce fracture points
    broken_points, broken_labels = random_subsample(
        broken_points, broken_labels, args.max_broken_points, seed=args.seed, prioritize_label=CRACK_LABEL
    )
    fragment_points, fragment_labels = random_subsample(
        fragment_points, fragment_labels, args.max_fragment_points, seed=args.seed
    )

    # Concatenate final points arrays uniformly
    points = np.vstack([broken_points, fragment_points])
    labels = np.concatenate([broken_labels, fragment_labels])

    out_stem = args.out_dir / args.object_id
    save_pts_seg_txt(points, labels, out_stem)
    save_qa_plot(points, labels, args.out_dir / f"{args.object_id}_qa.png", f"Seg Sample QA — {args.object_id}")

    unique, counts = np.unique(labels, return_counts=True)
    stats = {int(k): int(v) for k, v in zip(unique, counts)}
    print("Label distribution:", stats)
    print("Done.")

if __name__ == "__main__":
    main()