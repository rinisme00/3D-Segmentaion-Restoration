"""
Purpose:
    Batch-process the entire Fantastic Breaks dataset to generate aligned meshes and
    3-class segmentation point clouds for every object listed in the metadata CSV.
    For each object it produces:
      - An aligned fragment mesh (.ply)
      - Point cloud files (.pts, .seg, .txt) with labels:
            0 = Intact (unbroken gray surface)
            1 = Crack / Fracture (red-painted break surface on the broken mesh)
            2 = Fragment (the separated green piece)
      - A QA scatter-plot image (.png) for visual verification
    A summary CSV is written at the end recording per-object status and class counts.

Usage:
    python scripts/generate_full_dataset.py \
        --metadata-csv  data/processed/metadata_dict/objects.csv \
        --aligned-mesh-dir data/processed/aligned_meshes \
        --pts-dir       data/processed/points \
        --seg-dir       data/processed/seg \
        --txt-dir       data/processed/txt \
        --qa-dir        data/processed/qa_plots \
        --summary-csv   data/processed/summary.csv \
        --only-complete-rows

    Pass --max-broken-points -1 --max-fragment-points -1 to skip subsampling
    and export all raw vertices.

Core Logic:
    1. Read the metadata CSV built by build_metadata.py.
    2. Optionally filter to rows where all required files exist (--only-complete-rows).
    3. For each object row, call process_one_object() which:
       a. Loads the broken mesh (model_b) and fragment mesh (model_r).
       b. Applies the un-normalizing 4x4 transform T to BOTH meshes equally
          so they remain spatially aligned in real-world coordinates.
       c. Derives per-vertex crack labels from the .npz mask (preferred)
          or falls back to an RGB color heuristic on the broken mesh.
       d. Assigns all fragment vertices the FRAGMENT_LABEL (2).
       e. Subsamples points with minority-class (crack) prioritization
          capped at 50% to preserve intact surface coverage.
       f. Saves .pts / .seg / .txt files and a QA scatter plot.
    4. Collects per-object results and writes a summary CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Dict

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

def process_one_object(
    row: pd.Series,
    aligned_mesh_dir: Path,
    pts_dir: Path,
    seg_dir: Path,
    txt_dir: Path,
    qa_dir: Path,
    transform_key: Optional[str],
    mask_key: Optional[str],
    max_broken_points: int,
    max_fragment_points: int,
    seed: int,
) -> Dict[str, object]:
    """
    Process a single object from the metadata DataFrame.

    Steps:
      1. Load the broken mesh (model_b) and fragment mesh (model_r).
      2. Apply the un-normalizing transform T to BOTH meshes so they share
         the same real-world coordinate system.
      3. Export the aligned fragment mesh as a .ply file.
      4. Derive per-vertex crack labels (mask → color fallback).
      5. Subsample with crack-priority capping at 50%.
      6. Save .pts / .seg / .txt and a QA scatter plot.

    Returns a dict with object_id, status ('ok' | 'failed'), error message,
    and per-class point counts.
    """
    object_id = str(row["object_id"])

    # Initialize result dict — updated on success, left with zeros on failure
    result: Dict[str, object] = {
        "object_id": object_id,
        "status": "ok",
        "error": "",
        "n_intact": 0,
        "n_crack": 0,
        "n_fragment": 0,
        "n_total": 0,
    }

    try:
        # Resolve absolute paths from the metadata row
        path_b = Path(row["path_b"])
        path_r = Path(row["path_r"])
        path_npz = Path(row["path_npz"])

        # Load the raw meshes from disk
        mesh_b = load_mesh(path_b)
        mesh_r = load_mesh(path_r)

        # Load the un-normalizing 4x4 transform from the .npz metadata
        T = find_transform(path_npz, transform_key=transform_key)

        # Apply T to BOTH meshes so broken body and fragment remain
        # spatially consistent in real-world (un-normalized) coordinates
        mesh_b_aligned = o3d.geometry.TriangleMesh(mesh_b)
        mesh_b_aligned.transform(T)
        mesh_b_aligned.compute_vertex_normals()

        mesh_r_aligned = o3d.geometry.TriangleMesh(mesh_r)
        mesh_r_aligned.transform(T)
        mesh_r_aligned.compute_vertex_normals()

        # Export the aligned fragment mesh for downstream use
        aligned_mesh_dir.mkdir(parents=True, exist_ok=True)
        aligned_mesh_path = aligned_mesh_dir / f"{object_id}_r_aligned.ply"
        o3d.io.write_triangle_mesh(str(aligned_mesh_path), mesh_r_aligned)

        # Extract XYZ vertex arrays from the aligned meshes
        broken_points = get_mesh_vertices(mesh_b_aligned)
        fragment_points = get_mesh_vertices(mesh_r_aligned)

        # Derive crack labels: prefer explicit mask from .npz, fall back to color heuristic
        broken_labels = labels_from_mask(path_npz, mask_key, len(broken_points))
        if broken_labels is None:
            broken_colors = get_vertex_colors(mesh_b_aligned)
            broken_labels = labels_from_vertex_colors(broken_colors, len(broken_points))

        # All fragment vertices are uniformly labeled as FRAGMENT_LABEL (2)
        fragment_labels = np.full(len(fragment_points), FRAGMENT_LABEL, dtype=np.int64)

        # Subsample: prioritize crack points (capped at 50%) to prevent starvation
        broken_points, broken_labels = random_subsample(
            broken_points, broken_labels, max_broken_points, seed=seed, prioritize_label=CRACK_LABEL
        )
        fragment_points, fragment_labels = random_subsample(
            fragment_points, fragment_labels, max_fragment_points, seed=seed
        )

        # Merge broken + fragment into one unified point cloud
        points = np.vstack([broken_points, fragment_points])
        labels = np.concatenate([broken_labels, fragment_labels])

        # Define output file paths per object (each file type goes to its own directory)
        pts_path = pts_dir / f"{object_id}.pts"
        seg_path = seg_dir / f"{object_id}.seg"
        txt_path = txt_dir / f"{object_id}.txt"
        qa_path = qa_dir / f"{object_id}_qa.png"

        # Write point cloud files (.pts, .seg, .txt) and generate a visual QA scatter plot
        save_pts_seg_txt(points, labels, pts_path=pts_path, seg_path=seg_path, txt_path=txt_path)
        save_qa_plot(points, labels, qa_path, f"Seg Sample QA — {object_id}")

        # Record per-class statistics for the summary CSV
        result["n_intact"] = int(np.sum(labels == INTACT_LABEL))
        result["n_crack"] = int(np.sum(labels == CRACK_LABEL))
        result["n_fragment"] = int(np.sum(labels == FRAGMENT_LABEL))
        result["n_total"] = int(len(labels))

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def main() -> None:
    """
    Entry point: parse CLI arguments, iterate over the metadata CSV,
    process each object, and write a summary CSV with per-object results.
    """
    # ── CLI argument definitions ──────────────────────────────────────
    parser = argparse.ArgumentParser(description="Generate aligned meshes and segmentation samples for full dataset.")
    parser.add_argument("--metadata-csv", type=Path, required=True,
                        help="Path to the metadata CSV built by build_metadata.py.")
    parser.add_argument("--aligned-mesh-dir", type=Path, required=True,
                        help="Directory to store aligned fragment .ply meshes.")
    parser.add_argument("--pts-dir", type=Path, required=True,
                        help="Directory for .pts point cloud output files.")
    parser.add_argument("--seg-dir", type=Path, required=True,
                        help="Directory for .seg label output files.")
    parser.add_argument("--txt-dir", type=Path, required=True,
                        help="Directory for combined .txt (XYZ + label) files.")
    parser.add_argument("--qa-dir", type=Path, required=True,
                        help="Directory for QA scatter-plot PNG images.")
    parser.add_argument("--summary-csv", type=Path, required=True,
                        help="Path to write the per-object summary CSV.")
    parser.add_argument("--transform-key", type=str, default=None,
                        help="Explicit key name for the transform in the .npz file.")
    parser.add_argument("--mask-key", type=str, default=None,
                        help="Explicit key name for the crack mask in the .npz file.")
    parser.add_argument("--max-broken-points", type=int, default=15000,
                        help="Max points to keep from the broken mesh. Use -1 for all.")
    parser.add_argument("--max-fragment-points", type=int, default=5000,
                        help="Max points to keep from the fragment mesh. Use -1 for all.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible subsampling.")
    parser.add_argument("--only-complete-rows", action="store_true",
                        help="Skip objects missing any required file (b, r, npz).")
    args = parser.parse_args()

    # Load the metadata dictionary and optionally filter incomplete rows
    df = pd.read_csv(args.metadata_csv)

    if args.only_complete_rows:
        df = df[df["has_all_files"] == True].copy()

    results: List[Dict[str, object]] = []  # accumulate per-object results

    print(f"Processing {len(df)} objects...")

    # ── Main processing loop ──────────────────────────────────────────
    for _, row in df.iterrows():
        object_id = str(row["object_id"])
        print(f"[INFO] Processing object: {object_id}")

        result = process_one_object(
            row=row,
            aligned_mesh_dir=args.aligned_mesh_dir,
            pts_dir=args.pts_dir,
            seg_dir=args.seg_dir,
            txt_dir=args.txt_dir,
            qa_dir=args.qa_dir,
            transform_key=args.transform_key,
            mask_key=args.mask_key,
            max_broken_points=args.max_broken_points,
            max_fragment_points=args.max_fragment_points,
            seed=args.seed,
        )
        results.append(result)

        # Print per-object status for live monitoring
        if result["status"] == "ok":
            print(
                f"  -> ok | total={result['n_total']} "
                f"(intact={result['n_intact']}, crack={result['n_crack']}, fragment={result['n_fragment']})"
            )
        else:
            print(f"  -> FAILED | {result['error']}")

    # ── Write the summary CSV and print final statistics ──────────────
    summary_df = pd.DataFrame(results)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False)

    n_ok = int((summary_df["status"] == "ok").sum())
    n_fail = int((summary_df["status"] == "failed").sum())

    print("\nDone.")
    print(f"Successful: {n_ok}")
    print(f"Failed: {n_fail}")
    print(f"Saved summary to: {args.summary_csv}")


if __name__ == "__main__":
    main()
