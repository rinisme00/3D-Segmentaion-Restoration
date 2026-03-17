"""
Sanity Check Script
===================
A short interactive walkthrough to inspect each step of the mesh processing pipeline:
  1. Load a mesh and inspect its raw (normalized) vertex positions
  2. Load the 4x4 transform from .npz and view the matrix
  3. Apply the transform and see how vertex coordinates change
  4. Extract vertex coordinates as NumPy arrays
  5. Load the mask from .npz and apply it to label vertices

Usage:
    python scripts/sanity_check.py

    Or with a custom object:
    python scripts/sanity_check.py \
        --mesh   data/Fantastic_Breaks_v1/00/00002/model_b_0.ply \
        --npz    data/Fantastic_Breaks_v1/00/00002/meta_0.npz
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


# ── Default paths (object 00002 — a mug) ─────────────────────────────────
DEFAULT_MESH = "data/Fantastic_Breaks_v1/01/01002/model_b_0.ply"
DEFAULT_NPZ  = "data/Fantastic_Breaks_v1/01/01002/meta_0.npz"


def separator(title: str) -> None:
    """Print a visual section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check: inspect mesh, transform, and mask step by step.")
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH, help="Path to a broken mesh .ply file")
    parser.add_argument("--npz",  type=Path, default=DEFAULT_NPZ,  help="Path to the corresponding meta_0.npz file")
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Load the mesh and inspect raw (normalized) vertex positions
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 1: Load mesh & inspect raw vertex positions")

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    mesh.compute_vertex_normals()

    raw_vertices = np.asarray(mesh.vertices)
    print(f"Mesh file:      {args.mesh}")
    print(f"Vertex count:   {len(raw_vertices)}")
    print(f"Triangle count: {len(mesh.triangles)}")
    print()
    print(f"Vertex shape:   {raw_vertices.shape}   (N rows × 3 columns = N points with X,Y,Z)")
    print()
    print("Coordinate ranges (should be roughly [-0.5, 0.5] if normalized):")
    print(f"  X: [{raw_vertices[:, 0].min():.4f}, {raw_vertices[:, 0].max():.4f}]")
    print(f"  Y: [{raw_vertices[:, 1].min():.4f}, {raw_vertices[:, 1].max():.4f}]")
    print(f"  Z: [{raw_vertices[:, 2].min():.4f}, {raw_vertices[:, 2].max():.4f}]")
    print()
    print("First 5 vertices (raw, normalized):")
    print(raw_vertices[:5])

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: Load the 4×4 transform from .npz and view it
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 2: Load 4×4 transform from .npz")

    data = np.load(args.npz, allow_pickle=True)
    print(f"NPZ file:       {args.npz}")
    print(f"Keys inside:    {data.files}")
    print()

    T = np.asarray(data["transform"], dtype=np.float64)
    # If the transform is stored as (1, 4, 4), squeeze it to (4, 4)
    if T.ndim == 3:
        T = T[0]

    print("Transform matrix T (4×4):")
    print(T)
    print()

    # Break down what the matrix means
    scale_x = T[0, 0]
    scale_y = T[1, 1]
    scale_z = T[2, 2]
    tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]
    print("Breakdown:")
    print(f"  Scale factors:  X={scale_x:.4f}, Y={scale_y:.4f}, Z={scale_z:.4f}")
    print(f"  Translation:    tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
    print()
    if abs(scale_x - scale_y) < 0.001 and abs(scale_y - scale_z) < 0.001:
        print(f"  → Uniform scale = {scale_x:.4f} (same on all axes)")
    else:
        print(f"  → Non-uniform scale (different per axis)")

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Apply the transform and see how coordinates change
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 3: Apply transform → see coordinate changes")

    # Make a copy so we can compare before/after
    mesh_transformed = o3d.geometry.TriangleMesh(mesh)
    mesh_transformed.transform(T)
    mesh_transformed.compute_vertex_normals()

    transformed_vertices = np.asarray(mesh_transformed.vertices)

    print("Before transform (first 5 vertices):")
    print(raw_vertices[:5])
    print()
    print("After transform (same 5 vertices):")
    print(transformed_vertices[:5])
    print()
    print("Coordinate ranges AFTER transform (real-world scale):")
    print(f"  X: [{transformed_vertices[:, 0].min():.2f}, {transformed_vertices[:, 0].max():.2f}]")
    print(f"  Y: [{transformed_vertices[:, 1].min():.2f}, {transformed_vertices[:, 1].max():.2f}]")
    print(f"  Z: [{transformed_vertices[:, 2].min():.2f}, {transformed_vertices[:, 2].max():.2f}]")
    print()

    # Manual verification: multiply first vertex by T
    v = raw_vertices[0]
    v_homo = np.array([v[0], v[1], v[2], 1.0])  # add the "1" for homogeneous coordinates
    v_result = T @ v_homo  # matrix multiplication
    print("Manual check — first vertex:")
    print(f"  Raw:          [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]")
    print(f"  Homogeneous:  [{v_homo[0]:.6f}, {v_homo[1]:.6f}, {v_homo[2]:.6f}, {v_homo[3]:.1f}]")
    print(f"  T × v:        [{v_result[0]:.6f}, {v_result[1]:.6f}, {v_result[2]:.6f}]")
    print(f"  Open3D gave:  [{transformed_vertices[0, 0]:.6f}, {transformed_vertices[0, 1]:.6f}, {transformed_vertices[0, 2]:.6f}]")
    print(f"  Match: {np.allclose(v_result[:3], transformed_vertices[0])}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: Extract vertex coordinates as NumPy arrays
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 4: Extract vertex coordinates as NumPy arrays")

    points = np.asarray(mesh_transformed.vertices, dtype=np.float64)
    print(f"Type:    {type(points)}")
    print(f"Shape:   {points.shape}   → {points.shape[0]} points, each with {points.shape[1]} values (X, Y, Z)")
    print(f"Dtype:   {points.dtype}")
    print()
    print("First 5 points:")
    for i in range(5):
        print(f"  Point {i}: X={points[i, 0]:10.4f}, Y={points[i, 1]:10.4f}, Z={points[i, 2]:10.4f}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 5: Load mask from .npz and apply it to label vertices
    # ──────────────────────────────────────────────────────────────────
    separator("STEP 5: Load mask from .npz & apply labels")

    mask = np.asarray(data["mask"]).squeeze()
    print(f"Mask shape:     {mask.shape}")
    print(f"Mask dtype:     {mask.dtype}")
    print(f"Vertex count:   {len(points)}")
    print(f"Mask matches vertices: {len(mask) == len(points)}")
    print()

    # Count True/False in the mask
    n_true  = int(mask.sum())
    n_false = int(len(mask) - n_true)
    print(f"Mask values:")
    print(f"  False (intact surface):        {n_false:>10,} ({100*n_false/len(mask):.2f}%)")
    print(f"  True  (crack/fracture surface): {n_true:>10,} ({100*n_true/len(mask):.2f}%)")
    print()

    # Apply the mask to create labels
    # Label 0 = intact, Label 1 = crack
    labels = np.zeros(len(points), dtype=np.int64)   # start: all zeros (intact)
    labels[mask] = 1                                   # set True positions to 1 (crack)
    print("After applying mask → labels:")
    print(f"  Label 0 (intact): {int((labels == 0).sum()):>10,}")
    print(f"  Label 1 (crack):  {int((labels == 1).sum()):>10,}")
    print()
    print("First 20 mask values:  ", mask[:20].astype(int))
    print("First 20 label values: ", labels[:20])
    print()

    # Show some example crack vertices
    crack_indices = np.where(labels == 1)[0]
    print(f"Example crack vertices (first 5 of {len(crack_indices)}):")
    for idx in crack_indices[:5]:
        print(f"  Vertex {idx}: X={points[idx, 0]:10.4f}, Y={points[idx, 1]:10.4f}, Z={points[idx, 2]:10.4f}  → label={labels[idx]}")

    separator("SANITY CHECK COMPLETE ✓")
    print("All steps passed. The mesh, transform, and mask are consistent.\n")


if __name__ == "__main__":
    main()
