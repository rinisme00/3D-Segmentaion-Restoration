"""
Purpose:
This script provides a headless alignment preview for 3D meshes and point clouds. It loads a 
broken object mesh, a fragment mesh, a complete mesh, and a transformation scaling matrix from an `.npz` file.
Since the meshes in Fantastic Breaks are already aligned in a normalized generic `[-0.5, 0.5]` space,
it applies the scaling transform back to all parts equally to restore their original real-world coordinates, 
and visualizes the aligned results using Matplotlib or PyVista.

Usage:
The script is intended to be run from the command line, providing a metadata CSV and an object ID.
Example:
    python scripts/preview_alignment.py --metadata-csv metadata.csv --object-id obj_000** \
                                        --export-dir ./output --preview-backend both

Code Logic:
1. Parse command-line arguments to get the input paths (from the CSV) and export settings.
2. Load the transformation matrix for the specified object.
3. Load the 3D meshes (broken, fragment, and complete).
4. Apply the un-normalizing transformation matrix to all meshes to restore their original scale.
5. Convert the aligned meshes into point clouds by uniform sampling to efficiently render them.
6. Merge the point clouds and export them as `.ply` files.
7. Generate and save headless preview images using Matplotlib (2D side views) and/or PyVista (3D renders).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d 
import pandas as pd 

from utils import find_transform, load_mesh

# Matplotlib for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyvista as pv 
    HAS_PV = True
except Exception:
    HAS_PV = False 


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh, n_points: int) -> o3d.geometry.PointCloud:
    """
    Converts a TriangleMesh into a PointCloud by uniformly sampling a specified number of points.
    This creates an efficient representation for visualization.
    """
    return mesh.sample_points_uniformly(number_of_points=n_points)


def merge_point_clouds(*pcds: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Merges multiple PointCloud objects into a single PointCloud.
    It combines the points and colors from all input clouds. If a cloud lacks colors, 
    it defaults to black (zeros) for those points.
    """
    points = []
    colors = []

    for pcd in pcds:
        pts = np.asarray(pcd.points)
        points.append(pts)

        if pcd.has_colors():
            cols = np.asarray(pcd.colors)
        else:
            cols = np.zeros((len(pts), 3), dtype=np.float64)
        colors.append(cols)

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(points))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    return merged


def save_matplotlib_preview(
    broken_pcd: o3d.geometry.PointCloud,
    fragment_pcd: o3d.geometry.PointCloud,
    complete_pcd: o3d.geometry.PointCloud,
    out_png: Path,
    title: str,
    max_points_per_cloud: int = 15000,
) -> None:
    """
    Generates a 2D multi-view (XY, XZ, YZ) scatter plot preview of the point clouds using Matplotlib.
    The points are randomly subsampled up to `max_points_per_cloud` to avoid memory issues and overplotting.
    """
    def sample_np(pcd: o3d.geometry.PointCloud, max_points: int) -> np.ndarray:
        pts = np.asarray(pcd.points)
        if len(pts) <= max_points:
            return pts
        idx = np.random.choice(len(pts), size=max_points, replace=False)
        return pts[idx]

    b = sample_np(broken_pcd, max_points_per_cloud)
    r = sample_np(fragment_pcd, max_points_per_cloud)
    c = sample_np(complete_pcd, max_points_per_cloud)

    views = [
        ("XY view", 0, 1),
        ("XZ view", 0, 2),
        ("YZ view", 1, 2),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=14)

    for ax, (name, i, j) in zip(axes, views):
        ax.scatter(c[:, i], c[:, j], s=0.2, alpha=0.20, label="complete", rasterized=True)
        ax.scatter(b[:, i], b[:, j], s=0.2, alpha=0.60, label="broken", rasterized=True)
        ax.scatter(r[:, i], r[:, j], s=0.5, alpha=0.80, label="fragment_aligned", rasterized=True)
        ax.set_title(name)
        ax.set_xlabel(["X", "Y", "Z"][i])
        ax.set_ylabel(["X", "Y", "Z"][j])
        ax.set_aspect("equal", adjustable="box")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Matplotlib preview: {out_png}")


def save_pyvista_preview(
    path_b: Path,
    path_r_aligned: Path,
    path_c: Path,
    out_png: Path,
) -> None:
    """
    Generates a continuous 3D rendering preview using PyVista (if installed).
    It loads the broken, aligned fragment, and complete meshes and renders them 
    with different colors and opacities off-screen.
    """
    if not HAS_PV:
        print("PyVista not installed; skipping PyVista screenshot.")
        return

    broken = pv.read(str(path_b))
    fragment = pv.read(str(path_r_aligned))
    complete = pv.read(str(path_c))

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(broken, color="lightgray", opacity=1.0, label="broken")
    plotter.add_mesh(fragment, color="green", opacity=0.95, label="fragment_aligned")
    plotter.add_mesh(complete, color="lightblue", opacity=0.18, label="complete")
    plotter.add_legend()
    plotter.show(screenshot=str(out_png))
    print(f"Saved PyVista preview: {out_png}")


def main() -> None:
    """
    Core logic of the script. 
    It parses CLI arguments, retrieves file paths from the metadata CSV, aligns the fragment mesh, 
    exports the aligned/merged point clouds as PLY files, and generates visual previews.
    """
    # Parse command-line configurations
    parser = argparse.ArgumentParser(description="Headless alignment preview: export PLY + PNG.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--object-id", type=str, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--transform-key", type=str, default=None)
    parser.add_argument("--sample-points-broken", type=int, default=30000)
    parser.add_argument("--sample-points-fragment", type=int, default=12000)
    parser.add_argument("--sample-points-complete", type=int, default=30000)
    parser.add_argument(
        "--preview-backend",
        type=str,
        default="matplotlib",
        choices=["matplotlib", "pyvista", "both", "none"],
    )
    args = parser.parse_args()

    # Load metadata and find the row corresponding to the object_id
    df = pd.read_csv(args.metadata_csv, dtype={"object_id": str})
    row = df[df["object_id"] == args.object_id]
    if row.empty:
        raise ValueError(f"object_id '{args.object_id}' not found in metadata")
    row = row.iloc[0]

    # Extract file paths from the selected row
    path_b = Path(row["path_b"])
    path_r = Path(row["path_r"])
    path_c = Path(row["path_c"])
    path_npz = Path(row["path_npz"])

    # Find and load the transformation matrix to align the fragment
    T = find_transform(path_npz, transform_key=args.transform_key)
    print("Using transform:")
    print(T)

    # Load broken, fragment, and complete meshes
    mesh_b = load_mesh(path_b)
    mesh_r_aligned = load_mesh(path_r)
    mesh_c = load_mesh(path_c)

    # Note: Fantastic Breaks dataset meshes are ALREADY aligned in a bounding box 
    # of size [-0.5, 0.5]^3. The 'transform' matrix in the npz file is actually an 
    # un-normalizing transform (a uniform scaling factor + translation) to revert 
    # the normalized mesh back to its true original size from ShapeNet.
    # We must apply this transform to ALL meshes equally!
    
    mesh_b.transform(T)
    mesh_b.compute_vertex_normals()
    
    mesh_r_aligned.transform(T)
    mesh_r_aligned.compute_vertex_normals()
    
    mesh_c.transform(T)
    mesh_c.compute_vertex_normals()

    # Convert all meshes to point clouds by uniform sampling
    pcd_b = mesh_to_point_cloud(mesh_b, args.sample_points_broken)
    pcd_r = mesh_to_point_cloud(mesh_r_aligned, args.sample_points_fragment)
    pcd_c = mesh_to_point_cloud(mesh_c, args.sample_points_complete)

    # Merge point clouds for combined visualization and export
    merged_br = merge_point_clouds(pcd_b, pcd_r)
    merged_brc = merge_point_clouds(pcd_b, pcd_r, pcd_c)

    # Ensure export directory exists
    args.export_dir.mkdir(parents=True, exist_ok=True)

    # Define export paths for the point clouds and aligned fragment
    aligned_fragment_path = args.export_dir / f"{args.object_id}_aligned_fragment.ply"
    broken_fragment_path = args.export_dir / f"{args.object_id}_broken_fragment.ply"
    broken_fragment_complete_path = args.export_dir / f"{args.object_id}_broken_fragment_complete.ply"

    # Save the aligned meshes and combined point clouds to disk as PLY files
    o3d.io.write_triangle_mesh(str(aligned_fragment_path), mesh_r_aligned)
    o3d.io.write_point_cloud(str(broken_fragment_path), merged_br)
    o3d.io.write_point_cloud(str(broken_fragment_complete_path), merged_brc)

    print(f"Saved: {aligned_fragment_path}")
    print(f"Saved: {broken_fragment_path}")
    print(f"Saved: {broken_fragment_complete_path}")

    # Generate Matplotlib preview if requested
    if args.preview_backend in {"matplotlib", "both"}:
        save_matplotlib_preview(
            broken_pcd=pcd_b,
            fragment_pcd=pcd_r,
            complete_pcd=pcd_c,
            out_png=args.export_dir / f"{args.object_id}_preview_matplotlib.png",
            title=f"Alignment Preview — {args.object_id}",
        )

    # Generate PyVista preview if requested
    if args.preview_backend in {"pyvista", "both"}:
        save_pyvista_preview(
            path_b=path_b,
            path_r_aligned=aligned_fragment_path,
            path_c=path_c,
            out_png=args.export_dir / f"{args.object_id}_preview_pyvista.png",
        )


if __name__ == "__main__":
    main()