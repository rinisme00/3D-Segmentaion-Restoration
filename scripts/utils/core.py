from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Predefined semantic segmentation labels
INTACT_LABEL = 0
CRACK_LABEL = 1
FRAGMENT_LABEL = 2

# Common dictionary keys utilized for transformations in the .npz files
PREFERRED_TRANSFORM_KEYS = [
    "transform",
    "transforms",
    "T",
    "Rt",
    "pose",
    "matrix",
    "alignment",
]


def to_4x4(arr: np.ndarray) -> np.ndarray:
    """
    Standardize a 3x4 or 4x4 transformation matrix into a strict 4x4 homogeneous matrix.
    Raises an error if the array has an unexpected shape.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr
        return out
    raise ValueError(f"Unsupported transform shape: {arr.shape}")


def find_transform(npz_path: Path, transform_key: Optional[str] = None) -> np.ndarray:
    """
    Load a transformation matrix from an .npz file. It will search explicitly for `transform_key`
    if provided, otherwise it falls back to checking common key names defined in PREFERRED_TRANSFORM_KEYS,
    and finally examines any 2D/3D array with suitable dimensions.
    """
    data = np.load(npz_path, allow_pickle=True)

    # 1. Check an explicitly provided key
    if transform_key is not None:
        if transform_key not in data.files:
            raise KeyError(f"Key '{transform_key}' not found in {npz_path}")
        arr = np.asarray(data[transform_key])
        if arr.ndim == 3:
            arr = arr[0]
        return to_4x4(arr)

    # 2. Check common known transform key names
    for key in PREFERRED_TRANSFORM_KEYS:
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 3:
                arr = arr[0]
            try:
                return to_4x4(arr)
            except ValueError:
                pass

    # 3. Last resort: Inspect all arrays for appropriate shapes (3x4 or 4x4 matrix)
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape in {(4, 4), (3, 4)}:
            return to_4x4(arr)
        if arr.ndim == 3 and arr.shape[-2:] in {(4, 4), (3, 4)}:
            return to_4x4(arr[0])

    raise ValueError(f"No usable transform found in {npz_path}")


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """
    Load a 3D Triangle mesh from disk using Open3D and compute vertex normals.
    """
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Failed to read mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def get_mesh_vertices(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Extract the NumPy array of XYZ vertex coordinates from an Open3D mesh.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        raise ValueError("Mesh has no vertices.")
    return verts


def get_vertex_colors(mesh: o3d.geometry.TriangleMesh) -> Optional[np.ndarray]:
    """
    Extract the RGB colors [0.0, 1.0] from a mesh, returning None if colors are missing
    or do not match the vertex count.
    """
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if colors.shape[0] == len(mesh.vertices):
        return colors
    return None


def labels_from_vertex_colors(colors: Optional[np.ndarray], n_points: int) -> np.ndarray:
    """
    Determine class labels (0 or 1) strictly based on vertex RGB colors.
    Break surfaces (cracks) are indicated by strong red visual highlights.
    Returns an array of INTACT_LABEL (0) with CRACK_LABEL (1) applied where conditions are met.
    """
    labels = np.full(n_points, INTACT_LABEL, dtype=np.int64)

    if colors is None or len(colors) != n_points:
        return labels

    r = colors[:, 0]
    g = colors[:, 1]
    b = colors[:, 2]

    # Handle both [0, 1] and [0, 255] color scales
    if colors.max() > 1.0:
        r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Heuristic for detecting red painted labels on the broken mesh
    # Used if there's no precise boolean mask in the .npz file
    is_crack = (r > 0.6) & (g < 0.45) & (b < 0.45) & (r > g + 0.12) & (r > b + 0.12)
    labels[is_crack] = CRACK_LABEL
    return labels


def labels_from_mask(
    npz_path: Path,
    mask_key: Optional[str],
    n_broken_vertices: int,
) -> Optional[np.ndarray]:
    """
    Extract class labels using a boolean mask explicitly stored in an .npz file arrays.
    Validates array dimensions. Returns None as a fallback signal if missing or misaligned.
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Auto-discover mask key if None is provided
    if mask_key is None:
        if "mask" in data.files:
            mask_key = "mask"
        elif "labels" in data.files:
            mask_key = "labels"
        else:
            return None

    if mask_key not in data.files:
        raise KeyError(f"Mask key '{mask_key}' not found in {npz_path}")

    mask = np.asarray(data[mask_key]).squeeze()

    if mask.ndim != 1:
        print(f"[WARN] Mask key '{mask_key}' is not 1D after squeeze. Shape={mask.shape}. Fallback to color.")
        return None

    if len(mask) != n_broken_vertices:
        print(
            f"[WARN] Mask length {len(mask)} does not match broken vertex count {n_broken_vertices}. "
            "Fallback to color."
        )
        return None

    labels = np.full(n_broken_vertices, INTACT_LABEL, dtype=np.int64)
    labels[mask.astype(bool)] = CRACK_LABEL
    return labels


def random_subsample(points: np.ndarray, labels: np.ndarray, max_points: int, seed: int = 42, prioritize_label: Optional[int] = None):
    """
    Uniformly subsample points down to `max_points` to keep memory manageable and consistent.
    If `max_points` <= 0, the subsampling is bypassed entirely and the original arrays are returned.
    
    If `prioritize_label` is given (e.g. CRACK_LABEL=1), those specific points are kept unconditionally
    up to `max_points // 2`, ensuring minority classes are not erased by blind uniform sampling 
    while preserving sufficient room for majority intact labels.
    """
    if max_points <= 0 or len(points) <= max_points:
        return points, labels
        
    rng = np.random.default_rng(seed)
    
    # If we need to preserve a minority class like crack points (class 1)
    if prioritize_label is not None:
        priority_mask = (labels == prioritize_label)
        priority_idx = np.where(priority_mask)[0]
        other_idx = np.where(~priority_mask)[0]
        
        # We cap the priority inclusion at 50% of the maximum sample constraint to guarantee
        # the remaining points can capture the intact mesh structure.
        max_priority_points = max_points // 2
        
        if len(priority_idx) >= max_priority_points:
            selected_priority_idx = rng.choice(priority_idx, size=max_priority_points, replace=False)
        else:
            selected_priority_idx = priority_idx
            
        remaining = max_points - len(selected_priority_idx)
        
        # Fill the leftover quota with other points
        if len(other_idx) >= remaining:
            sub_other = rng.choice(other_idx, size=remaining, replace=False)
        else:
            sub_other = other_idx
            
        selected_idx = np.concatenate([selected_priority_idx, sub_other])
            
        # Shuffle the final selection to avoid clusters of priority items at the beginning
        rng.shuffle(selected_idx)
        return points[selected_idx], labels[selected_idx]
        
    # Standard uniform sampling, suitable for homogeneous sets (like the Fragment mesh)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], labels[idx]


def save_pts_seg_txt(
    points: np.ndarray,
    labels: np.ndarray,
    out_stem: Optional[Path] = None,
    pts_path: Optional[Path] = None,
    seg_path: Optional[Path] = None,
    txt_path: Optional[Path] = None,
) -> None:
    """
    Write 3 files:
    1. .pts: standard XYZ format.
    2. .seg: label index format matching lines in .pts
    3. .txt: comprehensive XYZ L space-delimited text.

    Can be called in two ways:
      - save_pts_seg_txt(points, labels, out_stem)  — derives .pts/.seg/.txt from stem
      - save_pts_seg_txt(points, labels, pts_path=..., seg_path=..., txt_path=...)
    """
    if out_stem is not None:
        pts_path = out_stem.with_suffix(".pts")
        seg_path = out_stem.with_suffix(".seg")
        txt_path = out_stem.with_suffix(".txt")

    if pts_path is None or seg_path is None or txt_path is None:
        raise ValueError("Provide either out_stem or all three of pts_path, seg_path, txt_path.")

    pts_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(pts_path, points, fmt="%.6f")
    np.savetxt(seg_path, labels, fmt="%d")
    np.savetxt(txt_path, np.column_stack([points, labels]), fmt=["%.6f", "%.6f", "%.6f", "%d"])


def save_qa_plot(points: np.ndarray, labels: np.ndarray, out_png: Path, title: str) -> None:
    """
    Render three 2D scatter plots (XY, XZ, YZ view panes) assigning colors to the three target semantic classes
    to verify alignment visually in QA scripts.
    """
    color_map = {
        INTACT_LABEL: "lightgray",
        CRACK_LABEL: "red",
        FRAGMENT_LABEL: "green",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=14)

    views = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]

    for ax, (name, i, j) in zip(axes, views):
        for label_value in [INTACT_LABEL, CRACK_LABEL, FRAGMENT_LABEL]:
            mask = labels == label_value
            if np.any(mask):
                ax.scatter(
                    points[mask, i],
                    points[mask, j],
                    s=0.5,
                    alpha=0.7,
                    c=color_map[label_value],
                    label=f"class_{label_value}" if name == "XY" else None,
                    rasterized=True,
                )
        ax.set_title(name)
        ax.set_xlabel(["X", "Y", "Z"][i])
        ax.set_ylabel(["X", "Y", "Z"][j])
        ax.set_aspect("equal", adjustable="box")

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
