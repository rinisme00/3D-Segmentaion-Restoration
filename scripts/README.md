# scripts/

Preprocessing, analysis, and diagnostic scripts for the 3D fracture classification
pipeline. All scripts are runnable from the repository root with:

```bash
conda activate ptv3
python scripts/<script_name>.py --help
```

---

## Dataset Preparation

| Script | Purpose |
|---|---|
| `prepare_classification_data.py` | Fantastic Breaks → HDF5 point clouds for the PointNet pipeline. Samples N points per mesh, normalizes, and writes `train_data.h5` / `test_data.h5`. |
| `prepare_breakingbad_cls.py` | Breaking Bad → HDF5 point clouds for the PointNeXt pipeline. Reads the official train/val split files, samples N points, optionally balances classes. |
| `compute_geometric_features.py` | Computes the 11D geometric feature set (XYZ + k1/k2/H/K + SA/V ratio + dist_centroid + local_density + boundary_dist) and writes enriched HDF5 files for both datasets. **Note:** the k1/k2/H/K and global scalar features are deprecated per the project report — they are computed here but will be re-engineered in a later task. |

## Dataset Analysis

| Script | Purpose |
|---|---|
| `analyze_breakingbad.py` | Reads Breaking Bad split files and samples a subset of objects for deep inspection. Reports class balance statistics and extrapolated classification strategies. Generates `bb_classification_metadata.json`. |
| `build_metadata.py` | Scans the Fantastic Breaks directory and builds a metadata CSV (`objects.csv`) listing all file paths per object (broken, complete, fragment, npz). Used by the segmentation pipeline. |

## Segmentation Pipeline

> These scripts support the future segmentation stage and are not part of the
> classification training pipeline.

| Script | Purpose |
|---|---|
| `generate_full_dataset.py` | Batch-processes all Fantastic Breaks objects from `objects.csv` to generate aligned meshes and 3-class segmentation point clouds (`.pts`, `.seg`, `.txt`). |
| `generate_one_seg_sample.py` | Generates a single segmentation sample for a given object ID. Useful for debugging and QA. |
| `compute_segmentation_stats.py` | Computes global and per-object class distribution statistics from `.pts`/`.seg` output files. |

## Diagnostics

| Script | Purpose |
|---|---|
| `sanity_check.py` | Interactive walkthrough: loads a broken mesh, reads the 4×4 transform from `.npz`, applies it, and applies the fracture mask. Good first-run validation for a new object. |
| `inspect_npz_schema.py` | Inspects all `.npz` metadata files listed in a metadata CSV and identifies transform and mask key candidates. |
| `preview_alignment.py` | Headless alignment preview: applies the un-normalizing transform to broken/fragment/complete meshes and saves Matplotlib or PyVista preview images. |

## Shared Utilities

| Module | Purpose |
|---|---|
| `utils/core.py` | Shared helpers used by segmentation scripts: `load_mesh`, `find_transform`, `labels_from_mask`, `labels_from_vertex_colors`, `random_subsample`, `save_pts_seg_txt`, `save_qa_plot`. |
