"""
Purpose:
This module provides a PyTorch Dataset implementation for loading 3D point-cloud 
segmentation data. It reads point coordinates from `.pts` files and corresponding 
segmentation labels from `.seg` files. 

Usage:
Instantiate the `SegmentationDatasetTorch` by providing the directories for the 
`.pts` and `.seg` files. It can be directly passed to a `torch.utils.data.DataLoader` 
for training or evaluation loops.

Example:
    dataset = SegmentationDatasetTorch(
        pts_dir="/path/to/pts",
        seg_dir="/path/to/seg",
        num_points=8192,
        normalize=True
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDatasetTorch(Dataset):
    """
    Dataset for point-cloud segmentation from .pts and .seg files.

    Each sample returns a dictionary containing:
        - "points": FloatTensor of shape [N, 3] representing the 3D coordinates.
        - "labels": LongTensor of shape [N] representing the segmentation labels.
        - "object_id": str representing the unique identifier of the object.
    """

    def __init__(
        self,
        pts_dir: str | Path,
        seg_dir: str | Path,
        object_ids: Optional[list[str]] = None,
        num_points: Optional[int] = 8192,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        """
        Initializes the dataset object.

        Args:
            pts_dir: Directory containing the .pts files.
            seg_dir: Directory containing the .seg files.
            object_ids: Optional list of specific object IDs to include. If None, all 
                        matching .pts files in the directory are used.
            num_points: Target number of points to sample for each object.
            normalize: Whether to mean-center and scale the point clouds.
            transform: Optional Callable for applying data augmentation.
            seed: Random seed for reproducible point sampling.
        """
        self.pts_dir = Path(pts_dir)
        self.seg_dir = Path(seg_dir)
        self.num_points = num_points
        self.normalize = normalize
        self.transform = transform
        self.seed = seed

        # Core Logic: Discover available files or use provided subset.
        if object_ids is None:
            self.object_ids = sorted([p.stem for p in self.pts_dir.glob("*.pts")])
        else:
            self.object_ids = sorted(object_ids)

        if not self.object_ids:
            raise ValueError(f"No samples found in {self.pts_dir}")

        # Validate that necessary files exist.
        self._check_files()

    def _check_files(self) -> None:
        """
        Cross-checks both directories to ensure that every object ID has 
        both a .pts and a .seg file.
        """
        missing = []
        for object_id in self.object_ids:
            pts_path = self.pts_dir / f"{object_id}.pts"
            seg_path = self.seg_dir / f"{object_id}.seg"
            if not pts_path.exists() or not seg_path.exists():
                missing.append(object_id)

        if missing:
            raise FileNotFoundError(f"Missing pts/seg files for: {missing[:10]}")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.object_ids)

    def _load_sample(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the raw point coordinates and segmentation labels from disk.
        
        Args:
            object_id: The identifier for the sample to load.
        """
        pts_path = self.pts_dir / f"{object_id}.pts"
        seg_path = self.seg_dir / f"{object_id}.seg"

        # Load data as float32 for coords and int64 for labels.
        points = np.loadtxt(pts_path, dtype=np.float32)
        labels = np.loadtxt(seg_path, dtype=np.int64)

        # Handle edge cases where there's only one point/label
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if labels.ndim == 0:
            labels = labels.reshape(1)

        # Validate shapes and counts
        if points.shape[1] != 3:
            raise ValueError(f"{pts_path} must have shape [N, 3], got {points.shape}")
        if len(points) != len(labels):
            raise ValueError(
                f"Point/label count mismatch for {object_id}: "
                f"{len(points)} points vs {len(labels)} labels"
            )

        return points, labels

    def _sample_points(self, points: np.ndarray, labels: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Core Logic: Downsamples or upsamples the point cloud to a fixed number of points (`num_points`).
        Uses random choice without replacement if there are enough points, otherwise with replacement.
        """
        if self.num_points is None:
            return points, labels

        rng = np.random.default_rng(self.seed + idx)
        n = len(points)

        if n == self.num_points:
            return points, labels

        if n > self.num_points:
            # Downsample without replacement
            choice = rng.choice(n, size=self.num_points, replace=False)
        else:
            # Upsample with replacement
            choice = rng.choice(n, size=self.num_points, replace=True)

        return points[choice], labels[choice]

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        Core Logic: Normalizes the point cloud.
        1. Centers the points at the origin (0, 0, 0) by subtracting the centroid.
        2. Scales the points so that the maximum distance from the origin is 1.0.
        """
        centroid = points.mean(axis=0, keepdims=True)
        points = points - centroid

        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

        return points

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the preprocessed data dict for the specified index.
        
        This handles the complete pipeline: loading, sampling, normalizing, application 
        of custom transforms, and conversion to PyTorch tensors.
        """
        object_id = self.object_ids[idx]
        points, labels = self._load_sample(object_id)
        points, labels = self._sample_points(points, labels, idx)

        if self.normalize:
            points = self._normalize_points(points)

        if self.transform is not None:
            points = self.transform(points)

        return {
            "points": torch.from_numpy(points).float(),   # [N, 3]
            "labels": torch.from_numpy(labels).long(),    # [N]
            "object_id": object_id,
        }