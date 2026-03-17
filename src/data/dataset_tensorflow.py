"""
Purpose:
This module provides a TensorFlow Dataset implementation for loading 3D point-cloud 
segmentation data. It mirrors the PyTorch dataset logic but outputs a tf.data.Dataset.

Usage:
Use the `create_tf_dataset` function to generate a tf.data.Dataset.
    
Example:
    dataset = create_tf_dataset(
        pts_dir="/path/to/pts",
        seg_dir="/path/to/seg",
        num_points=8192,
        normalize=True
    )
    dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import tensorflow as tf


class TFDatasetGenerator:
    """
    Generator class used internally to read files and yield samples.
    Contains the core logic for loading, sampling, and normalizing point clouds.
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
        self.pts_dir = Path(pts_dir)
        self.seg_dir = Path(seg_dir)
        self.num_points = num_points
        self.normalize = normalize
        self.transform = transform
        self.seed = seed

        if object_ids is None:
            self.object_ids = sorted([p.stem for p in self.pts_dir.glob("*.pts")])
        else:
            self.object_ids = sorted(object_ids)

        if not self.object_ids:
            raise ValueError(f"No samples found in {self.pts_dir}")

        self._check_files()

    def _check_files(self) -> None:
        """Cross-checks directories to ensure every object ID has .pts and .seg files."""
        missing = []
        for object_id in self.object_ids:
            pts_path = self.pts_dir / f"{object_id}.pts"
            seg_path = self.seg_dir / f"{object_id}.seg"
            if not pts_path.exists() or not seg_path.exists():
                missing.append(object_id)

        if missing:
            raise FileNotFoundError(f"Missing pts/seg files for: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.object_ids)

    def _load_sample(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Loads points and labels from disk as numpy arrays."""
        pts_path = self.pts_dir / f"{object_id}.pts"
        seg_path = self.seg_dir / f"{object_id}.seg"

        points = np.loadtxt(pts_path, dtype=np.float32)
        labels = np.loadtxt(seg_path, dtype=np.int64)

        if points.ndim == 1:
            points = points.reshape(1, -1)
        if labels.ndim == 0:
            labels = labels.reshape(1)

        if points.shape[1] != 3:
            raise ValueError(f"{pts_path} must have shape [N, 3], got {points.shape}")
        if len(points) != len(labels):
            raise ValueError(
                f"Count mismatch for {object_id}: {len(points)} pts vs {len(labels)} labels"
            )

        return points, labels

    def _sample_points(self, points: np.ndarray, labels: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Downsamples/Upsamples to a fixed number of points (`num_points`)."""
        if self.num_points is None:
            return points, labels

        rng = np.random.default_rng(self.seed + idx)
        n = len(points)

        if n == self.num_points:
            return points, labels

        if n > self.num_points:
            choice = rng.choice(n, size=self.num_points, replace=False)
        else:
            choice = rng.choice(n, size=self.num_points, replace=True)

        return points[choice], labels[choice]

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Centers points at origin and scales max distance to 1.0."""
        centroid = points.mean(axis=0, keepdims=True)
        points = points - centroid

        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

        return points

    def __call__(self):
        """
        Generator yielding individual samples. This enables seamless 
        integration with tf.data.Dataset.from_generator.
        """
        for idx, object_id in enumerate(self.object_ids):
            points, labels = self._load_sample(object_id)
            points, labels = self._sample_points(points, labels, idx)

            if self.normalize:
                points = self._normalize_points(points)

            if self.transform is not None:
                points = self.transform(points)

            yield {
                "points": points,
                "labels": labels,
                "object_id": object_id,
            }

def create_tf_dataset(
    pts_dir: str | Path,
    seg_dir: str | Path,
    object_ids: Optional[list[str]] = None,
    num_points: Optional[int] = 8192,
    normalize: bool = True,
    transform: Optional[Callable] = None,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Creates and returns a `tf.data.Dataset` for point cloud segmentation.

    Args:
        pts_dir: Directory containing the .pts files.
        seg_dir: Directory containing the .seg files.
        object_ids: Optional list of specific object IDs to include.
        num_points: Target number of points to sample for each object.
        normalize: Whether to mean-center and scale the point clouds.
        transform: Optional Callable for applying data augmentation/transform.
        seed: Random seed for reproducible point sampling.

    Returns:
        A tf.data.Dataset instance that yields dictionaries with 'points', 
        'labels', and 'object_id'.
    """
    generator = TFDatasetGenerator(
        pts_dir=pts_dir,
        seg_dir=seg_dir,
        object_ids=object_ids,
        num_points=num_points,
        normalize=normalize,
        transform=transform,
        seed=seed,
    )

    output_signature = {
        "points": tf.TensorSpec(shape=(num_points, 3) if num_points else (None, 3), dtype=tf.float32),
        "labels": tf.TensorSpec(shape=(num_points,) if num_points else (None,), dtype=tf.int64),
        "object_id": tf.TensorSpec(shape=(), dtype=tf.string),
    }

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    return dataset
