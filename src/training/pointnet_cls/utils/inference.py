from pathlib import Path
import os

# Headless-safe matplotlib cache directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from ..model import create_session_config, tf1
from .compat import PROJECT_ROOT
from .data import load_object_ids, reconstruct_classification_split
from .metrics import softmax_np


def normalize_point_cloud_np(points):
    """Center and unit-normalize a point cloud array shaped [N, 3]."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Expected a non-empty [N, 3] point cloud, got {}".format(points.shape))
    centered = points.copy()
    centered -= centered.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered /= scale
    return centered


def prepare_inference_batch(point_cloud_np, num_point, seed=0):
    """Resample/pad one point cloud to exactly `num_point` points."""
    points = np.asarray(point_cloud_np, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Expected a non-empty [N, 3] point cloud, got {}".format(points.shape))

    rng = np.random.default_rng(seed)
    target_points = int(num_point)
    if len(points) >= target_points:
        indices = rng.choice(len(points), size=target_points, replace=False)
        processed = points[indices].copy()
    else:
        extra_indices = rng.choice(len(points), size=target_points - len(points), replace=True)
        processed = np.concatenate([points, points[extra_indices]], axis=0)

    processed = normalize_point_cloud_np(processed)
    return processed[np.newaxis, ...].astype(np.float32), processed.astype(np.float32)


def validate_reconstructed_split(dataset, train_idx, test_idx, labels):
    """Guard against object-id split mismatch versus loaded H5 splits."""
    train_full_indices = np.asarray(
        dataset.get("train_full_indices", np.arange(len(dataset["train_label"]))),
        dtype=np.int32,
    )
    test_full_indices = np.asarray(
        dataset.get("test_full_indices", np.arange(len(dataset["test_label"]))),
        dtype=np.int32,
    )

    expected_train_labels = labels[train_idx][train_full_indices]
    expected_test_labels = labels[test_idx][test_full_indices]
    if not np.array_equal(expected_train_labels, dataset["train_label"]):
        raise ValueError("Reconstructed train split does not match loaded train labels.")
    if not np.array_equal(expected_test_labels, dataset["test_label"]):
        raise ValueError("Reconstructed test split does not match loaded test labels.")


def resolve_mesh_path(object_id, data_root=None):
    """Map an object id like `09006_c` to the corresponding mesh file path."""
    data_root = Path(data_root or PROJECT_ROOT / "data" / "Fantastic_Breaks_v1")
    if not data_root.exists():
        raise FileNotFoundError("Could not locate mesh root at {}".format(data_root))

    base_id, suffix = object_id.rsplit("_", 1)
    mesh_name = "model_c.ply" if suffix == "c" else "model_b_0.ply"
    for category_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        candidate = category_dir / base_id / mesh_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find mesh for object_id={}".format(object_id))


def sample_mesh_for_visualization(mesh, max_points=25000, seed=0):
    """Sample mesh surface points for a dense visualization reference cloud."""
    if len(mesh.faces) > 0:
        sampled_points, _ = trimesh.sample.sample_surface(mesh, count=max_points, seed=seed)
        return normalize_point_cloud_np(sampled_points.astype(np.float32))

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0:
        raise ValueError("Mesh has no vertices for visualization.")
    if len(vertices) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(vertices), size=max_points, replace=False)
        vertices = vertices[keep]
    return normalize_point_cloud_np(vertices)


def set_axes_equal(axis, points, margin=0.05):
    """Set equal XYZ scales so point clouds are not visually distorted."""
    points = np.asarray(points, dtype=np.float32)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) / 2.0, 1e-3)
    radius *= 1.0 + margin

    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")


def plot_inference_comparison(
    processed_points,
    mesh_points,
    object_id,
    true_label,
    pred_label,
    probabilities,
    label_names,
    sample_label,
):
    """Create side-by-side plot: model input cloud vs matched mesh cloud."""
    figure, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, 7), subplot_kw={"projection": "3d"}
    )

    ax_left.scatter(
        processed_points[:, 0],
        processed_points[:, 1],
        processed_points[:, 2],
        c=processed_points[:, 2],
        cmap="viridis",
        s=3,
        alpha=0.9,
        linewidths=0,
    )
    ax_left.set_title(
        "Processed PointNet input\n{} -> {}\npred={} | p(broken)={:.3f}".format(
            sample_label, object_id, label_names[pred_label], probabilities[1]
        )
    )
    set_axes_equal(ax_left, processed_points)
    ax_left.view_init(elev=20, azim=45)

    ax_right.scatter(
        mesh_points[:, 0],
        mesh_points[:, 1],
        mesh_points[:, 2],
        c=mesh_points[:, 2],
        cmap="plasma",
        s=1.5,
        alpha=0.7,
        linewidths=0,
    )
    ax_right.set_title(
        "Matched mesh surface\nobject_id={} | true={}".format(
            object_id, label_names[true_label]
        )
    )
    set_axes_equal(ax_right, mesh_points)
    ax_right.view_init(elev=20, azim=45)

    figure.suptitle("PointNet inference vs. mesh reference", y=0.98)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    return figure


def resolve_test_object_ids(dataset, cfg):
    """Return test object IDs aligned to the loaded evaluation split."""
    test_object_ids = np.asarray(dataset.get("test_ids", []), dtype=object)
    if len(test_object_ids) == len(dataset["test_data"]):
        return test_object_ids

    object_ids = load_object_ids(Path(cfg["data_dir"]) / "object_ids.txt")
    train_idx, test_idx, labels = reconstruct_classification_split(
        object_ids,
        test_ratio=cfg["classification_test_ratio"],
        seed=cfg["classification_split_seed"],
    )
    validate_reconstructed_split(dataset, train_idx, test_idx, labels)

    full_test_object_ids = np.asarray(object_ids, dtype=object)[test_idx]
    test_full_indices = np.asarray(
        dataset.get("test_full_indices", np.arange(len(dataset["test_label"]))),
        dtype=np.int32,
    )
    return full_test_object_ids[test_full_indices]


def resolve_inference_input(dataset, cfg, test_object_ids, test_sample_pos=0, input_object_id=None):
    """Resolve either a dataset sample or a manually requested object id."""
    if input_object_id is not None:
        matches = np.where(test_object_ids == input_object_id)[0]
        if len(matches) > 0:
            sample_pos = int(matches[0])
            return {
                "object_id": str(test_object_ids[sample_pos]),
                "true_label": int(dataset["test_label"][sample_pos]),
                "raw_cloud": dataset["test_data"][sample_pos],
                "test_sample_pos": sample_pos,
                "source_desc": "{} [sample {}]".format(
                    Path(cfg["data_dir"]) / "test_data.h5", sample_pos
                ),
                "mesh_path": None,
                "mesh": None,
            }

        mesh_path = resolve_mesh_path(input_object_id, data_root=cfg.get("mesh_data_dir"))
        mesh = trimesh.load(mesh_path, force="mesh")
        sampled_points, _ = trimesh.sample.sample_surface(
            mesh,
            cfg["num_point"],
            seed=cfg["classification_split_seed"],
        )
        return {
            "object_id": input_object_id,
            "true_label": 0 if input_object_id.endswith("_c") else 1,
            "raw_cloud": sampled_points.astype(np.float32),
            "test_sample_pos": None,
            "source_desc": "mesh sampling ({})".format(mesh_path),
            "mesh_path": mesh_path,
            "mesh": mesh,
        }

    sample_pos = int(test_sample_pos)
    if sample_pos < 0 or sample_pos >= len(dataset["test_data"]):
        raise IndexError(
            "test_sample_pos={} out of range for {} test samples.".format(
                sample_pos, len(dataset["test_data"])
            )
        )

    return {
        "object_id": str(test_object_ids[sample_pos]),
        "true_label": int(dataset["test_label"][sample_pos]),
        "raw_cloud": dataset["test_data"][sample_pos],
        "test_sample_pos": sample_pos,
        "source_desc": "{} [sample {}]".format(
            Path(cfg["data_dir"]) / "test_data.h5", sample_pos
        ),
        "mesh_path": None,
        "mesh": None,
    }


def run_visual_inference(
    handles,
    dataset,
    cfg,
    checkpoint_path,
    label_names,
    test_sample_pos=0,
    input_object_id=None,
):
    """Run inference for one sample/object and return figure + metadata."""
    test_object_ids = resolve_test_object_ids(dataset, cfg)
    resolved = resolve_inference_input(
        dataset,
        cfg,
        test_object_ids,
        test_sample_pos=test_sample_pos,
        input_object_id=input_object_id,
    )

    batch_input, processed_points = prepare_inference_batch(
        resolved["raw_cloud"],
        num_point=cfg["num_point"],
        seed=cfg["seed"],
    )

    checkpoint_path = str(checkpoint_path)
    if not Path(checkpoint_path + ".index").exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    with tf1.Session(graph=handles["graph"], config=create_session_config()) as sess:
        handles["saver"].restore(sess, checkpoint_path)
        logits = sess.run(
            handles["pred"],
            feed_dict={
                handles["pointclouds_pl"]: batch_input,
                handles["labels_pl"]: np.zeros(batch_input.shape[0], dtype=np.int32),
                handles["is_training_pl"]: False,
            },
        )[0]

    probabilities = softmax_np(logits[np.newaxis, :])[0]
    pred_label = int(np.argmax(probabilities))

    mesh = resolved["mesh"]
    mesh_path = resolved["mesh_path"]
    if mesh is None:
        mesh_path = resolve_mesh_path(
            resolved["object_id"], data_root=cfg.get("mesh_data_dir")
        )
        mesh = trimesh.load(mesh_path, force="mesh")

    mesh_points = sample_mesh_for_visualization(
        mesh,
        max_points=cfg["inference_plot_limit"],
        seed=cfg["classification_split_seed"],
    )
    sample_label = (
        "test sample {}".format(resolved["test_sample_pos"])
        if resolved["test_sample_pos"] is not None
        else "manual mesh input"
    )

    figure = plot_inference_comparison(
        processed_points,
        mesh_points,
        resolved["object_id"],
        resolved["true_label"],
        pred_label,
        probabilities,
        label_names,
        sample_label,
    )
    return {
        "object_id": resolved["object_id"],
        "test_sample_pos": resolved["test_sample_pos"],
        "true_label": resolved["true_label"],
        "pred_label": pred_label,
        "probs": probabilities,
        "mesh_path": Path(mesh_path),
        "source_desc": resolved["source_desc"],
        "figure": figure,
    }


def run_random_test_inference(
    handles,
    dataset,
    cfg,
    checkpoint_path,
    label_names,
    num_samples=5,
    seed=None,
):
    """Run visual inference on random test indices for quick qualitative checks."""
    total_test = len(dataset["test_data"])
    if total_test == 0:
        raise ValueError("Loaded dataset has no test samples.")
    if int(num_samples) <= 0:
        return []

    sample_count = max(1, min(int(num_samples), total_test))
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    sampled_indices = np.sort(rng.choice(total_test, size=sample_count, replace=False))

    results = []
    for sample_index in sampled_indices:
        results.append(
            run_visual_inference(
                handles,
                dataset,
                cfg,
                checkpoint_path=checkpoint_path,
                label_names=label_names,
                test_sample_pos=int(sample_index),
            )
        )
    return results
