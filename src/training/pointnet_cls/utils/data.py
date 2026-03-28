from pathlib import Path

import h5py
import numpy as np


def load_h5(path):
    with h5py.File(path, "r") as handle:
        data = handle["data"][:]
        labels = handle["label"][:].reshape(-1).astype(np.int32)
    return data, labels


def apply_sample_limit(data, labels, limit, seed):
    if limit is None or int(limit) >= len(data):
        indices = np.arange(len(data), dtype=np.int32)
        return data, labels, indices

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=int(limit), replace=False)
    indices.sort()
    return data[indices], labels[indices], indices.astype(np.int32)


def load_object_ids(object_ids_path):
    object_ids = [
        line.strip()
        for line in Path(object_ids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not object_ids:
        raise ValueError("No object IDs found in {}".format(object_ids_path))
    return object_ids


def reconstruct_classification_split(object_ids, test_ratio, seed):
    labels = np.array(
        [0 if object_id.endswith("_c") else 1 for object_id in object_ids], dtype=np.int32
    )
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    complete_indices = indices[labels[indices] == 0]
    broken_indices = indices[labels[indices] == 1]

    n_test_complete = int(len(complete_indices) * test_ratio)
    n_test_broken = int(len(broken_indices) * test_ratio)

    test_indices = np.concatenate(
        [complete_indices[:n_test_complete], broken_indices[:n_test_broken]]
    )
    train_indices = np.concatenate(
        [complete_indices[n_test_complete:], broken_indices[n_test_broken:]]
    )

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices, labels


def attach_split_object_ids(cfg, dataset, train_full_label, test_full_label, log_fn=print):
    object_ids_path = Path(cfg["data_dir"]) / "object_ids.txt"
    if not object_ids_path.exists():
        return dataset

    object_ids = np.asarray(load_object_ids(object_ids_path), dtype=object)
    expected_total = len(train_full_label) + len(test_full_label)
    if len(object_ids) != expected_total:
        log_fn(
            "Skipping object ID attachment because {} has {} IDs but dataset expects {}.".format(
                object_ids_path, len(object_ids), expected_total
            )
        )
        return dataset

    train_idx, test_idx, labels = reconstruct_classification_split(
        object_ids,
        test_ratio=cfg["classification_test_ratio"],
        seed=cfg["classification_split_seed"],
    )

    train_full_indices = np.asarray(dataset["train_full_indices"], dtype=np.int32)
    test_full_indices = np.asarray(dataset["test_full_indices"], dtype=np.int32)
    expected_train_labels = labels[train_idx][train_full_indices]
    expected_test_labels = labels[test_idx][test_full_indices]

    if not np.array_equal(expected_train_labels, dataset["train_label"]):
        log_fn(
            "Skipping train object ID attachment because reconstructed labels do not match the loaded train split."
        )
        return dataset
    if not np.array_equal(expected_test_labels, dataset["test_label"]):
        log_fn(
            "Skipping test object ID attachment because reconstructed labels do not match the loaded test split."
        )
        return dataset

    dataset = dict(dataset)
    dataset["object_ids"] = object_ids
    dataset["train_ids"] = np.asarray(object_ids[train_idx][train_full_indices], dtype=object)
    dataset["test_ids"] = np.asarray(object_ids[test_idx][test_full_indices], dtype=object)
    return dataset


def summarize_split(name, labels, log_fn=print):
    labels = np.asarray(labels).reshape(-1)
    complete = int(np.sum(labels == 0))
    broken = int(np.sum(labels == 1))
    log_fn(
        "{:<10s} n={:4d} | complete={:4d} | broken={:4d}".format(
            name, len(labels), complete, broken
        )
    )


def load_datasets(cfg, log_fn=print):
    train_path = Path(cfg["data_dir"]) / "train_data.h5"
    test_path = Path(cfg["data_dir"]) / "test_data.h5"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing classification H5 files in {}.".format(cfg["data_dir"])
        )

    train_full_data, train_full_label = load_h5(train_path)
    test_full_data, test_full_label = load_h5(test_path)

    cfg = dict(cfg)
    if cfg["num_point"] is None:
        cfg["num_point"] = int(train_full_data.shape[1])
    if cfg["num_point"] > train_full_data.shape[1]:
        raise ValueError(
            "num_point={} exceeds available points per cloud ({})".format(
                cfg["num_point"], train_full_data.shape[1]
            )
        )

    train_data, train_label, train_full_indices = apply_sample_limit(
        train_full_data,
        train_full_label,
        cfg["train_limit"],
        cfg["seed"],
    )
    test_data, test_label, test_full_indices = apply_sample_limit(
        test_full_data,
        test_full_label,
        cfg["test_limit"],
        cfg["seed"] + 1,
    )

    dataset = {
        "train_data": train_data,
        "train_label": train_label,
        "test_data": test_data,
        "test_label": test_label,
        "train_full_indices": train_full_indices,
        "test_full_indices": test_full_indices,
        "train_total_count": int(len(train_full_label)),
        "test_total_count": int(len(test_full_label)),
    }
    dataset = attach_split_object_ids(
        cfg,
        dataset,
        train_full_label=train_full_label,
        test_full_label=test_full_label,
        log_fn=log_fn,
    )

    log_fn(
        "Using num_point={} from data shape {}".format(
            cfg["num_point"], train_full_data.shape[1]
        )
    )
    summarize_split("Train", train_label, log_fn=log_fn)
    summarize_split("Validation", test_label, log_fn=log_fn)
    if "test_ids" in dataset:
        log_fn("Attached {} validation object IDs.".format(len(dataset["test_ids"])))
    return cfg, dataset
