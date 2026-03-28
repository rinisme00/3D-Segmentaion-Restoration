import os
from copy import deepcopy
from pathlib import Path

from ..utils.compat import PROJECT_ROOT

LABEL_NAMES = ("Complete", "Broken")

BASE_CONFIG = {
    "data_dir": str(PROJECT_ROOT / "data" / "classification"),
    "mesh_data_dir": str(PROJECT_ROOT / "data" / "Fantastic_Breaks_v1"),
    "log_dir": str(PROJECT_ROOT / "src" / "results" / "pointnet_cls"),
    "model_name": "pointnet_cls",
    "gpu_index": 0,
    "num_point": None,
    "num_classes": 2,
    "batch_size": 16,
    "max_epoch": 200,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "momentum": 0.9,
    "decay_step": 200000,
    "decay_rate": 0.7,
    "reg_weight": 0.001,
    "class_weights": [1.0, 1.25],
    "use_class_weights": True,
    "train_limit": None,
    "test_limit": None,
    "eval_limit": None,
    "seed": 42,
    "rotate_augment": True,
    "jitter_augment": True,
    "scale_augment": True,
    "shift_augment": True,
    "dropout_augment": True,
    "point_dropout_ratio": 0.1,
    "checkpoint_metric": "broken_f1",
    "checkpoint_metric_mode": "max",
    "save_last_checkpoint": True,
    "classification_test_ratio": 0.20,
    "classification_split_seed": 42,
    "num_inference_samples": 5,
    "inference_plot_limit": 25000,
}

SMOKE_CONFIG = {
    "log_dir": "/tmp/pointnet_cls_smoke",
    "num_point": 1024,
    "max_epoch": 2,
    "train_limit": 64,
    "test_limit": 32,
    "eval_limit": 32,
    "num_inference_samples": 2,
    "save_last_checkpoint": False,
}


def build_config(run_mode=None, **overrides):
    resolved_mode = (
        run_mode
        or overrides.pop("run_mode", None)
        or os.environ.get("POINTNET_RUN_MODE", "full")
    )
    resolved_mode = str(resolved_mode).strip().lower()
    if resolved_mode not in {"full", "smoke"}:
        raise ValueError(
            "Unsupported run_mode={!r}. Expected 'full' or 'smoke'.".format(
                resolved_mode
            )
        )

    cfg = deepcopy(BASE_CONFIG)
    if resolved_mode == "smoke":
        cfg.update(SMOKE_CONFIG)

    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    cfg["run_mode"] = resolved_mode
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def print_config(cfg, log_fn=print):
    keys = [
        "run_mode",
        "data_dir",
        "mesh_data_dir",
        "log_dir",
        "model_name",
        "gpu_index",
        "num_point",
        "batch_size",
        "max_epoch",
        "learning_rate",
        "optimizer",
        "train_limit",
        "test_limit",
        "eval_limit",
        "checkpoint_metric",
        "seed",
    ]
    log_fn("PointNet classification configuration")
    for key in keys:
        log_fn("  {:20s}: {}".format(key, cfg.get(key)))
