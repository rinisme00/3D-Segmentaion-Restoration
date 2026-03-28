"""Build a TF1 graph for PointNet classification using external model files.

This module keeps training/eval code decoupled from the model definition by
loading `src/pointnet-master/models/<model_name>.py` at runtime.
"""

import argparse
import importlib.util
import inspect
import sys
from functools import lru_cache
from pathlib import Path

# Support both `python -m pointnet_cls.model` and direct script execution.
if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from pointnet_cls.utils.compat import (
        POINTNET_ROOT,
        create_session_config as _create_session_config,
        ensure_pointnet_paths,
        load_tensorflow,
        resolve_device_name,
    )
else:
    from .utils.compat import (
        POINTNET_ROOT,
        create_session_config as _create_session_config,
        ensure_pointnet_paths,
        load_tensorflow,
        resolve_device_name,
    )

ensure_pointnet_paths()
tf, tf1 = load_tensorflow()


@lru_cache(maxsize=None)
def load_model_module(model_name):
    """Load and cache the external PointNet model module by filename."""
    model_path = POINTNET_ROOT / "models" / "{}.py".format(model_name)
    if not model_path.exists():
        raise FileNotFoundError("Model definition not found: {}".format(model_path))

    module_name = "pointnet_external_{}".format(model_name)
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("Could not load model module from {}".format(model_path))
    spec.loader.exec_module(module)
    return module


def call_model_get_model(model_module, point_cloud, is_training, bn_decay, cfg):
    """Call `get_model` while adapting to signature differences across model files."""
    signature = inspect.signature(model_module.get_model)
    kwargs = {}
    if "bn_decay" in signature.parameters:
        kwargs["bn_decay"] = bn_decay
    if "num_classes" in signature.parameters:
        kwargs["num_classes"] = cfg["num_classes"]
    elif cfg["num_classes"] != 40:
        raise ValueError(
            "Model {} does not support configurable num_classes, but cfg['num_classes']={}.".format(
                cfg["model_name"], cfg["num_classes"]
            )
        )
    return model_module.get_model(point_cloud, is_training, **kwargs)


def call_model_get_loss(model_module, pred, label, end_points, cfg):
    """Call `get_loss` while passing only args supported by the target model."""
    signature = inspect.signature(model_module.get_loss)
    kwargs = {}
    if "reg_weight" in signature.parameters:
        kwargs["reg_weight"] = cfg["reg_weight"]
    if "class_weights" in signature.parameters and cfg.get("use_class_weights"):
        kwargs["class_weights"] = cfg["class_weights"]
    return model_module.get_loss(pred, label, end_points, **kwargs)


def get_learning_rate(global_step, cfg):
    """Standard exponential LR decay with a lower bound."""
    learning_rate = tf1.train.exponential_decay(
        cfg["learning_rate"],
        global_step * cfg["batch_size"],
        cfg["decay_step"],
        cfg["decay_rate"],
        staircase=True,
    )
    return tf.maximum(learning_rate, 1e-5)


def get_bn_decay(global_step, cfg):
    """BatchNorm decay schedule used by original PointNet code."""
    bn_momentum = tf1.train.exponential_decay(
        0.5,
        global_step * cfg["batch_size"],
        float(cfg["decay_step"]),
        0.5,
        staircase=True,
    )
    return tf.minimum(0.99, 1 - bn_momentum)


def build_graph(cfg):
    """Create placeholders, model/loss ops, optimizer, and saver in one graph."""
    graph = tf.Graph()
    device_name = resolve_device_name(tf, cfg.get("gpu_index", 0))
    model_module = load_model_module(cfg.get("model_name", "pointnet_cls"))

    with graph.as_default():
        with tf1.device(device_name):
            # Use dynamic batch (`None`) so train/eval can share one graph.
            pointclouds_pl, labels_pl = model_module.placeholder_inputs(None, cfg["num_point"])
            is_training_pl = tf1.placeholder(tf.bool, shape=(), name="is_training")

            global_step = tf.Variable(0, name="global_step")
            bn_decay = get_bn_decay(global_step, cfg)
            pred, end_points = call_model_get_model(
                model_module,
                pointclouds_pl,
                is_training_pl,
                bn_decay,
                cfg,
            )
            loss = call_model_get_loss(
                model_module,
                pred,
                labels_pl,
                end_points,
                cfg,
            )
            predictions = tf.argmax(pred, axis=1, output_type=tf.int32, name="predictions")
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions, labels_pl), tf.float32), name="accuracy"
            )

            learning_rate = get_learning_rate(global_step, cfg)
            optimizer_name = str(cfg.get("optimizer", "adam")).strip().lower()
            if optimizer_name == "momentum":
                optimizer = tf1.train.MomentumOptimizer(
                    learning_rate, momentum=cfg.get("momentum", 0.9)
                )
            elif optimizer_name == "adam":
                optimizer = tf1.train.AdamOptimizer(learning_rate)
            else:
                raise ValueError("Unsupported optimizer={!r}".format(optimizer_name))
            train_op = optimizer.minimize(loss, global_step=global_step)
            saver = tf1.train.Saver(max_to_keep=2)

    return {
        "graph": graph,
        "device_name": device_name,
        "model_name": cfg.get("model_name", "pointnet_cls"),
        "pointclouds_pl": pointclouds_pl,
        "labels_pl": labels_pl,
        "is_training_pl": is_training_pl,
        "pred": pred,
        "predictions": predictions,
        "loss": loss,
        "accuracy": accuracy,
        "train_op": train_op,
        "global_step": global_step,
        "saver": saver,
    }


def create_session_config():
    return _create_session_config(tf1)


def parse_args():
    """CLI for quick graph sanity checks from shell wrappers."""
    parser = argparse.ArgumentParser(
        description="Build PointNet graph and print resolved model/device info."
    )
    parser.add_argument("--run-mode", choices=["full", "smoke"], default="smoke")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--num-point", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build graph and print metadata (no session run).",
    )
    return parser.parse_args()


def main():
    """Entry point used by `model.sh` for fast graph validation."""
    if __package__ in (None, ""):
        from pointnet_cls.configs import build_config
        from pointnet_cls.utils.data import load_datasets
    else:
        from .configs import build_config
        from .utils.data import load_datasets

    args = parse_args()
    cfg = build_config(
        run_mode=args.run_mode,
        data_dir=args.data_dir,
        model_name=args.model_name,
        gpu_index=args.gpu_index,
        num_point=args.num_point,
        batch_size=args.batch_size,
    )
    cfg, dataset = load_datasets(cfg)
    handles = build_graph(cfg)
    print("model_name:", handles["model_name"])
    print("device:", handles["device_name"])
    print("train_shape:", dataset["train_data"].shape)
    print("test_shape:", dataset["test_data"].shape)

    if not args.dry_run:
        # Create/close one session to ensure graph init path is valid.
        with tf1.Session(graph=handles["graph"], config=create_session_config()) as sess:
            sess.run(tf1.global_variables_initializer())
        print("session_init: ok")


if __name__ == "__main__":
    main()
