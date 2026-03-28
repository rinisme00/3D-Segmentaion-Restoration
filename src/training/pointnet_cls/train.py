"""Train PointNet classification and save checkpoints/learning curves.

This file is intentionally thin: it only parses CLI flags, resolves config/data,
calls the training loop, and saves outputs.
"""

import argparse
import sys
from pathlib import Path

# Allow `python src/training/pointnet_cls/train.py` from repo root.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pointnet_cls.configs import build_config, print_config
from pointnet_cls.model import build_graph
from pointnet_cls.utils import load_datasets, make_logger, save_json, save_training_curves, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNet classification model.")
    parser.add_argument("--run-mode", choices=["full", "smoke"], default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--num-point", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "momentum"], default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--checkpoint-metric", default=None)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--disable-rotate-augment", action="store_true")
    parser.add_argument("--disable-jitter-augment", action="store_true")
    parser.add_argument("--disable-scale-augment", action="store_true")
    parser.add_argument("--disable-shift-augment", action="store_true")
    parser.add_argument("--disable-dropout-augment", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    # Merge defaults + run-mode + CLI overrides into one config dict.
    cfg = build_config(
        run_mode=args.run_mode,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        gpu_index=args.gpu_index,
        num_point=args.num_point,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        momentum=args.momentum,
        seed=args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        checkpoint_metric=args.checkpoint_metric,
        use_class_weights=False if args.disable_class_weights else None,
        rotate_augment=False if args.disable_rotate_augment else None,
        jitter_augment=False if args.disable_jitter_augment else None,
        scale_augment=False if args.disable_scale_augment else None,
        shift_augment=False if args.disable_shift_augment else None,
        dropout_augment=False if args.disable_dropout_augment else None,
    )

    logger = make_logger(Path(cfg["log_dir"]) / "train.log")
    print_config(cfg, log_fn=logger)

    # Load prepared H5 splits and build the TF graph once.
    cfg, dataset = load_datasets(cfg, log_fn=logger)
    handles = build_graph(cfg)
    logger("Resolved training device: {}".format(handles["device_name"]))

    # Save run metadata for reproducibility.
    config_path = Path(cfg["log_dir"]) / "config.json"
    save_json(config_path, cfg)
    logger("Saved config to {}".format(config_path))

    # Execute training loop and collect epoch-by-epoch history.
    history = train_model(cfg, dataset, handles, log_fn=logger)
    history_path = Path(cfg["log_dir"]) / "history.json"
    curve_path = Path(cfg["log_dir"]) / "training_curves.png"

    save_json(history_path, history)
    save_training_curves(history, curve_path)

    logger("Saved training history to {}".format(history_path))
    logger("Saved training curves to {}".format(curve_path))
    logger("Best checkpoint: {}".format(history["best_checkpoint"]))
    logger("Last checkpoint: {}".format(history["last_checkpoint"]))


if __name__ == "__main__":
    main()
