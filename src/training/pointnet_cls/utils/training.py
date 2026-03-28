from pathlib import Path

import numpy as np

from ..model import create_session_config, tf1
from .augmentations import (
    jitter_point_cloud,
    random_point_dropout,
    random_scale_point_cloud,
    rotate_point_cloud,
    shift_point_cloud,
    shuffle_data,
)
from .metrics import compute_eval_metrics


def trim_to_full_batches(data, labels, batch_size):
    """Keep only full batches to preserve stable tensor shapes across steps."""
    num_batches = len(data) // batch_size
    if num_batches == 0:
        raise ValueError(
            "Need at least one full batch. Got {} samples with batch_size={}.".format(
                len(data), batch_size
            )
        )
    usable = num_batches * batch_size
    return data[:usable], labels[:usable], num_batches


def iter_batches(data, labels, batch_size):
    """Yield sequential mini-batches, allowing a smaller final batch."""
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        yield data[start:end], labels[start:end]


def run_train_epoch(sess, handles, cfg, train_data, train_label, epoch_seed):
    """Run one training epoch with augmentation and return aggregate metrics."""
    rng = np.random.default_rng(epoch_seed)
    shuffled_data, shuffled_label = shuffle_data(train_data, train_label, rng=rng)
    shuffled_data, shuffled_label, num_batches = trim_to_full_batches(
        shuffled_data, shuffled_label, cfg["batch_size"]
    )

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx in range(num_batches):
        start = batch_idx * cfg["batch_size"]
        end = (batch_idx + 1) * cfg["batch_size"]
        batch_data = shuffled_data[start:end, : cfg["num_point"], :]
        batch_label = shuffled_label[start:end]

        # Apply enabled augmentations in a fixed, readable order.
        if cfg["rotate_augment"]:
            batch_data = rotate_point_cloud(batch_data, rng=rng)
        if cfg["jitter_augment"]:
            batch_data = jitter_point_cloud(batch_data, rng=rng)
        if cfg.get("scale_augment", False):
            batch_data = random_scale_point_cloud(batch_data, rng=rng)
        if cfg.get("shift_augment", False):
            batch_data = shift_point_cloud(batch_data, rng=rng)
        if cfg.get("dropout_augment", False):
            batch_data = random_point_dropout(
                batch_data,
                max_dropout_ratio=cfg.get("point_dropout_ratio", 0.1),
                rng=rng,
            )

        _, loss_value, logits = sess.run(
            [handles["train_op"], handles["loss"], handles["pred"]],
            feed_dict={
                handles["pointclouds_pl"]: batch_data,
                handles["labels_pl"]: batch_label,
                handles["is_training_pl"]: True,
            },
        )

        predictions = np.argmax(logits, axis=1)
        total_loss += float(loss_value)
        total_correct += int(np.sum(predictions == batch_label))
        total_seen += len(batch_label)

    return {
        "loss": total_loss / float(num_batches),
        "acc": total_correct / float(total_seen),
        "num_batches": num_batches,
    }


def run_eval_epoch(sess, handles, cfg, eval_data, eval_label):
    """Run one validation pass and compute both scalar and per-class metrics."""
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_labels = []
    all_preds = []
    num_batches = 0

    for batch_data, batch_label in iter_batches(
        eval_data[:, : cfg["num_point"], :], eval_label, cfg["batch_size"]
    ):
        logits, loss_value = sess.run(
            [handles["pred"], handles["loss"]],
            feed_dict={
                handles["pointclouds_pl"]: batch_data,
                handles["labels_pl"]: batch_label,
                handles["is_training_pl"]: False,
            },
        )
        predictions = np.argmax(logits, axis=1)
        all_preds.extend(predictions.tolist())
        all_labels.extend(batch_label.tolist())
        total_loss += float(loss_value) * len(batch_label)
        total_correct += int(np.sum(predictions == batch_label))
        total_seen += len(batch_label)
        num_batches += 1

    if total_seen == 0:
        raise ValueError("Validation split is empty.")

    all_labels = np.asarray(all_labels, dtype=np.int32)
    all_preds = np.asarray(all_preds, dtype=np.int32)
    metrics = compute_eval_metrics(all_labels, all_preds)
    return {
        "loss": total_loss / float(total_seen),
        "acc": total_correct / float(total_seen),
        "labels": all_labels,
        "preds": all_preds,
        "metrics": metrics,
        "num_batches": num_batches,
    }


def get_checkpoint_score(eval_out, cfg):
    """Map configured checkpoint metric name to a concrete scalar score."""
    metric_name = cfg.get("checkpoint_metric", "accuracy")
    metrics = eval_out.get("metrics")
    if metrics is None:
        metrics = compute_eval_metrics(eval_out["labels"], eval_out["preds"])

    if metric_name == "accuracy":
        return metrics["accuracy"]
    if metric_name == "macro_f1":
        return metrics["macro_f1"]
    if metric_name == "broken_recall":
        return metrics["broken_recall"]
    if metric_name == "broken_f1":
        return metrics["broken_f1"]
    raise ValueError("Unsupported checkpoint_metric={!r}".format(metric_name))


def is_better_checkpoint_score(current_score, best_score, cfg):
    """Compare scores according to `checkpoint_metric_mode` (max/min)."""
    if best_score is None:
        return True
    mode = str(cfg.get("checkpoint_metric_mode", "max")).strip().lower()
    if mode == "max":
        return current_score >= best_score
    if mode == "min":
        return current_score <= best_score
    raise ValueError("Unsupported checkpoint_metric_mode={!r}".format(mode))


def train_model(cfg, dataset, handles, log_fn=print):
    """Full training loop with best/last checkpoint management."""
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_broken_recall": [],
        "val_broken_f1": [],
        "best_accuracy": 0.0,
        "best_score": None,
        "best_epoch": None,
        "best_checkpoint": None,
        "last_checkpoint": None,
    }

    train_data = dataset["train_data"][:, : cfg["num_point"], :]
    train_label = dataset["train_label"]
    val_data = dataset["test_data"][:, : cfg["num_point"], :]
    val_label = dataset["test_label"]

    best_checkpoint = str(Path(cfg["log_dir"]) / "best_model.ckpt")
    last_checkpoint = str(Path(cfg["log_dir"]) / "last_model.ckpt")

    with tf1.Session(graph=handles["graph"], config=create_session_config()) as sess:
        sess.run(tf1.global_variables_initializer())

        for epoch in range(cfg["max_epoch"]):
            train_metrics = run_train_epoch(
                sess,
                handles,
                cfg,
                train_data,
                train_label,
                epoch_seed=cfg["seed"] + epoch,
            )
            val_metrics = run_eval_epoch(sess, handles, cfg, val_data, val_label)
            val_summary = val_metrics["metrics"]

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])
            history["val_macro_f1"].append(val_summary["macro_f1"])
            history["val_broken_recall"].append(val_summary["broken_recall"])
            history["val_broken_f1"].append(val_summary["broken_f1"])

            current_score = float(get_checkpoint_score(val_metrics, cfg))
            # Save whenever the selected validation metric improves.
            if is_better_checkpoint_score(current_score, history["best_score"], cfg):
                history["best_score"] = current_score
                history["best_epoch"] = epoch
                history["best_accuracy"] = float(val_metrics["acc"])
                handles["saver"].save(sess, best_checkpoint)

            log_fn(
                "Epoch {:03d}/{:03d} | train_loss={:.4f} train_acc={:.4f} | "
                "val_loss={:.4f} val_acc={:.4f} val_macro_f1={:.4f} val_broken_f1={:.4f}".format(
                    epoch + 1,
                    cfg["max_epoch"],
                    train_metrics["loss"],
                    train_metrics["acc"],
                    val_metrics["loss"],
                    val_metrics["acc"],
                    val_summary["macro_f1"],
                    val_summary["broken_f1"],
                )
            )

        if cfg.get("save_last_checkpoint", True):
            handles["saver"].save(sess, last_checkpoint)

    history["best_checkpoint"] = best_checkpoint
    history["last_checkpoint"] = last_checkpoint if cfg.get("save_last_checkpoint", True) else None
    return history
