from pathlib import Path

import numpy as np

from ..model import create_session_config, tf1
from .data import apply_sample_limit
from .metrics import (
    build_classification_report,
    compute_eval_metrics,
    softmax_np,
)


def resolve_checkpoint_path(cfg, checkpoint_path=None):
    candidates = []
    if checkpoint_path is not None:
        candidates.append(Path(checkpoint_path))
    else:
        log_dir = Path(cfg["log_dir"])
        candidates.extend(
            [
                log_dir / "best_model.ckpt",
                log_dir / "last_model.ckpt",
                log_dir / "final_model.ckpt",
            ]
        )

    for candidate in candidates:
        if Path(str(candidate) + ".index").exists():
            return str(candidate)
    raise FileNotFoundError("No checkpoint found in {}".format([str(path) for path in candidates]))


def evaluate_checkpoint(cfg, dataset, handles, checkpoint_path, limit=None):
    checkpoint_path = resolve_checkpoint_path(cfg, checkpoint_path)

    eval_data = dataset["test_data"][:, : cfg["num_point"], :]
    eval_label = dataset["test_label"]
    eval_ids = np.asarray(
        dataset.get(
            "test_ids", ["test_sample_{}".format(idx) for idx in range(len(eval_label))]
        ),
        dtype=object,
    )

    if limit is not None:
        eval_data, eval_label, eval_indices = apply_sample_limit(
            eval_data, eval_label, limit, cfg["seed"] + 99
        )
        eval_ids = eval_ids[eval_indices]

    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_seen = 0

    with tf1.Session(graph=handles["graph"], config=create_session_config()) as sess:
        handles["saver"].restore(sess, checkpoint_path)
        for start in range(0, len(eval_data), cfg["batch_size"]):
            end = min(start + cfg["batch_size"], len(eval_data))
            batch_data = eval_data[start:end]
            batch_label = eval_label[start:end]

            logits, loss_value = sess.run(
                [handles["pred"], handles["loss"]],
                feed_dict={
                    handles["pointclouds_pl"]: batch_data,
                    handles["labels_pl"]: batch_label,
                    handles["is_training_pl"]: False,
                },
            )

            probabilities = softmax_np(logits)
            predictions = np.argmax(logits, axis=1)

            all_preds.extend(predictions.tolist())
            all_probs.extend(probabilities[:, 1].tolist())
            all_labels.extend(batch_label.tolist())
            total_loss += float(loss_value) * len(batch_label)
            total_seen += len(batch_label)

    if total_seen == 0:
        raise ValueError("Evaluation split is empty.")

    all_preds = np.asarray(all_preds, dtype=np.int32)
    all_probs = np.asarray(all_probs, dtype=np.float32)
    all_labels = np.asarray(all_labels, dtype=np.int32)
    metrics = compute_eval_metrics(all_labels, all_preds)

    return {
        "checkpoint_path": checkpoint_path,
        "loss": total_loss / float(total_seen),
        "accuracy": metrics["accuracy"],
        "labels": all_labels,
        "preds": all_preds,
        "probs_broken": all_probs,
        "confusion_matrix": metrics["confusion_matrix"],
        "report": build_classification_report(
            all_labels,
            all_preds,
            label_names=("Complete", "Broken"),
        ),
        "metrics": metrics,
        "object_ids": eval_ids,
    }


def collect_error_rows(eval_result, label_names):
    labels = np.asarray(eval_result["labels"], dtype=np.int32)
    preds = np.asarray(eval_result["preds"], dtype=np.int32)
    probs_broken = np.asarray(eval_result["probs_broken"], dtype=np.float32)
    object_ids = np.asarray(
        eval_result.get(
            "object_ids", ["sample_{}".format(idx) for idx in range(len(labels))]
        ),
        dtype=object,
    )

    wrong_indices = np.where(labels != preds)[0]
    rows = []
    for index in wrong_indices:
        rows.append(
            {
                "sample_index": int(index),
                "object_id": str(object_ids[index]),
                "true_label": str(label_names[labels[index]]),
                "pred_label": str(label_names[preds[index]]),
                "p_broken": float(probs_broken[index]),
                "confidence": float(
                    max(probs_broken[index], 1.0 - probs_broken[index])
                ),
            }
        )
    rows.sort(key=lambda row: row["confidence"], reverse=True)
    return rows


def format_error_summary(error_rows, top_k=10):
    if not error_rows:
        return "No classification errors."

    lines = ["Most confident mistakes:"]
    for row in error_rows[:top_k]:
        lines.append(
            "  {sample_index:3d} | {object_id} | true={true_label} pred={pred_label} | "
            "p_broken={p_broken:.4f} | conf={confidence:.4f}".format(**row)
        )
    return "\n".join(lines)
