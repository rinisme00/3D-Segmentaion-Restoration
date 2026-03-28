from pathlib import Path
import os

# Headless-safe matplotlib cache directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .metrics import (
    binary_precision_recall_curve,
    binary_roc_curve,
)


def save_training_curves(history, output_path):
    """Save a two-panel training/validation loss+accuracy plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    ax_loss.plot(epochs, history["val_loss"], label="Validation loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss curves")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="Train accuracy")
    ax_acc.plot(epochs, history["val_acc"], label="Validation accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy curves")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    # Mark the epoch where the best checkpoint was chosen.
    best_epoch = history.get("best_epoch")
    if best_epoch is not None:
        ax_loss.axvline(best_epoch + 1, linestyle="--", alpha=0.5, color="black")
        ax_acc.axvline(best_epoch + 1, linestyle="--", alpha=0.5, color="black")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_confusion_matrix(eval_result, output_path, label_names):
    """Render and save the confusion matrix heatmap."""
    output_path = Path(output_path)
    confusion = np.asarray(eval_result["confusion_matrix"], dtype=np.int32)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion, cmap=plt.cm.Blues)
    axis.set_xticks(range(len(label_names)))
    axis.set_yticks(range(len(label_names)))
    axis.set_xticklabels(label_names)
    axis.set_yticklabels(label_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix")

    threshold = confusion.max() / 2.0 if confusion.size else 0.0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            color = "white" if confusion[row, col] > threshold else "black"
            axis.text(col, row, str(confusion[row, col]), ha="center", va="center", color=color)

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_roc_curve(eval_result, output_path):
    """Render and save ROC curve; fallback gracefully when unavailable."""
    output_path = Path(output_path)
    labels = np.asarray(eval_result["labels"], dtype=np.int32)
    probabilities = np.asarray(eval_result["probs_broken"], dtype=np.float32)

    figure, axis = plt.subplots(figsize=(6, 5))
    try:
        fpr, tpr, _, auc = binary_roc_curve(labels, probabilities, positive_label=1)
        axis.plot(fpr, tpr, label="PointNet (AUC={:.3f})".format(auc))
        axis.plot([0, 1], [0, 1], "k--", alpha=0.5)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title("ROC curve")
        axis.grid(True, alpha=0.3)
        axis.legend()
    except Exception as exc:
        axis.text(0.05, 0.5, "ROC unavailable: {}".format(exc), fontsize=10)
        axis.set_axis_off()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_precision_recall_curve(eval_result, output_path):
    """Render and save broken-class precision-recall curve."""
    output_path = Path(output_path)
    labels = np.asarray(eval_result["labels"], dtype=np.int32)
    probabilities = np.asarray(eval_result["probs_broken"], dtype=np.float32)

    figure, axis = plt.subplots(figsize=(6, 5))
    try:
        precision, recall, _, ap = binary_precision_recall_curve(
            labels, probabilities, positive_label=1
        )
        axis.plot(recall, precision, label="PointNet (AP={:.3f})".format(ap))
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Precision-Recall curve (Broken)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    except Exception as exc:
        axis.text(0.05, 0.5, "Precision-recall unavailable: {}".format(exc), fontsize=10)
        axis.set_axis_off()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_evaluation_plots(eval_result, output_dir, label_names):
    """Save all standard evaluation plots and return their output paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "confusion_matrix": save_confusion_matrix(
            eval_result, output_dir / "confusion_matrix.png", label_names
        ),
        "roc_curve": save_roc_curve(eval_result, output_dir / "roc_curve.png"),
        "precision_recall_curve_broken": save_precision_recall_curve(
            eval_result, output_dir / "precision_recall_curve_broken.png"
        ),
    }
