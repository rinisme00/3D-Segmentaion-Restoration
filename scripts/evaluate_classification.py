#!/usr/bin/env python
"""
Standalone evaluation script for 3D Fracture Classification (PointNeXt).

Produces thesis-quality outputs:
  - metrics.json            (OA, mAcc, Precision, Recall, F1 per class + macro)
  - classification_report.txt  (sklearn-style report)
  - confusion_matrix.png    (annotated heatmap)
  - predictions.csv         (per-sample: true_label, pred_label, prob_complete, prob_broken)
  - experiment_summary.json (metadata: checkpoint, config, feature_dims, results)

Usage:
  conda activate pointnext
  python scripts/evaluate_classification.py \\
      --cfg src/pointnext/cfgs/fantasticbreaks/pointnext-b.yaml \\
      --checkpoint src/pointnext/log/fantasticbreaks/<run>/checkpoint/<run>_ckpt_best.pth \\
      --split val \\
      --output_dir results/fb_9d \\
      --gpu 0
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---- resolve PointNeXt package root ----------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
POINTNEXT_ROOT = REPO_ROOT / "src" / "pointnext"
sys.path.insert(0, str(POINTNEXT_ROOT))

from openpoints.dataset import build_dataloader_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample
from openpoints.utils import EasyConfig, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PointNeXt classification checkpoint.")
    p.add_argument("--cfg",        required=True,  help="YAML config (same as used for training)")
    p.add_argument("--checkpoint", required=True,  help="Path to .pth checkpoint to evaluate")
    p.add_argument("--split",      default="val",  choices=["train", "val", "test"],
                   help="Dataset split to evaluate on (default: val)")
    p.add_argument("--output_dir", required=True,  help="Directory to save evaluation outputs")
    p.add_argument("--gpu",        default="0",    help="CUDA device index (default: 0)")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    return p.parse_args()


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], save_path: str,
                           title: str = "Confusion Matrix") -> None:
    """Save a publication-quality annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=11)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.tick_params(labelsize=11)

    # Annotate each cell with raw count and percentage
    total = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i, j] / max(cm[i].sum(), 1)
            ax.text(
                j, i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved confusion matrix → {save_path}")


def run_inference(model, loader, cfg, device):
    """Run full inference; return (all_preds, all_targets, all_probs)."""
    model.eval()
    npoints = cfg.num_points
    in_channels = cfg.model.encoder_args.in_channels

    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for data in loader:
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)

            points = data["x"]
            num_curr_pts = points.shape[1]

            # FPS resampling (mirrors train.py logic)
            if num_curr_pts > npoints:
                point_all = {1024: 1200, 2048: 2400, 4096: 4800, 8192: 8192}.get(
                    npoints, int(npoints * 1.2)
                )
                point_all = min(point_all, num_curr_pts)
                fps_idx = furthest_point_sample(points[:, :, :3].contiguous(), point_all)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                points = torch.gather(
                    points, 1,
                    fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1])
                )

            data["pos"] = points[:, :, :3].contiguous()
            data["x"]   = points[:, :, :in_channels].transpose(1, 2).contiguous()

            logits = model(data)                          # [B, num_classes]
            probs  = F.softmax(logits, dim=-1)            # [B, num_classes]
            preds  = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(data["y"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_probs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "evaluate.log"),
            logging.StreamHandler(),
        ],
    )

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    # Ensure distributed is off for standalone eval
    cfg.distributed = False
    cfg.rank = 0
    if args.batch_size:
        cfg.val_batch_size = args.batch_size

    classes: list[str] = cfg.dataset.common.get("classes", ["complete", "broken"])
    num_classes: int = cfg.num_classes
    in_channels: int = cfg.model.encoder_args.in_channels

    logging.info(f"Config   : {args.cfg}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Split    : {args.split}")
    logging.info(f"Classes  : {classes}  |  in_channels={in_channels}")

    # ── Build model & load checkpoint ────────────────────────────────────────
    model = build_model_from_cfg(cfg.model).to(device)
    epoch, metrics = load_checkpoint(model, pretrained_path=args.checkpoint)
    
    # Extract best_val from metrics dict if available
    best_val_val = metrics.get('best_val', metrics.get('acc', 0.0))
    if isinstance(best_val_val, dict):
        best_val_val = 0.0 # fallback if it's still a dict
        
    logging.info(f"Loaded checkpoint @ epoch={epoch}, metrics={metrics}")
    model.eval()

    # ── Resolve data_dir ─────────────────────────────────────────────────────
    # PointNeXt configs often use relative paths like '../../data/...' 
    # intended for runs starting from src/pointnext/. We resolve them here.
    data_dir = cfg.dataset.common.data_dir
    if not os.path.isabs(data_dir):
        # Resolve relative to POINTNEXT_ROOT
        resolved_path = (POINTNEXT_ROOT / data_dir).resolve()
        cfg.dataset.common.data_dir = str(resolved_path)
        logging.info(f"Resolved relative data_dir: {data_dir} → {resolved_path}")

    # ── Build dataloader ─────────────────────────────────────────────────────
    loader = build_dataloader_from_cfg(
        cfg.get("val_batch_size", cfg.batch_size),
        cfg.dataset,
        cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms,
        split=args.split,
        distributed=False,
    )
    logging.info(f"Loaded {len(loader.dataset)} samples from split='{args.split}'")

    # ── Run inference ────────────────────────────────────────────────────────
    preds, targets, probs = run_inference(model, loader, cfg, device)

    # ── Compute metrics ──────────────────────────────────────────────────────
    oa   = float((preds == targets).mean() * 100)
    macc = float(np.mean([
        (preds[targets == c] == c).mean() * 100
        for c in range(num_classes) if (targets == c).sum() > 0
    ]))

    prec  = precision_score(targets, preds, average=None, zero_division=0)
    rec   = recall_score(targets, preds, average=None, zero_division=0)
    f1    = f1_score(targets, preds, average=None, zero_division=0)
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    broken_idx = 1   # class index for "broken"

    per_class = {
        classes[c]: {
            "precision": float(prec[c]),
            "recall":    float(rec[c]),
            "f1":        float(f1[c]),
        }
        for c in range(num_classes)
    }

    metrics = {
        "overall_accuracy_pct": oa,
        "mean_class_accuracy_pct": macc,
        "macro_f1": macro_f1,
        "broken_precision": float(prec[broken_idx]),
        "broken_recall":    float(rec[broken_idx]),
        "broken_f1":        float(f1[broken_idx]),
        "per_class": per_class,
    }

    # ── Classification report ────────────────────────────────────────────────
    report_str = classification_report(
        targets, preds, target_names=classes, digits=4
    )
    logging.info("\n" + report_str)
    (output_dir / "classification_report.txt").write_text(
        f"Split: {args.split}\nCheckpoint: {args.checkpoint}\n\n{report_str}"
    )

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(targets, preds)
    dataset_name = cfg.dataset.common.NAME
    plot_confusion_matrix(
        cm, classes,
        save_path=str(output_dir / "confusion_matrix.png"),
        title=f"{dataset_name} — {args.split} split",
    )

    # ── Per-sample predictions CSV ───────────────────────────────────────────
    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx", "true_label", "pred_label",
            "true_class", "pred_class",
            *[f"prob_{c}" for c in classes],
            "correct",
        ])
        for i in range(len(preds)):
            writer.writerow([
                i,
                int(targets[i]), int(preds[i]),
                classes[targets[i]], classes[preds[i]],
                *[f"{probs[i, c]:.6f}" for c in range(num_classes)],
                int(preds[i] == targets[i]),
            ])
    logging.info(f"Saved per-sample predictions → {csv_path}")

    # ── metrics.json ─────────────────────────────────────────────────────────
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logging.info(f"Saved metrics → {metrics_path}")

    # ── experiment_summary.json (for compare_experiments.py) ─────────────────
    summary = {
        "checkpoint": str(args.checkpoint),
        "cfg": str(args.cfg),
        "split": args.split,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "checkpoint_epoch": epoch,
        "checkpoint_best_val": float(best_val_val),
        "n_samples": int(len(targets)),
        "results": metrics,
        "confusion_matrix": cm.tolist(),
    }
    summary_path = output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info(f"Saved experiment summary → {summary_path}")

    # ── Terminal summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print(f"  Evaluation: {dataset_name}  |  split={args.split}")
    print("═" * 55)
    print(f"  Overall Accuracy : {oa:.2f}%")
    print(f"  Mean Class Acc   : {macc:.2f}%")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"  Broken  Precision: {prec[broken_idx]:.4f}")
    print(f"  Broken  Recall   : {rec[broken_idx]:.4f}")
    print(f"  Broken  F1       : {f1[broken_idx]:.4f}")
    print("═" * 55)
    print(f"  Outputs saved to: {output_dir.resolve()}")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
