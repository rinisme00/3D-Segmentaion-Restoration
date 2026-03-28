"""Evaluate a trained PointNet checkpoint and save analysis artifacts.

Outputs include classification report text, metrics JSON, confusion/ROC/PR plots,
error tables, and point-cloud inference images.
"""

import argparse
import os
import sys
from pathlib import Path

# Use a writable matplotlib cache dir for headless/shared environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pointnet_cls.configs import LABEL_NAMES, build_config, print_config
from pointnet_cls.model import build_graph
from pointnet_cls.utils import (
    collect_error_rows,
    evaluate_checkpoint,
    format_error_summary,
    load_datasets,
    make_logger,
    resolve_checkpoint_path,
    save_evaluation_plots,
    save_json,
    save_rows_as_csv,
    write_text,
)
from pointnet_cls.utils.inference import run_random_test_inference, run_visual_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PointNet classification model.")
    parser.add_argument("--run-mode", choices=["full", "smoke"], default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--num-point", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-inference-samples", type=int, default=None)
    parser.add_argument("--inference-seed", type=int, default=None)
    parser.add_argument("--top-k-errors", type=int, default=15)
    parser.add_argument("--test-sample-pos", type=int, default=None)
    parser.add_argument("--input-object-id", default=None)
    return parser.parse_args()


def save_inference_result(result, output_path, label_names):
    """Persist one inference figure and return a CSV-friendly metadata row."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["figure"].savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(result["figure"])

    return {
        "sample_index": result["test_sample_pos"],
        "object_id": result["object_id"],
        "true_label": label_names[result["true_label"]],
        "pred_label": label_names[result["pred_label"]],
        "p_complete": float(result["probs"][0]),
        "p_broken": float(result["probs"][1]),
        "match": bool(result["pred_label"] == result["true_label"]),
        "mesh_path": str(result["mesh_path"]),
        "source_desc": result["source_desc"],
        "image_path": str(output_path),
    }


def main():
    args = parse_args()
    # Build evaluation config from defaults + CLI overrides.
    cfg = build_config(
        run_mode=args.run_mode,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        gpu_index=args.gpu_index,
        num_point=args.num_point,
        batch_size=args.batch_size,
        eval_limit=args.eval_limit,
        seed=args.seed,
        num_inference_samples=args.num_inference_samples,
    )

    logger = make_logger(Path(cfg["log_dir"]) / "evaluate.log")
    print_config(cfg, log_fn=logger)

    # Rebuild the same model graph shape/settings used for training.
    cfg, dataset = load_datasets(cfg, log_fn=logger)
    handles = build_graph(cfg)
    logger("Resolved evaluation device: {}".format(handles["device_name"]))

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint_path)
    logger("Evaluating checkpoint: {}".format(checkpoint_path))

    eval_result = evaluate_checkpoint(
        cfg,
        dataset,
        handles,
        checkpoint_path=checkpoint_path,
        limit=cfg.get("eval_limit"),
    )

    # Core report/metrics files.
    report_path = Path(cfg["log_dir"]) / "classification_report.txt"
    metrics_path = Path(cfg["log_dir"]) / "evaluation_metrics.json"
    errors_path = Path(cfg["log_dir"]) / "top_errors.csv"
    inference_path = Path(cfg["log_dir"]) / "inference_results.csv"

    write_text(report_path, eval_result["report"] + "\n")
    logger(eval_result["report"])

    plot_paths = save_evaluation_plots(eval_result, cfg["log_dir"], LABEL_NAMES)
    logger("Saved evaluation plots to {}".format(cfg["log_dir"]))

    error_rows = collect_error_rows(eval_result, LABEL_NAMES)
    save_rows_as_csv(
        errors_path,
        error_rows,
        fieldnames=[
            "sample_index",
            "object_id",
            "true_label",
            "pred_label",
            "p_broken",
            "confidence",
        ],
    )
    logger(format_error_summary(error_rows, top_k=args.top_k_errors))

    metrics_payload = {
        "checkpoint_path": checkpoint_path,
        "loss": eval_result["loss"],
        "accuracy": eval_result["accuracy"],
        "metrics": eval_result["metrics"],
        "confusion_matrix": eval_result["confusion_matrix"],
        "plot_paths": plot_paths,
        "report_path": report_path,
        "top_errors_path": errors_path,
    }
    save_json(metrics_path, metrics_payload)

    # Optional targeted inference plus random sampled inference visualizations.
    inference_rows = []
    if args.test_sample_pos is not None or args.input_object_id is not None:
        targeted = run_visual_inference(
            handles,
            dataset,
            cfg,
            checkpoint_path=checkpoint_path,
            label_names=LABEL_NAMES,
            test_sample_pos=0 if args.test_sample_pos is None else args.test_sample_pos,
            input_object_id=args.input_object_id,
        )
        targeted_name = "inference_target_{}.png".format(targeted["object_id"])
        inference_rows.append(
            save_inference_result(targeted, Path(cfg["log_dir"]) / targeted_name, LABEL_NAMES)
        )

    random_results = run_random_test_inference(
        handles,
        dataset,
        cfg,
        checkpoint_path=checkpoint_path,
        label_names=LABEL_NAMES,
        num_samples=cfg["num_inference_samples"],
        seed=args.inference_seed,
    )
    for result in random_results:
        filename = "inference_{}.png".format(result["object_id"])
        inference_rows.append(
            save_inference_result(result, Path(cfg["log_dir"]) / filename, LABEL_NAMES)
        )

    save_rows_as_csv(
        inference_path,
        inference_rows,
        fieldnames=[
            "sample_index",
            "object_id",
            "true_label",
            "pred_label",
            "p_complete",
            "p_broken",
            "match",
            "mesh_path",
            "source_desc",
            "image_path",
        ],
    )
    logger("Saved inference results to {}".format(inference_path))


if __name__ == "__main__":
    main()
