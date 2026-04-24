#!/usr/bin/env python
"""
Cross-experiment comparison for 3D Fracture Classification.

Reads multiple experiment_summary.json files produced by evaluate_classification.py
and outputs a unified comparison table (terminal + CSV).

Usage:
  python scripts/compare_experiments.py \\
      results/fb_9d/experiment_summary.json \\
      results/bb_9d/experiment_summary.json \\
      --output results/comparison.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Compare classification experiment results.")
    p.add_argument("summaries", nargs="+", help="Paths to experiment_summary.json files")
    p.add_argument("--output", default=None, help="Save comparison table as CSV (optional)")
    return p.parse_args()


COLUMNS = [
    ("dataset",              "Dataset"),
    ("split",                "Split"),
    ("in_channels",          "In_Ch"),
    ("checkpoint_epoch",     "Epoch"),
    ("oa",                   "OA (%)"),
    ("macc",                 "mAcc (%)"),
    ("macro_f1",             "Macro F1"),
    ("broken_precision",     "Broken Prec"),
    ("broken_recall",        "Broken Rec"),
    ("broken_f1",            "Broken F1"),
    ("complete_f1",          "Complete F1"),
    ("n_samples",            "N"),
]


def extract_row(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    r = data.get("results", {})
    per_class = r.get("per_class", {})
    broken_key   = "broken"   if "broken"   in per_class else list(per_class.keys())[-1]
    complete_key = "complete" if "complete" in per_class else list(per_class.keys())[0]

    return {
        "dataset":          data.get("dataset", "?"),
        "split":            data.get("split", "?"),
        "in_channels":      data.get("in_channels", "?"),
        "checkpoint_epoch": data.get("checkpoint_epoch", "?"),
        "oa":               f"{r.get('overall_accuracy_pct', 0):.2f}",
        "macc":             f"{r.get('mean_class_accuracy_pct', 0):.2f}",
        "macro_f1":         f"{r.get('macro_f1', 0):.4f}",
        "broken_precision": f"{r.get('broken_precision', 0):.4f}",
        "broken_recall":    f"{r.get('broken_recall', 0):.4f}",
        "broken_f1":        f"{r.get('broken_f1', 0):.4f}",
        "complete_f1":      f"{per_class.get(complete_key, {}).get('f1', 0):.4f}",
        "n_samples":        data.get("n_samples", "?"),
        "_source":          path,
    }


def pretty_print(rows: list[dict]) -> None:
    headers = [col[1] for col in COLUMNS]
    col_keys = [col[0] for col in COLUMNS]

    # compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, key in enumerate(col_keys):
            widths[i] = max(widths[i], len(str(row.get(key, ""))))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"

    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(row.get(k, "")).ljust(widths[i]) for i, k in enumerate(col_keys)) + " |"
        print(line)
    print(sep)


def main():
    args = parse_args()
    rows = []
    for path in args.summaries:
        try:
            rows.append(extract_row(path))
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)

    if not rows:
        print("No valid experiment summaries found.", file=sys.stderr)
        sys.exit(1)

    pretty_print(rows)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        col_keys = [col[0] for col in COLUMNS] + ["_source"]
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=col_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved comparison table → {out.resolve()}")


if __name__ == "__main__":
    main()
