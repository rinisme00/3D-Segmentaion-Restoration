"""
Main script to build canonical dataset manifests for Fantastic Breaks and Breaking Bad.
Enforces object-disjoint splits and standardizes the schema.
"""

import os
import argparse
from pathlib import Path
import sys
import pandas as pd

# Add src to path so we can import our new package
sys.path.append(str(Path(__file__).parent.parent))

from src.data.manifests import (
    parse_fantastic_breaks,
    parse_breaking_bad,
    assign_object_disjoint_splits,
    validate_manifest
)

def _print_manifest_summary(df: pd.DataFrame, name: str) -> None:
    """Print class balance and split composition for a manifest DataFrame."""
    total = len(df)
    n_complete = (df['label'] == 0).sum()
    n_broken   = (df['label'] == 1).sum()
    n_objects  = df['base_object_id'].nunique()
    print(f"  Samples : {total:>6} ({n_complete} complete, {n_broken} broken)")
    print(f"  Objects : {n_objects:>6} unique base objects")
    for split_name, grp in df.groupby('split', sort=True):
        nc = (grp['label'] == 0).sum()
        nb = (grp['label'] == 1).sum()
        no = grp['base_object_id'].nunique()
        print(f"  {split_name:<8}: {len(grp):>5} samples  "
              f"({nc} complete / {nb} broken)  {no} objects")


def _verify_manifest_integrity(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Run post-save integrity checks on a manifest.
    Prints PASS / FAIL for each check. Raises RuntimeError on any failure.
    """
    errors = []
    print("  Integrity checks:")

    # Check 1: no unknown splits
    unknown = df[df['split'].isin(['unknown', 'None']) | df['split'].isna()]
    if unknown.empty:
        print("    [PASS] No unknown/unassigned splits")
    else:
        msg = f"{len(unknown)} rows have split='unknown' or NaN"
        print(f"    [FAIL] {msg}")
        errors.append(msg)

    # Check 2: both labels present in every split
    for split_name, grp in df.groupby('split', sort=True):
        labels_present = set(grp['label'].unique())
        if {0, 1}.issubset(labels_present):
            print(f"    [PASS] Split '{split_name}' has both label classes")
        else:
            msg = f"Split '{split_name}' is missing label(s): {({0,1} - labels_present)}"
            print(f"    [FAIL] {msg}")
            errors.append(msg)

    # Check 3: no duplicate mesh paths
    dup_paths = df['file_path_mesh'][df['file_path_mesh'].duplicated()]
    if dup_paths.empty:
        print("    [PASS] No duplicate mesh file paths")
    else:
        msg = f"{len(dup_paths)} duplicate mesh paths found"
        print(f"    [FAIL] {msg}")
        errors.append(msg)

    # Check 4 (FB only): every base object has exactly 1 complete + 1 broken
    if dataset_name == "fantastic_breaks":
        obj_counts = df.groupby('base_object_id')['label'].value_counts().unstack(fill_value=0)
        bad = obj_counts[(obj_counts.get(0, 0) != 1) | (obj_counts.get(1, 0) != 1)]
        if bad.empty:
            print("    [PASS] Every FB base object has exactly 1 complete + 1 broken")
        else:
            msg = f"{len(bad)} FB objects do not have exactly 1 complete + 1 broken pair"
            print(f"    [FAIL] {msg}")
            errors.append(msg)

    # Check 5: split ratio sanity (each split within 5%–95%)
    total = len(df)
    for split_name, grp in df.groupby('split', sort=True):
        ratio = len(grp) / total
        if 0.03 <= ratio <= 0.97:
            print(f"    [PASS] Split '{split_name}' ratio = {ratio:.1%} (within bounds)")
        else:
            msg = f"Split '{split_name}' ratio = {ratio:.1%} is outside [3%, 97%]"
            print(f"    [WARN] {msg}")
            # Warning only, not error

    if errors:
        raise RuntimeError(
            f"Manifest integrity check failed for '{dataset_name}':\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


def main():
    parser = argparse.ArgumentParser(description="Build dataset manifests.")
    parser.add_argument("--fb_root", type=str, default="data/Fantastic_Breaks_v1")
    parser.add_argument("--bb_root", type=str, default="data/BreakingBad")
    parser.add_argument("--bb_split_dir", type=str, default="data/BreakingBad/data_split")
    parser.add_argument("--output_dir", type=str, default="data/manifests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    fb_root = project_root / args.fb_root
    bb_root = project_root / args.bb_root
    bb_split_dir = project_root / args.bb_split_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- 1. Processing Fantastic Breaks ---")
    if fb_root.exists():
        df_fb = parse_fantastic_breaks(fb_root)
        print(f"Found {len(df_fb)} FB samples. Assigning object-disjoint splits...")
        df_fb = assign_object_disjoint_splits(df_fb, seed=args.seed)
        validate_manifest(df_fb)
        _verify_manifest_integrity(df_fb, "fantastic_breaks")
        fb_path = output_dir / "fantastic_breaks_classification.csv"
        df_fb.to_csv(fb_path, index=False)
        print(f"Saved FB manifest → {fb_path}")
        _print_manifest_summary(df_fb, "Fantastic Breaks")
    else:
        print(f"WARNING: FB root {fb_root} not found. Skipping.")

    print("\n--- 2. Processing Breaking Bad ---")
    if bb_root.exists() and bb_split_dir.exists():
        subsets = ['artifact', 'everyday/Vase', 'everyday/Cup', 'everyday/Mug', 'everyday/Plate']
        df_bb = parse_breaking_bad(bb_root, bb_split_dir, subsets=subsets)
        print(f"Found {len(df_bb)} BB samples from decompressed subsets.")
        validate_manifest(df_bb)
        _verify_manifest_integrity(df_bb, "breaking_bad")
        bb_path = output_dir / "breaking_bad_classification.csv"
        df_bb.to_csv(bb_path, index=False)
        print(f"Saved BB manifest → {bb_path}")
        _print_manifest_summary(df_bb, "Breaking Bad")
    else:
        print(f"WARNING: BB root or split dir not found. Skipping.")

    print("\nDone.")

if __name__ == "__main__":
    main()
