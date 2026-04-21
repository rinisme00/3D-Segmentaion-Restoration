"""
Main script to build canonical dataset manifests for Fantastic Breaks and Breaking Bad.
Enforces object-disjoint splits and standardizes the schema.
"""

import os
import argparse
from pathlib import Path
import sys

# Add src to path so we can import our new package
sys.path.append(str(Path(__file__).parent.parent))

from src.data.manifests import (
    parse_fantastic_breaks,
    parse_breaking_bad,
    assign_object_disjoint_splits,
    validate_manifest
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
        print(f"Found {len(df_fb)} FB samples.")
        
        print("Assigning object-disjoint splits to FB...")
        df_fb = assign_object_disjoint_splits(df_fb, seed=args.seed)
        
        validate_manifest(df_fb)
        fb_path = output_dir / "fantastic_breaks_classification.csv"
        df_fb.to_csv(fb_path, index=False)
        print(f"Saved FB manifest to {fb_path}")
    else:
        print(f"WARNING: FB root {fb_root} not found. Skipping.")

    print("\n--- 2. Processing Breaking Bad ---")
    if bb_root.exists() and bb_split_dir.exists():
        # Using subsets mentioned: artifact and specific everyday categories
        subsets = ['artifact', 'everyday/Vase', 'everyday/Cup', 'everyday/Mug', 'everyday/Plate']
        df_bb = parse_breaking_bad(bb_root, bb_split_dir, subsets=subsets)
        print(f"Found {len(df_bb)} BB samples from decompressed subsets.")
        
        # BB splits are already assigned by official files, we just validate
        validate_manifest(df_bb)
        bb_path = output_dir / "breaking_bad_classification.csv"
        df_bb.to_csv(bb_path, index=False)
        print(f"Saved BB manifest to {bb_path}")
    else:
        print(f"WARNING: BB root or split dir not found. Skipping.")

    print("\nDone.")

if __name__ == "__main__":
    main()
