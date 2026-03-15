"""
Purpose:
This script scans a raw dataset directory (following the 'Fantastic Breaks' structure),
identifies relevant 3D object files (.npz, _b_.ply, _c.ply, _r_.ply), and builds a
metadata CSV file summarizing the available files for each object and their dataset split.

Usage:
    python scripts/build_metadata.py \\
        --raw-root /path/to/dataset/root \\
        --output-csv /path/to/output/metadata.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

PATTERNS = {
    # .npz files contain metadata and other object properties. The _\d+ captures the fracture index.
    "npz": re.compile(r"Fantastic_Breaks_v1/\d{2}/(?P<object_id>\d+)/meta_(?P<fracture_id>\d+)\.npz"),
    # _b_.*.ply files represent the 'broken' or fractured part of the object
    "b": re.compile(r"Fantastic_Breaks_v1/\d{2}/(?P<object_id>\d+)/model_b_(?P<fracture_id>\d+)\.ply"),
    # _c.ply files represent the 'complete' or unbroken object (applies to all fractures)
    "c": re.compile(r"Fantastic_Breaks_v1/\d{2}/(?P<object_id>\d+)/model_c\.ply"),
    # _r_.*.ply files represent the 'restored' or broken piece to be restored
    "r": re.compile(r"Fantastic_Breaks_v1/\d{2}/(?P<object_id>\d+)/model_r_(?P<fracture_id>\d+)\.ply"),
}

def match_file(path: Path) -> Optional[Tuple[str, str, str]]:
    """
    Match a file path to its kind ('npz', 'b', 'c', or 'r') and extract its object ID and fracture ID.
    
    Args:
        path (Path): The file path to check.
        
    Returns:
        Optional[Tuple[str, str, str]]: A tuple of (file_kind, object_id, fracture_id) if matched, else None.
    """
    for kind, pattern in PATTERNS.items():
        m = pattern.search(path.as_posix())
        if m:
            # Complete meshes ('c') do not have a fracture_id, so default to '0'.
            fracture_id = m.group("fracture_id") if "fracture_id" in pattern.groupindex else "0"
            return kind, m.group("object_id"), fracture_id
    return None

def infer_split(path: Path) -> str:
    """
    Infer the dataset split (train/val/test) from the directory path.
    Since the directory structure might omit the split names if the dataset 
    is moved, default to 'unknown' unless 'train', 'val', or 'test' is found.
    
    Args:
        path (Path): The file path to infer the split from.
        
    Returns:
        str: The split name if found ('train', 'val', 'test'), else 'unknown'.
    """
    lowered = [p.lower() for p in path.parts]
    for split in ("train", "val", "test", "training", "validation", "testing"):
        if split in lowered:
            return split.replace("ing", "").replace("ion", "")  # Normalize names
    return "unknown"

def collect_metadata(raw_root: Path) -> pd.DataFrame:
    """
    Traverse the raw dataset root directory to find all relevant files, group them by object ID
    and fracture ID, and build a pandas DataFrame containing the metadata for each fractured piece.
    
    Args:
        raw_root (Path): The root directory of the dataset.
        
    Returns:
        pd.DataFrame: A dataframe where each row corresponds to a single object fracture ID.
    """
    # Dictionary to keep track of files belonging to each object_id + fracture_id combination.
    # Format: {f"{object_id}_{fracture_id}": {"object_id": ..., "split": ..., "path_<kind>": ...}}
    records: Dict[str, Dict[str, str]] = {}
    
    # Complete paths are shared across fractures for the same object
    # Format: {object_id: path_c}
    complete_paths: Dict[str, str] = {}

    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
            
        # Check if the file matches any of our predefined patterns
        matched = match_file(path)
        if matched is None:
            continue
        
        kind, object_id, fracture_id = matched
        
        # Save complete paths to share across multiple fractures
        if kind == "c":
            complete_paths[object_id] = str(path.resolve())
            continue
            
        # Use a composite key for fractures to avoid overwriting pieces of the same object
        # The user requested no _0 suffix for the primary fracture.
        composite_id = object_id if fracture_id == "0" else f"{object_id}_{fracture_id}"
        
        # Initialize the record for this object_id if it doesn't exist yet
        rec = records.setdefault(
            composite_id,
            {
                "object_id": composite_id,
                "split": infer_split(path),
            }
        )
        
        # Save the absolute path for this specific file kind
        rec[f"path_{kind}"] = str(path.resolve())
    
    rows = []
    # Sort by object_id to ensure consistent ordering in the output
    for composite_id, rec in sorted(records.items()):
        # Retrieve the complete mesh path for the base object id, stripping the fracture suffix
        base_obj_id = composite_id.split("_")[0]
        if base_obj_id in complete_paths:
            rec["path_c"] = complete_paths[base_obj_id]
            
        # Ensure that every key exists even if the file is missing
        for kind in ("npz", "b", "c", "r"):
            rec.setdefault(f"path_{kind}", "")
            
        # Determine if this object has all the required files 
        rec["has_all_files"] = all(rec[f"path_{kind}"] for kind in ("npz", "b", "c", "r"))
        rows.append(rec)
    
    return pd.DataFrame(rows)

def main() -> None:
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Build metadata CSV for Fantastic Breaks-like dataset.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Raw dataset root directory to scan.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path where the output metadata CSV will be saved.")
    args = parser.parse_args()

    # Collect metadata into a dataframe
    df = collect_metadata(args.raw_root)
    
    # Ensure the output directory exists before saving
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    # Calculate statistics based on the dataframe
    total = len(df)
    complete = int(df["has_all_files"].sum()) if total > 0 else 0

    print(f"Saved metadata to: {args.output_csv}")
    print(f"Total objects: {total}")
    print(f"Complete objects: {complete}")
    print(f"Incomplete objects: {total - complete}")

    # Print a quick preview of the top 10 rows
    if total > 0:
        print("\nPreview:")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()