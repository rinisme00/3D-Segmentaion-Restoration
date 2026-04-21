"""
Parser for Breaking Bad dataset.
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from .builder import (
    COL_DATASET, COL_SUBSET, COL_BASE_ID, COL_CASE_ID, 
    COL_VARIANT_ID, COL_LABEL, COL_IS_COMPLETE, 
    COL_PATH_MESH, COL_PATH_META, COL_SPLIT
)

def load_split_entries(split_dir: Path, subset: str, split: str) -> list[str]:
    """
    Load object entries for a given subset and split from .txt files.
    subset e.g. 'artifact' or 'everyday/Vase'
    """
    prefix = subset.split('/')[0]  # 'artifact' or 'everyday'
    filename = f"{prefix}.{split}.txt"
    filepath = split_dir / filename

    if not filepath.exists():
        return []

    with open(filepath, 'r') as f:
        entries = [line.strip() for line in f if line.strip()]

    # Filter for specific category if needed (e.g. everyday/Vase)
    if '/' in subset:
        entries = [e for e in entries if e.startswith(subset + '/')]

    return entries

def parse_breaking_bad(
    data_root: str | Path, 
    split_dir: str | Path,
    subsets: list[str] = ['artifact', 'everyday/Vase', 'everyday/Cup', 'everyday/Mug', 'everyday/Plate']
) -> pd.DataFrame:
    """
    Scans Breaking Bad data based on official splits and existing files.
    """
    data_root = Path(data_root)
    split_dir = Path(split_dir)
    records = []
    
    for subset in subsets:
        for split_name in ['train', 'val', 'test']:
            # official BB splits are often train/val. The main repo calls val 'test' in H5.
            # We'll preserve the split from the file.
            entries = load_split_entries(split_dir, subset, split_name)
            
            for entry in entries:
                # entry is e.g. 'artifact/73400_sf'
                obj_dir = data_root / entry
                if not obj_dir.exists():
                    continue
                
                base_id = entry.replace('/', '_') # Flat ID
                
                # 1. Complete object: mode_0/piece_0.obj
                complete_path = obj_dir / 'mode_0' / 'piece_0.obj'
                if complete_path.exists():
                    records.append({
                        COL_DATASET: 'breaking_bad',
                        COL_SUBSET: subset,
                        COL_BASE_ID: base_id,
                        COL_CASE_ID: "mode_0",
                        COL_VARIANT_ID: "piece_0",
                        COL_LABEL: 0,
                        COL_IS_COMPLETE: True,
                        COL_PATH_MESH: str(complete_path.resolve()),
                        COL_PATH_META: "",
                        COL_SPLIT: split_name
                    })
                
                # 2. Broken fragments: fractured_0/piece_*.obj
                # We typically only use fractured_0 for simplicity in classification
                fractured_dir = obj_dir / 'fractured_0'
                if fractured_dir.exists():
                    for piece_path in fractured_dir.glob("piece_*.obj"):
                        records.append({
                            COL_DATASET: 'breaking_bad',
                            COL_SUBSET: subset,
                            COL_BASE_ID: base_id,
                            COL_CASE_ID: "fractured_0",
                            COL_VARIANT_ID: piece_path.stem,
                            COL_LABEL: 1,
                            COL_IS_COMPLETE: False,
                            COL_PATH_MESH: str(piece_path.resolve()),
                            COL_PATH_META: "",
                            COL_SPLIT: split_name
                        })
                        
    return pd.DataFrame(records)
