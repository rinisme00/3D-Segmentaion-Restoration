"""
Canonical Manifest Builder for 3D Fracture Classification.
Defines the unified schema and provides utilities for saving, loading, 
and validating object-disjoint splits.
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Canonical manifest columns
# Identity columns
COL_DATASET = 'dataset_name'        # 'fantastic_breaks' | 'breaking_bad'
COL_SUBSET = 'subset_category'      # e.g. 'artifact' | 'everyday/Vase'
COL_BASE_ID = 'base_object_id'      # Unique identity of the original unbroken mesh
COL_CASE_ID = 'fracture_case_id'    # ID of the specific fracture event
COL_VARIANT_ID = 'variant_id'       # ID of the specific piece (e.g. piece_0, model_b)

# Label columns
COL_LABEL = 'label'                 # 0 (Complete) | 1 (Broken)
COL_IS_COMPLETE = 'is_complete'     # bool

# Path columns
COL_PATH_MESH = 'file_path_mesh'    # Absolute path to .obj/.ply
COL_PATH_META = 'file_path_meta'    # Absolute path to .npz (if applicable)

# Split column
COL_SPLIT = 'split'                 # 'train' | 'val' | 'test'

REQUIRED_COLUMNS = [
    COL_DATASET, COL_SUBSET, COL_BASE_ID, COL_CASE_ID, COL_VARIANT_ID,
    COL_LABEL, COL_IS_COMPLETE, COL_PATH_MESH, COL_SPLIT
]

def create_empty_manifest() -> pd.DataFrame:
    """Returns an empty DataFrame with the canonical schema."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)

def validate_manifest(df: pd.DataFrame) -> bool:
    """
    Performs critical safety checks:
    1. Check for missing required columns.
    2. Check for mathematically perfect object-disjoint splitting (zero overlap).
    3. Check for existence of all mesh files.
    """
    # 1. Missing columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    # 2. Object-disjoint check
    splits = df[COL_SPLIT].unique()
    split_ids = {}
    for s in splits:
        if s == 'unknown' or pd.isna(s) or s == 'None': continue
        split_ids[s] = set(df[df[COL_SPLIT] == s][COL_BASE_ID])

    possible_splits = list(split_ids.keys())
    for i in range(len(possible_splits)):
        for j in range(i + 1, len(possible_splits)):
            s1 = possible_splits[i]
            s2 = possible_splits[j]
            overlap = split_ids[s1].intersection(split_ids[s2])
            if overlap:
                raise ValueError(f"CRITICAL: Data Leakage detected! {len(overlap)} base_object_ids "
                                 f"overlap between {s1} and {s2}. First 5: {list(overlap)[:5]}")

    return True

def assign_object_disjoint_splits(
    df: pd.DataFrame, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Shuffles unique base_object_ids and assigns them train/val/test splits.
    Ensures that all variants of a base object stay in the same split.
    """
    rng = np.random.default_rng(seed)
    
    unique_ids = df[COL_BASE_ID].unique()
    rng.shuffle(unique_ids)
    
    n = len(unique_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])
    
    def get_split(base_id):
        if base_id in train_ids: return 'train'
        if base_id in val_ids: return 'val'
        return 'test'
        
    df[COL_SPLIT] = df[COL_BASE_ID].apply(get_split)
    return df
