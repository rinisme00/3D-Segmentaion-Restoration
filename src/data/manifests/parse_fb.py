"""
Parser for Fantastic Breaks dataset.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
import pandas as pd
from .builder import (
    create_empty_manifest, COL_DATASET, COL_SUBSET, COL_BASE_ID, 
    COL_CASE_ID, COL_VARIANT_ID, COL_LABEL, COL_IS_COMPLETE, 
    COL_PATH_MESH, COL_PATH_META, COL_SPLIT
)

# Pattern matches: .../Fantastic_Breaks_v1/{category}/{object_id}/model_{type}_{fracture_id}.ply
# Or: .../Fantastic_Breaks_v1/{category}/{object_id}/model_c.ply
PATTERNS = {
    "b": re.compile(r"Fantastic_Breaks_v1/(?P<category>[^/]+)/(?P<obj_id>[^/]+)/model_b_(?P<fracture_id>\d+)\.ply"),
    "c": re.compile(r"Fantastic_Breaks_v1/(?P<category>[^/]+)/(?P<obj_id>[^/]+)/model_c\.ply"),
}

def parse_fantastic_breaks(data_root: str | Path) -> pd.DataFrame:
    """
    Scans data_root for FB files and returns a manifest DataFrame.
    """
    data_root = Path(data_root)
    records = []
    
    # We find all .ply files
    for path in data_root.rglob("*.ply"):
        path_str = path.as_posix()
        
        match_b = PATTERNS["b"].search(path_str)
        if match_b:
            records.append({
                COL_DATASET: 'fantastic_breaks',
                COL_SUBSET: match_b.group("category"),
                COL_BASE_ID: match_b.group("obj_id"),
                COL_CASE_ID: match_b.group("fracture_id"),
                COL_VARIANT_ID: f"model_b_{match_b.group('fracture_id')}",
                COL_LABEL: 1,  # Broken
                COL_IS_COMPLETE: False,
                COL_PATH_MESH: str(path.resolve()),
                COL_PATH_META: str(path.with_name(f"meta_{match_b.group('fracture_id')}.npz").resolve()),
                COL_SPLIT: 'unknown'
            })
            continue
            
        match_c = PATTERNS["c"].search(path_str)
        if match_c:
            records.append({
                COL_DATASET: 'fantastic_breaks',
                COL_SUBSET: match_c.group("category"),
                COL_BASE_ID: match_c.group("obj_id"),
                COL_CASE_ID: "0",
                COL_VARIANT_ID: "model_c",
                COL_LABEL: 0,  # Complete
                COL_IS_COMPLETE: True,
                COL_PATH_MESH: str(path.resolve()),
                COL_PATH_META: str(path.with_name("meta_0.npz").resolve()), # Fallback
                COL_SPLIT: 'unknown'
            })
            
    df = pd.DataFrame(records)
    
    # Check meta existence - if it doesn't exist, clear it
    if not df.empty:
        df[COL_PATH_META] = df[COL_PATH_META].apply(lambda p: p if os.path.exists(p) else "")
        
    return df
