import json
import random
from typing import List, Dict, Tuple

def group_based_split(source_ids: List[str], train_ratio=0.7, val_ratio=0.15) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits source IDs into train, val, and test sets.
    """
    random.shuffle(source_ids)
    n = len(source_ids)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_ids = source_ids[:train_end]
    val_ids = source_ids[train_end:val_end]
    test_ids = source_ids[val_end:]
    
    return train_ids, val_ids, test_ids

def filter_manifest_by_source(manifest: List[Dict], allowed_sources: List[str]) -> List[Dict]:
    """
    Filters a manifest keeping only records derived from the allowed source IDs.
    """
    allowed_set = set(allowed_sources)
    return [record for record in manifest if record["source_id"] in allowed_set]
