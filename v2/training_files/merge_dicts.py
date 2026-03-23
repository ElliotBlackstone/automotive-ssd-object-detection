# merge_dicts.py
import torch
import numpy as np

def merge_dicts_preserve_order(d1: dict, d2: dict) -> dict:
    """
    Merge two dictionaries with identical keys while preserving order.

    Inputs
    d1: Dictionary
    d2: Dictionary with same keys as d1

    Output
    Merged dictionary
    Example:
    d1 = {"a": [1, 2], "b": ["python", 8]}
    d2 = {"a": [3, "alpha"], "b": [2]}
    merge_dicts_preserve_order(d1, d2) -> {"a": [1, 2, 3, "alpha"], "b": ["python", 8, 2]}
    merge_dicts_preserve_order(d2, d1) -> {"a": [3, "alpha", 1, 2], "b": [2, "python", 8]}
    """
    if set(d1.keys()) != set(d2.keys()):
        raise KeyError("Dicts must have identical key sets.")

    out = {}
    for k in d1.keys():  # preserves key order from d1
        v1, v2 = d1[k], d2[k]

        # torch tensors
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            out[k] = torch.cat([v1, v2], dim=0)
            continue

        # numpy arrays
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            out[k] = np.concatenate([v1, v2], axis=0)
            continue

        # lists / tuples
        if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            if k == 'epochs':
                out[k] = list(v2)
            else:
                seq = list(v1) + list(v2)  # v1-order then v2-order
                out[k] = type(v1)(seq) if type(v1) is type(v2) else seq
            continue

        # sets are unordered
        if isinstance(v1, set) and isinstance(v2, set):
            out[k] = list(v1) + [x for x in v2 if x not in v1]  # insertion-style, no dups
            continue

        # fallback: keep both values
        out[k] = (v1, v2)

    return out