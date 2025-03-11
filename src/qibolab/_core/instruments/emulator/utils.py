import numpy as np

GHZ_TO_HZ = 1e9
"""Converting GHz to Hz."""
HZ_TO_GHZ = 1e-9
"""Converting Hz to GHz."""


def merge_results(a: dict, b: dict) -> dict:
    """Merge dictionary together using np.column_stack."""
    merged = a.copy()
    for key, value in b.items():
        x = (a[key],) if key in a else ()
        merged[key] = np.column_stack(x + (value,))
    return merged
