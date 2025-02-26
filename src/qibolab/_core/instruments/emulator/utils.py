import numpy as np

GHZ_TO_HZ = 1e9
"""Converting GHz to Hz."""
HZ_TO_GHZ = 1e-9
"""Converting Hz to GHz."""


def merge_results(a: dict, b: dict) -> dict:
    """Merge dictionary together using np.column_stack."""
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    for key, value in b.items():
        a[key] = np.column_stack((a[key], value))
    return a
