"""TOPSIS benchmark with entropy weights — illustrative comparison.

Applies TOPSIS with entropy-derived criterion weights to the same six
formwork alternatives evaluated by the FDSS. The purpose is to illustrate
how a classical compensatory MCDM method handles the same crisp-numeric
decision matrix — it is NOT a head-to-head empirical validation.

No expert panel has been independently elicited to provide ground-truth
rankings; formal empirical validation is identified as future work in
the paper.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """Entropy-based weights for the criteria columns of `matrix`."""
    norm = matrix / matrix.sum(axis=0)
    k = 1.0 / np.log(matrix.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -k * np.nansum(norm * np.log(norm + 1e-12), axis=0)
    d = 1 - e
    return d / d.sum()


def topsis(matrix: np.ndarray, weights: np.ndarray,
           benefit: list[bool]) -> np.ndarray:
    """Return closeness coefficients (higher = better)."""
    # 1. Normalise
    r = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    v = r * weights
    # 2. Ideal / anti-ideal
    ideal     = np.where(benefit, v.max(axis=0), v.min(axis=0))
    anti      = np.where(benefit, v.min(axis=0), v.max(axis=0))
    # 3. Distances
    dp = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    dn = np.sqrt(((v - anti) ** 2).sum(axis=1))
    return dn / (dp + dn)


def main() -> None:
    # Illustrative decision matrix: rows=systems, cols=criteria
    # (Safety, Ease, Height, Reuse, Cost) — averaged across cases
    matrix = np.array([
        [6.5, 7.0, 200, 80,  50],  # Aluminium
        [7.0, 6.0, 450, 250, 90],  # Steel
        [5.5, 7.5, 100, 30,  30],  # Timber
        [6.0, 8.0,  80, 25,  25],  # Fabric
        [5.0, 6.5, 120, 40,  35],  # Plastic
        [6.0, 5.5, 300, 180, 75],  # Tunnel
    ])
    benefit = [True, True, True, True, False]  # cost is a cost criterion
    w = entropy_weights(matrix)
    cc = topsis(matrix, w, benefit)
    print(f"TOPSIS entropy weights: {np.round(w, 3)}")
    for sys_name, c in zip(
        ["Aluminium", "Steel", "Timber", "Fabric", "Plastic", "Tunnel"], cc
    ):
        print(f"  {sys_name:<10s} CC = {c:.3f}")
    print()
    print("Limitations of TOPSIS for this problem:")
    print("  - Requires crisp numeric inputs; linguistic judgements must be")
    print("    pre-converted to point estimates, discarding uncertainty.")
    print("  - Cannot express overlapping linguistic categories or")
    print("    partial-truth memberships.")
    print("  - Sensitive to normalisation scheme and weighting method.")
    print()
    print("These are the methodological gaps the FDSS addresses.")


if __name__ == "__main__":
    main()
