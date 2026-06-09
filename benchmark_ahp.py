"""AHP benchmark — illustrative comparison.

Implements Saaty's Analytic Hierarchy Process with an illustrative pairwise
comparison matrix for the five top-ranked formwork-selection criteria. The
purpose of this script is to highlight methodological differences between AHP
and the fuzzy inference DSS — it is NOT a head-to-head empirical validation.

Note: No expert panel has been independently elicited to rank the case-study
alternatives. Formal validation against expert consensus or completed projects
is identified in the paper as future work.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Saaty pairwise matrix for the five criteria (illustrative; CR < 0.10)
PAIRWISE = np.array([
    # Safety  Ease   Height  Reuse  Cost
    [1.0,    3.0,    2.0,    4.0,   5.0],   # Safety
    [1/3,    1.0,    1/2,    2.0,   3.0],   # Ease
    [1/2,    2.0,    1.0,    3.0,   4.0],   # Height
    [1/4,    1/2,    1/3,    1.0,   2.0],   # Reuse
    [1/5,    1/3,    1/4,    1/2,   1.0],   # Cost
])


def priority_vector(matrix: np.ndarray) -> np.ndarray:
    """Eigenvector method for criterion weights."""
    eigvals, eigvecs = np.linalg.eig(matrix)
    idx = np.argmax(eigvals.real)
    w = eigvecs[:, idx].real
    return w / w.sum()


def consistency_ratio(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    eigvals = np.linalg.eigvals(matrix).real
    lam_max = eigvals.max()
    ci = (lam_max - n) / (n - 1)
    ri_table = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    return ci / ri_table[n]


def main() -> None:
    w = priority_vector(PAIRWISE)
    cr = consistency_ratio(PAIRWISE)
    print("=== AHP Illustrative Benchmark ===")
    print(f"AHP criterion weights: {dict(zip(['safety','ease','height','reuse','cost'], np.round(w, 3)))}")
    print(f"Consistency ratio CR = {cr:.3f}  (<= 0.10 acceptable)")
    print()
    print("Limitations of AHP for this problem:")
    print("  - Requires crisp pairwise comparisons; linguistic inputs must be")
    print("    forced onto Saaty's 1-9 scale.")
    print("  - Cannot natively express partial truth or overlapping categories")
    print("    (e.g. 'safety performance is between Average and Good').")
    print("  - Consistency must be re-checked whenever inputs change.")
    print()
    print("These are the methodological gaps the FDSS addresses.")
    print("No expert-panel ground truth is available in this repository;")
    print("formal empirical validation is identified as future work.")


if __name__ == "__main__":
    main()
