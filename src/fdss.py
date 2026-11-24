"""Weighted Mamdani fuzzy inference engine (manuscript Eqs. 4-8).

Usage:  python -m src.fdss      (from the repository root)
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .membership_functions import OUTPUT_MF, Z, trimf
from .rule_base import WEIGHTS, CONSEQUENT, firing_strengths

ROOT = Path(__file__).resolve().parent.parent
OUT_SETS = {k: trimf(Z, v) for k, v in OUTPUT_MF.items()}


def evaluate(x: dict, weights: dict | None = None) -> float:
    """Return the crisp suitability score z* in [0, 100] for one option."""
    w = WEIGHTS if weights is None else weights
    agg = np.zeros_like(Z)
    for k, alpha in firing_strengths(x).items():
        lam = min(1.0, abs(w[k]) * alpha)                      # Eq. 5
        clipped = np.minimum(lam, OUT_SETS[CONSEQUENT[k]])      # Eq. 6
        agg = np.maximum(agg, clipped)                          # Eq. 7
    if agg.sum() == 0:
        return float("nan")                                     # no rule fired
    return float((Z * agg).sum() / agg.sum())                   # Eq. 8


def label(score: float) -> str:
    if score < 25: return "Unsuitable"
    if score < 50: return "Marginally Suitable"
    if score < 80: return "Suitable"
    return "Highly Suitable"


def load_cases(path: Path = ROOT / "data" / "case_studies.csv") -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("safety", "ease", "h_cap", "h_req", "reuse", "cost"):
            r[k] = float(r[k])
        r["height"] = r["h_cap"] / r["h_req"]   # capability ratio rho
    return rows


def main() -> None:
    print("=== FDSS case study evaluations ===")
    case = None
    for r in load_cases():
        if r["case"] != case:
            case = r["case"]; print(f"\n{case}")
        s = evaluate(r)
        print(f"  {r['system']:<11s} rho={r['height']:5.2f}  {s:6.1f}  ({label(s)})")


if __name__ == "__main__":
    main()
