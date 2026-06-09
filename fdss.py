"""Mamdani Fuzzy Inference Decision-Support System for formwork selection.

Usage
-----
    python src/fdss.py

Loads the three case-study input vectors from data/case_studies.csv and
prints the defuzzified suitability score for each of six candidate
formwork systems.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from skfuzzy import control as ctrl

# allow `python src/fdss.py` from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rule_base import build_rules  # noqa: E402


def make_simulator() -> ctrl.ControlSystemSimulation:
    system = ctrl.ControlSystem(build_rules())
    return ctrl.ControlSystemSimulation(system)


def evaluate(inputs: dict) -> float:
    """Run the FDSS on a single 5-tuple of inputs and return a crisp
    suitability score in [0, 100]."""
    sim = make_simulator()
    sim.input["safety"] = inputs["safety"]
    sim.input["ease"]   = inputs["ease"]
    sim.input["height"] = inputs["height"]
    sim.input["reuse"]  = inputs["reuse"]
    sim.input["cost"]   = inputs["cost"]
    sim.compute()
    return float(sim.output["suitability"])


def label(score: float) -> str:
    if score < 25:   return "Unsuitable"
    if score < 50:   return "Marginally Suitable"
    if score < 80:   return "Suitable"
    return "Highly Suitable"


def main() -> None:
    csv = ROOT / "data" / "case_studies.csv"
    df = pd.read_csv(csv)
    print("=== FDSS Case Study Evaluations ===\n")
    for case, group in df.groupby("case"):
        print(f"Case {case}")
        for _, row in group.iterrows():
            score = evaluate(row.to_dict())
            print(f"  {row['system']:<14s} {score:6.2f}  ({label(score)})")
        print()


if __name__ == "__main__":
    main()
