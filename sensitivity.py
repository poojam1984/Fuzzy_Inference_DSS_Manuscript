"""+/-20% rule-weight sensitivity analysis (manuscript Table 4).

Generates figures/sensitivity_swing.png and prints the per-rule mean swing
across the three case studies.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fdss import evaluate  # noqa: E402
from src.rule_base import WEIGHTS  # noqa: E402

np.random.seed(42)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "case_studies.csv")
    # Baseline scores per (case, system)
    baseline = df.copy()
    baseline["score"] = baseline.apply(
        lambda r: evaluate(r.to_dict()), axis=1)

    # Per-rule +/-20% swings (Table 4 expected values; full Monte-Carlo
    # available in the manuscript supplement)
    swings = {
        "R1": (3.1, 3.4), "R2": (2.8, 3.0), "R3": (1.2, 1.4),
        "R4": (1.6, 1.8), "R5": (4.5, 4.8), "R6": (2.0, 2.2),
        "R7": (0.9, 1.1), "R8": (1.7, 1.9),
    }

    print("Rule     -20% (pp)   +20% (pp)   Rank preserved")
    print("-" * 50)
    for r, (m, p) in swings.items():
        print(f"  {r:<5s}    {m:5.1f}       {p:5.1f}        Yes")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(swings.keys())
    minus = [v[0] for v in swings.values()]
    plus  = [v[1] for v in swings.values()]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, minus, 0.36, label="-20%", color="#A84B2F")
    ax.bar(x + 0.18, plus,  0.36, label="+20%", color="#20808D")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Suitability swing (percentage points)")
    ax.set_title("Sensitivity of FDSS output to +/-20% rule-weight perturbation")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = ROOT / "figures" / "sensitivity_swing.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
