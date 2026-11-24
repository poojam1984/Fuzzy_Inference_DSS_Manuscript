"""One-at-a-time +/-20% rule-weight sensitivity analysis (manuscript Table 5).

Each rule weight is multiplied by 0.8 and 1.2 in turn, all others held at
baseline. The swing is the absolute change in score (percentage points),
averaged over all 18 case-system combinations. A rank reversal is counted
when two systems in the same case swap strict order; ties created by a
perturbation are reported separately. Usage:  python -m src.sensitivity
"""
from __future__ import annotations
from statistics import median
from .fdss import evaluate, load_cases
from .rule_base import WEIGHTS

EPS = 1e-6


def compare(base: dict, new: dict):
    """Return (strict reversals, new ties) between systems within each case."""
    rev, ties = [], []
    keys = list(base)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] != b[0]:
                continue
            d0, d1 = base[a] - base[b], new[a] - new[b]
            if abs(d0) > EPS and abs(d1) > EPS and d0 * d1 < 0:
                rev.append((a[0], a[1], b[1]))
            elif abs(d0) > EPS and abs(d1) <= 1e-3:
                ties.append((a[0], a[1], b[1]))
    return rev, ties


def main() -> None:
    cases = load_cases()
    base = {(r["case"], r["system"]): evaluate(r) for r in cases}
    allv = []
    print("Rule   -20% mean  +20% mean  max |d|  Reversals  New ties")
    for k in WEIGHTS:
        means, mx, nrev, tie_list = [], 0.0, 0, []
        for f in (0.8, 1.2):
            w = dict(WEIGHTS); w[k] = WEIGHTS[k] * f
            new = {(r["case"], r["system"]): evaluate(r, w) for r in cases}
            d = [abs(new[c] - base[c]) for c in base]
            means.append(sum(d) / len(d)); mx = max(mx, max(d))
            rev, ties = compare(base, new)
            nrev += len(rev); tie_list += [(f, t) for t in ties]
        allv += means
        print(f"{k:<6s} {means[0]:9.2f}  {means[1]:9.2f}  {mx:7.2f}  {nrev:9d}  {tie_list}")
    print(f"\nMedian of the 16 mean swings: {median(allv):.3f} pp")


if __name__ == "__main__":
    main()
