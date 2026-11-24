"""The eight expert-weighted IF-THEN rules (manuscript Table 4).

Weights were elicited from the 35 pilot-study participants. A negative
weight marks a veto rule (R1, R2) whose consequent is always Unsuitable;
the magnitude |w_k| scales the firing strength (Eq. 5).
"""
from __future__ import annotations
from .membership_functions import grade

WEIGHTS = {"R1": -1.0, "R2": -1.0, "R3": 0.6, "R4": 0.8,
           "R5": 1.0, "R6": 0.9, "R7": 0.4, "R8": 0.8}

CONSEQUENT = {"R1": "unsuitable", "R2": "unsuitable",
              "R3": "marginally_suitable", "R4": "suitable",
              "R5": "highly_suitable", "R6": "suitable",
              "R7": "marginally_suitable", "R8": "suitable"}


def firing_strengths(x: dict) -> dict:
    """Firing strength alpha_k of each rule (Eq. 4): AND = min, OR = max."""
    g = lambda v, t: grade(v, t, x[v])
    # "(Ease/Reuse/Cost = Moderate)" in R3/R4 is read as "at least Moderate":
    # each of the three inputs is graded Moderate-or-better, and the OR is a max.
    at_least = lambda v, mid, best: max(g(v, mid), g(v, best))
    any_moderate = max(at_least("ease", "moderate", "easy"),
                       at_least("reuse", "moderate", "high"),
                       at_least("cost", "moderate", "low"))
    # "Height = Sufficient" in R3, R4 and R7 is read as "at least Sufficient":
    # a system with an ample margin (Optimal) also has sufficient height.
    h_suff = at_least("height", "sufficient", "optimal")
    any_favourable = max(g("ease", "easy"), g("reuse", "high"), g("cost", "low"))
    return {
        "R1": g("safety", "poor"),
        "R2": g("height", "insufficient"),
        "R3": min(g("safety", "average"), h_suff, any_moderate),
        "R4": min(g("safety", "good"), h_suff, any_moderate),
        "R5": min(g("safety", "good"), g("height", "optimal"), g("ease", "easy"),
                  g("reuse", "high"), g("cost", "low")),
        "R6": min(g("safety", "good"), g("height", "optimal"), any_favourable),
        "R7": min(g("safety", "average"), h_suff, g("ease", "difficult"),
                  g("reuse", "low"), g("cost", "high")),
        "R8": min(g("safety", "good"), g("height", "optimal"), g("ease", "moderate"),
                  g("reuse", "moderate"), g("cost", "moderate")),
    }
