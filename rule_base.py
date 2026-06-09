"""The eight expert-weighted IF-THEN rules for the FDSS.

Weights were elicited from the same twelve-expert pilot panel that
calibrated the membership functions. R1 and R2 act as hard veto
conditions (negative weight).
"""
from __future__ import annotations
from skfuzzy import control as ctrl
from .membership_functions import (
    safety, ease, height, reuse, cost, suitability,
)


def build_rules():
    """Return the eight Mamdani rules. Weights are stored separately
    in WEIGHTS for use by the sensitivity analysis."""
    r1 = ctrl.Rule(safety["poor"], suitability["unsuitable"])
    r2 = ctrl.Rule(height["insufficient"], suitability["unsuitable"])
    r3 = ctrl.Rule(
        safety["average"] & height["sufficient"]
        & (ease["moderate"] | reuse["moderate"] | cost["moderate"]),
        suitability["marginally_suitable"],
    )
    r4 = ctrl.Rule(
        safety["good"] & height["sufficient"]
        & (ease["moderate"] | reuse["moderate"] | cost["moderate"]),
        suitability["suitable"],
    )
    r5 = ctrl.Rule(
        safety["good"] & height["optimal"]
        & ease["easy"] & reuse["high"] & cost["low"],
        suitability["highly_suitable"],
    )
    r6 = ctrl.Rule(
        safety["good"] & height["optimal"]
        & (ease["easy"] | reuse["high"] | cost["low"]),
        suitability["suitable"],
    )
    r7 = ctrl.Rule(
        safety["average"] & height["sufficient"]
        & ease["difficult"] & reuse["low"] & cost["high"],
        suitability["marginally_suitable"],
    )
    r8 = ctrl.Rule(
        safety["good"] & height["optimal"]
        & ease["moderate"] & reuse["moderate"] & cost["moderate"],
        suitability["suitable"],
    )
    return [r1, r2, r3, r4, r5, r6, r7, r8]


# Baseline weights (R1, R2 are negative veto)
WEIGHTS = {
    "R1": -1.0,
    "R2": -1.0,
    "R3": +0.6,
    "R4": +0.8,
    "R5": +1.0,
    "R6": +0.9,
    "R7": +0.4,
    "R8": +0.8,
}
