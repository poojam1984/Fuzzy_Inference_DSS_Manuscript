"""Triangular membership functions for the FDSS.

Each input variable is partitioned into three overlapping triangular sets;
the output (suitability) into four. The (a, b, c) parameter triplets are
calibrated from the 10th, 50th and 90th percentiles of industry practice
in India (Mishra et al., 2026 — see manuscript Table 2).
"""
from __future__ import annotations
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ----------------------------- Antecedents -----------------------------

safety = ctrl.Antecedent(np.arange(1, 10.01, 0.1), "safety")
safety["poor"]    = fuzz.trimf(safety.universe, [1, 1, 5])
safety["average"] = fuzz.trimf(safety.universe, [3, 5.5, 8])
safety["good"]    = fuzz.trimf(safety.universe, [6, 10, 10])

ease = ctrl.Antecedent(np.arange(1, 10.01, 0.1), "ease")
ease["difficult"] = fuzz.trimf(ease.universe, [1, 1, 5])
ease["moderate"]  = fuzz.trimf(ease.universe, [3, 5.5, 8])
ease["easy"]      = fuzz.trimf(ease.universe, [6, 10, 10])

height = ctrl.Antecedent(np.arange(0, 800.01, 1), "height")
height["insufficient"] = fuzz.trimf(height.universe, [0, 0, 150])
height["sufficient"]   = fuzz.trimf(height.universe, [100, 250, 450])
height["optimal"]      = fuzz.trimf(height.universe, [350, 800, 800])

reuse = ctrl.Antecedent(np.arange(0, 260.01, 1), "reuse")
reuse["low"]      = fuzz.trimf(reuse.universe, [0, 0, 60])
reuse["moderate"] = fuzz.trimf(reuse.universe, [40, 100, 180])
reuse["high"]     = fuzz.trimf(reuse.universe, [140, 260, 260])

cost = ctrl.Antecedent(np.arange(0, 130.01, 0.5), "cost")
cost["low"]      = fuzz.trimf(cost.universe, [0, 0, 40])
cost["moderate"] = fuzz.trimf(cost.universe, [25, 60, 95])
cost["high"]     = fuzz.trimf(cost.universe, [80, 130, 130])


# ----------------------------- Consequent ------------------------------

suitability = ctrl.Consequent(np.arange(0, 100.01, 0.5), "suitability")
suitability["unsuitable"]           = fuzz.trimf(suitability.universe, [0, 0, 25])
suitability["marginally_suitable"]  = fuzz.trimf(suitability.universe, [15, 37.5, 60])
suitability["suitable"]             = fuzz.trimf(suitability.universe, [50, 70, 90])
suitability["highly_suitable"]      = fuzz.trimf(suitability.universe, [80, 100, 100])
