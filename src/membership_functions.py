"""Triangular membership functions for the FDSS (manuscript Table 3).

Each input has three overlapping triangular sets and the output has four.
Parameters are (a, b, c): membership is 0 at a, rises to 1 at b and
returns to 0 at c. Shoulder sets use a == b or b == c.
"""
from __future__ import annotations
import numpy as np

INPUT_MF = {
    "safety": {"poor": (1, 1, 5), "average": (3, 5.5, 8), "good": (6, 10, 10)},
    "ease": {"difficult": (1, 1, 5), "moderate": (3, 5.5, 8), "easy": (6, 10, 10)},
    # Height is the capability ratio rho = H_cap / H_req (dimensionless, capped at 3)
    "height": {"insufficient": (0, 0, 1.0), "sufficient": (0.8, 1.4, 2.2), "optimal": (1.8, 3.0, 3.0)},
    "reuse": {"low": (0, 0, 60), "moderate": (40, 100, 180), "high": (140, 260, 260)},
    "cost": {"low": (0, 0, 40), "moderate": (25, 60, 95), "high": (80, 130, 130)},
}
OUTPUT_MF = {
    "unsuitable": (0, 0, 25),
    "marginally_suitable": (15, 37.5, 60),
    "suitable": (50, 70, 90),
    "highly_suitable": (80, 100, 100),
}
# Output universe Z = [0, 100], discretised at 0.5-unit intervals (Eq. 8)
Z = np.arange(0, 100.0001, 0.5)


def trimf(x, abc):
    """Triangular membership grade of x (scalar or array) for set (a, b, c)."""
    a, b, c = abc
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b > a:
        m = (x > a) & (x < b)
        y[m] = (x[m] - a) / (b - a)
    if c > b:
        m = (x > b) & (x < c)
        y[m] = (c - x[m]) / (c - b)
    y[x == b] = 1.0
    return y


UNIVERSE = {"safety": (1, 10), "ease": (1, 10), "height": (0, 3.0),
            "reuse": (0, 260), "cost": (0, 130)}


def clamp(variable: str, value: float) -> float:
    """Clip a crisp input to the universe of discourse of its variable."""
    lo, hi = UNIVERSE[variable]
    return min(max(float(value), lo), hi)


def grade(variable: str, term: str, value: float) -> float:
    """Membership grade mu_A(x) of a crisp input value in a linguistic term."""
    return float(trimf([clamp(variable, value)], INPUT_MF[variable][term])[0])
