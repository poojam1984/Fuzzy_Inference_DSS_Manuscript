"""Unit tests for the FDSS engine."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.fdss import evaluate, label


def test_veto_low_safety():
    """R1 should drive output toward 'Unsuitable' when safety = poor."""
    s = evaluate({"safety": 1, "ease": 5, "height": 200,
                  "reuse": 100, "cost": 60})
    assert s < 50, f"Expected low score, got {s}"


def test_veto_low_height():
    """R2 should drive output toward 'Unsuitable' when height insufficient."""
    s = evaluate({"safety": 8, "ease": 7, "height": 50,
                  "reuse": 150, "cost": 50})
    assert s < 50, f"Expected low score, got {s}"


def test_high_suitability_case():
    """R5 should fire strongly with all-favourable inputs."""
    s = evaluate({"safety": 9, "ease": 9, "height": 600,
                  "reuse": 220, "cost": 20})
    assert s > 70, f"Expected high score, got {s}"


def test_label_boundaries():
    assert label(10)  == "Unsuitable"
    assert label(35)  == "Marginally Suitable"
    assert label(65)  == "Suitable"
    assert label(90)  == "Highly Suitable"


def test_output_in_range():
    """All evaluations must lie in [0, 100]."""
    s = evaluate({"safety": 5, "ease": 5, "height": 250,
                  "reuse": 100, "cost": 60})
    assert 0 <= s <= 100
