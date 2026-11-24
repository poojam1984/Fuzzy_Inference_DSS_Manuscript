import math
from src.fdss import evaluate, label, load_cases

# height is the capability ratio rho = H_cap / H_req


def test_veto_low_safety():
    assert evaluate({"safety": 1, "ease": 5, "height": 2.0, "reuse": 100, "cost": 60}) < 25


def test_veto_low_height():
    assert evaluate({"safety": 8, "ease": 7, "height": 0.3, "reuse": 150, "cost": 50}) < 25


def test_high_suitability_case():
    assert evaluate({"safety": 9, "ease": 9, "height": 3.0, "reuse": 250, "cost": 20}) > 75


def test_all_cases_fire_a_rule():
    assert all(not math.isnan(evaluate(r)) for r in load_cases())


def test_label_boundaries():
    assert [label(v) for v in (10, 35, 65, 90)] == [
        "Unsuitable", "Marginally Suitable", "Suitable", "Highly Suitable"]
