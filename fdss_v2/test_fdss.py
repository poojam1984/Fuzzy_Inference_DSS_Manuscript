import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fdss import FDSS, label_cutoffs
from cases import load

m = FDSS()
GOOD = dict(safety=9, ease=9, height=2.5, reuse=220, cost=15)

def test_range_and_best_case():
    s = m.evaluate(**GOOD); assert 85 <= s <= 100

def test_poor_safety_is_unsuitable():
    assert m.evaluate(**{**GOOD, "safety": 1}) < label_cutoffs()[0]

def test_insufficient_height_is_unsuitable():
    assert m.evaluate(**{**GOOD, "height": 0.4}) < label_cutoffs()[0]

def test_cutoffs_ordered():
    c = label_cutoffs(); assert c == sorted(c) and len(c) == 3

def test_missing_input_still_returns_score():
    s = m.evaluate(safety=9, ease=None, height=2.5, reuse=220, cost=15); assert not math.isnan(s)

def test_full_coverage():
    for st, v in (("tech", dict(safety=4, height=1, ease=5)), ("life", dict(reuse=70, cost=50))):
        mem = {k: m._fuzz(k, x) for k, x in v.items()}; assert m._stage(st, mem)[1]

def test_case_feasibility_flags():
    d = load(); assert d.loc[(d.case == "C_90_storey") & (d.system == "Timber"), "feasible"].item() is False
    assert d.loc[(d.case == "C_90_storey") & (d.system == "Aluminium"), "feasible"].item() is True

def test_deterministic():
    assert m.evaluate(**GOOD) == FDSS().evaluate(**GOOD)
