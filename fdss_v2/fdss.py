"""Hierarchical Mamdani fuzzy inference DSS for formwork-system selection (v2).

Three small Mamdani sub-systems replace the earlier flat eight-rule base:
  Stage A  technical performance  = f(safety, height adequacy, ease)      20 rules
  Stage B  lifecycle-economic     = f(reusability, operational cost)       9 rules
  Stage C  suitability            = f(technical, lifecycle-economic)       9 rules
Operators: AND = min, OR = max, implication = product (Larsen; 'min' optional), aggregation = max,
centroid defuzzification on a fixed grid. Rule weights w_k in [0, 1] scale the
firing strength (default 1). Missing antecedents are treated as "don't care".
"""
from __future__ import annotations
import itertools
import numpy as np

# ---------------------------------------------------------------- membership
def mf(x, p):
    """Trapezoidal MF with breakpoints p=(a,b,c,d); triangles have b==c."""
    a, b, c, d = p
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), (x >= a).astype(float))
    right = np.where(d > c, (d - x) / (d - c + 1e-12), (x <= d).astype(float))
    return np.clip(np.minimum(np.minimum(left, right), 1.0), 0.0, 1.0)

def tri(a, b, c):  return (a, b, b, c)

# universes (lo, hi)
UNIVERSE = {"safety": (1, 10), "ease": (1, 10), "height": (0, 3),
            "reuse": (0, 260), "cost": (0, 130), "tech": (0, 100),
            "life": (0, 100), "suit": (0, 100)}

INPUT_SETS = {  # Ruspini partitions: memberships of each variable sum to 1 everywhere
    "safety": {"Poor": (1, 1, 3, 5), "Average": (3, 5, 6, 8), "Good": (6, 8, 10, 10)},
    "ease":   {"Difficult": (1, 1, 3, 5), "Moderate": (3, 5, 6, 8), "Easy": (6, 8, 10, 10)},
    "height": {"Insufficient": (0, 0, 0.9, 1.1), "Adequate": (0.9, 1.1, 1.5, 1.9), "Ample": (1.5, 1.9, 3, 3)},
    "reuse":  {"Low": (0, 0, 30, 60), "Moderate": (30, 60, 140, 180), "High": (140, 180, 260, 260)},
    "cost":   {"Low": (0, 0, 20, 40), "Moderate": (20, 40, 80, 100), "High": (80, 100, 130, 130)},
}
# intermediate scores (Ruspini partitions)
LEVEL_SETS = {"Low": tri(0, 0, 50), "Medium": tri(0, 50, 100), "High": tri(50, 100, 100)}
SUIT_SETS = {"Unsuitable": tri(0, 0, 25), "Marginally Suitable": tri(15, 37.5, 60),
             "Suitable": tri(50, 70, 90), "Highly Suitable": tri(80, 100, 100)}
OUT_SETS = {"tech": LEVEL_SETS, "life": LEVEL_SETS, "suit": SUIT_SETS}
INPUT_SETS_ALL = {**INPUT_SETS, "tech": LEVEL_SETS, "life": LEVEL_SETS}

# ---------------------------------------------------------------- rule bases
def rules_A():
    """Stage A (20 rules). Two veto-type rules (Poor safety -> Low; Insufficient height -> Low)
    plus the 18 combinations of {Average, Good} x {Adequate, Ample} x Ease, whose consequent
    follows ordinal points 2*S + H + E (S, H in {1,2}; E in {0,1,2}): >=7 High, 5-6 Medium, else Low."""
    S = ["Poor", "Average", "Good"]; H = ["Insufficient", "Adequate", "Ample"]
    E = ["Difficult", "Moderate", "Easy"]
    out = [([("safety", "Poor")], "Low", 1.0), ([("height", "Insufficient")], "Low", 1.0)]
    for si, hi, ei in itertools.product((1, 2), (1, 2), range(3)):
        pts = 2 * si + hi + ei
        c = "High" if pts >= 7 else "Medium" if pts >= 5 else "Low"
        out.append(([("safety", S[si]), ("height", H[hi]), ("ease", E[ei])], c, 1.0))
    return out

def rules_B():
    R = ["Low", "Moderate", "High"]; C = ["High", "Moderate", "Low"]  # index = cost advantage
    out = []
    for ri, ci in itertools.product(range(3), range(3)):
        pts = ri + ci
        c = "High" if pts >= 3 else "Medium" if pts == 2 else "Low"
        out.append(([("reuse", R[ri]), ("cost", C[ci])], c, 1.0))
    return out

C_TABLE = {("Low", "Low"): "Unsuitable", ("Low", "Medium"): "Unsuitable", ("Low", "High"): "Unsuitable",
           ("Medium", "Low"): "Marginally Suitable", ("Medium", "Medium"): "Suitable", ("Medium", "High"): "Suitable",
           ("High", "Low"): "Suitable", ("High", "Medium"): "Suitable", ("High", "High"): "Highly Suitable"}
def rules_C():
    return [([("tech", t), ("life", l)], c, 1.0) for (t, l), c in C_TABLE.items()]

STAGES = {"tech": rules_A, "life": rules_B, "suit": rules_C}

# ---------------------------------------------------------------- engine
class FDSS:
    def __init__(self, input_sets=None, weights=None, defuzz="centroid", n_grid=401, impl="product", agg="max"):
        self.impl = impl; self.agg = agg
        self.sets = {v: dict(s) for v, s in (input_sets or INPUT_SETS_ALL).items()}
        self.out_sets = {k: dict(v) for k, v in OUT_SETS.items()}
        self.rules = {k: f() for k, f in STAGES.items()}
        if weights is not None:  # dict: (stage, idx) -> w
            for (st, i), w in weights.items():
                a, c, _ = self.rules[st][i]; self.rules[st][i] = (a, c, w)
        self.defuzz = defuzz
        self.grid = np.linspace(0, 100, n_grid)
        self._mu_out = {k: {n: mf(self.grid, p) for n, p in s.items()} for k, s in self.out_sets.items()}
        # raw centroids of the extreme level sets, used to normalise stage scores to 0-100
        lo = self._cent(self._mu_out["tech"]["Low"]); hi = self._cent(self._mu_out["tech"]["High"])
        self._norm = (lo, hi)

    def _cent(self, mu):
        s = mu.sum()
        if self.defuzz == "bisector":
            cs = np.cumsum(mu); return float(self.grid[np.searchsorted(cs, cs[-1] / 2)])
        return float((mu * self.grid).sum() / s) if s > 0 else float("nan")

    def _fuzz(self, var, x):
        return {n: float(mf(x, p)) for n, p in self.sets[var].items()}

    def _stage(self, stage, mem):
        """mem: dict var -> {set: mu} (var missing => don't care)."""
        alpha = {}
        for ante, cons, w in self.rules[stage]:
            vals = [mem[v][s] for v, s in ante if v in mem]
            if not vals: continue
            a = w * min(vals)
            if a > alpha.get(cons, 0.0): alpha[cons] = a
        if not alpha: return float("nan"), {}
        agg = np.zeros_like(self.grid)
        for cons, a in alpha.items():
            t = (a * self._mu_out[stage][cons]) if self.impl == "product" else np.minimum(a, self._mu_out[stage][cons])
            agg = agg + t if self.agg == "sum" else np.maximum(agg, t)
        return self._cent(agg), alpha

    def evaluate(self, safety=None, ease=None, height=None, reuse=None, cost=None, trace=False):
        raw = dict(safety=safety, ease=ease, height=height, reuse=reuse, cost=cost)
        mem = {v: self._fuzz(v, x) for v, x in raw.items() if x is not None and not np.isnan(x)}
        lo, hi = self._norm
        scores, fired = {}, {}
        for st in ("tech", "life"):
            c, al = self._stage(st, mem)
            scores[st] = float("nan") if np.isnan(c) else float(np.clip((c - lo) / (hi - lo) * 100, 0, 100))
            fired[st] = al
        if np.isnan(scores["tech"]) or np.isnan(scores["life"]):
            # a stage with no usable antecedent: pass the other stage through as "don't care"
            mem2 = {}
            for st in ("tech", "life"):
                if not np.isnan(scores[st]):
                    mem2[st] = {n: float(mf(scores[st], p)) for n, p in LEVEL_SETS.items()}
        else:
            mem2 = {st: {n: float(mf(scores[st], p)) for n, p in LEVEL_SETS.items()} for st in ("tech", "life")}
        if not mem2: return (float("nan"), {}) if trace else float("nan")
        c, al = self._stage("suit", mem2); fired["suit"] = al
        return (c, dict(stage_scores=scores, fired=fired)) if trace else c

    # crossover points between adjacent output sets -> linguistic labels
    def label(self, score):
        names = list(SUIT_SETS)
        mu = [float(mf(score, SUIT_SETS[n])) for n in names]
        return names[int(np.argmax(mu))]

def label_cutoffs():
    g = np.linspace(0, 100, 100001); names = list(SUIT_SETS); cuts = []
    for a, b in zip(names[:-1], names[1:]):
        ma, mb = mf(g, SUIT_SETS[a]), mf(g, SUIT_SETS[b])
        lo, hi = SUIT_SETS[b][0], SUIT_SETS[a][3]
        m = (g >= lo) & (g <= hi)
        cuts.append(float(g[m][np.argmin(np.abs(ma[m] - mb[m]))]))
    return cuts
