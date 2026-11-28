"""Reproduce every number in the v2 manuscript:  python fdss_v2/run_all.py"""
import json, time, copy, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau, spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from fdss import FDSS, INPUT_SETS_ALL, UNIVERSE, label_cutoffs, mf, STAGES
from cases import load, CASE_LABEL
from benchmarks import ahp_weights, ahp_scores, topsis_scores, CRITERIA

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
R = {}
IN = ["safety", "ease", "height", "reuse", "cost"]
df = load(); base = FDSS()

def scores(model, d):
    return np.array([model.evaluate(**r[IN].to_dict()) for _, r in d.iterrows()])

# ---------- 1. baseline
rows, traces = [], {}
for i, r in df.iterrows():
    s, tr = base.evaluate(**r[IN].to_dict(), trace=True)
    rows.append(dict(case=r.case, system=r.system, safety=r.safety, ease=r.ease, required_m=r.required_m,
                     capability_m=r.capability_m, height_ratio=round(r.height_raw, 2), reuse=r.reuse, cost=r.cost,
                     feasible=bool(r.feasible), tech=tr["stage_scores"]["tech"], life=tr["stage_scores"]["life"],
                     score=s, label=base.label(s)))
    traces[f"{r.case}|{r.system}"] = {k: {a: round(b, 3) for a, b in v.items()} for k, v in tr["fired"].items()}
res = pd.DataFrame(rows)
res["rank_feasible"] = res.groupby("case").apply(
    lambda g: g.score.where(g.feasible).rank(ascending=False, method="min")).reset_index(level=0, drop=True)
res.to_csv(OUT / "baseline_scores.csv", index=False); json.dump(traces, open(OUT / "firing_traces.json", "w"), indent=1)
R["cutoffs"] = label_cutoffs()

# ---------- 2. verification
# coverage: every stage fires >=1 rule everywhere
cov = {}
def cover(stage, vars_):
    ok = 0; n = 0
    grids = [np.linspace(*UNIVERSE[v], 25) for v in vars_]
    for pt in np.array(np.meshgrid(*grids)).reshape(len(vars_), -1).T:
        mem = {v: base._fuzz(v, x) for v, x in zip(vars_, pt)}
        c, al = base._stage(stage, mem); n += 1; ok += int(bool(al) and max(al.values()) > 0)
    return ok / n
cov["A"] = cover("tech", ["safety", "height", "ease"]); cov["B"] = cover("life", ["reuse", "cost"]); cov["C"] = cover("suit", ["tech", "life"])
R["coverage"] = cov
# monotonicity (score non-decreasing in benefit inputs, non-increasing in cost) on 5000 random points
def mono(model, tot=3000):
    viol = {v: [0, 0.0] for v in IN}
    for _ in range(tot):
        x = {v: rng.uniform(*UNIVERSE[v]) for v in IN}
        s0 = model.evaluate(**x)
        for v in IN:
            lo, hi = UNIVERSE[v]; y = dict(x); y[v] = min(hi, x[v] + 0.1 * (hi - lo))
            dd = (model.evaluate(**y) - s0) * (-1 if v == "cost" else 1)
            if dd < -1e-6: viol[v][0] += 1; viol[v][1] = max(viol[v][1], -dd)
    return {v: dict(rate=a / tot, max_drop=m) for v, (a, m) in viol.items()}
R["monotonicity_product"] = mono(base); R["monotonicity_min"] = mono(FDSS(impl="min"))
# range of attainable scores
allv = [base.evaluate(**{v: rng.uniform(*UNIVERSE[v]) for v in IN}) for _ in range(5000)]
R["score_range_random"] = [float(np.min(allv)), float(np.max(allv))]
corner_lo = base.evaluate(safety=1, ease=1, height=0, reuse=0, cost=130)
corner_hi = base.evaluate(safety=10, ease=10, height=3, reuse=260, cost=0)
R["score_corners"] = [corner_lo, corner_hi]
# cross-check with scikit-fuzzy on Stage B (raw centroid)
try:
    import skfuzzy as fuzz; from skfuzzy import control as ctrl
    g = base.grid
    reuse = ctrl.Antecedent(np.linspace(0, 260, 261), "reuse"); cost = ctrl.Antecedent(np.linspace(0, 130, 261), "cost")
    out = ctrl.Consequent(g, "o")
    for n, p in INPUT_SETS_ALL["reuse"].items(): reuse[n] = fuzz.trapmf(reuse.universe, p)
    for n, p in INPUT_SETS_ALL["cost"].items(): cost[n] = fuzz.trapmf(cost.universe, p)
    from fdss import LEVEL_SETS
    for n, p in LEVEL_SETS.items(): out[n] = fuzz.trapmf(g, [p[0], p[1], p[2], p[3]])
    rmin = FDSS(impl="min"); rl = [ctrl.Rule(reuse[a[0][1]] & cost[a[1][1]], out[c]) for a, c, w in base.rules["life"]]
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rl)); diffs = []
    for _ in range(200):
        a, b = rng.uniform(0, 260), rng.uniform(0, 130)
        sim.input["reuse"] = a; sim.input["cost"] = b; sim.compute()
        mem = {"reuse": base._fuzz("reuse", a), "cost": base._fuzz("cost", b)}
        diffs.append(abs(sim.output["o"] - rmin._stage("life", mem)[0]))
    R["skfuzzy_stageB_max_abs_diff"] = float(np.max(diffs))
except Exception as e:
    R["skfuzzy_stageB_max_abs_diff"] = f"not run: {e}"
# timing
x = df.iloc[8][IN].to_dict(); t0 = time.perf_counter()
for _ in range(2000): base.evaluate(**x)
R["time_ms_per_alternative"] = (time.perf_counter() - t0) / 2000 * 1e3
t0 = time.perf_counter()
for _ in range(200): FDSS()
R["time_ms_build_model"] = (time.perf_counter() - t0) / 200 * 1e3

# ---------- 3. decision surface (others moderate)
sg = np.linspace(1, 10, 46); hg = np.linspace(0, 3, 61)
surf = np.array([[base.evaluate(safety=s, height=h, ease=5.5, reuse=100, cost=60) for s in sg] for h in hg])
np.savez(OUT / "surface.npz", s=sg, h=hg, z=surf)
R["surface_min_max"] = [float(surf.min()), float(surf.max())]

# ---------- 4. global sensitivity (Monte Carlo)
N = 600
def perturb_sets(delta):
    sets = {v: dict(s) for v, s in INPUT_SETS_ALL.items()}
    for v in IN:
        lo, hi = UNIVERSE[v]; span = hi - lo
        for n, p in sets[v].items():
            q = []
            for val in p:
                q.append(val if val in (lo, hi) else float(np.clip(val + rng.uniform(-delta, delta) * span, lo, hi)))
            sets[v][n] = tuple(sorted(q))
    return sets
def rand_weights(lo=0.7):
    w = {}
    for st in ("tech", "life", "suit"):
        for i in range(len(STAGES[st]())): w[(st, i)] = rng.uniform(lo, 1.0)
    return w
def noisy(d, sd_rating=0.5, rel=0.10):
    d = d.copy()
    for v in ("safety", "ease"): d[v] = np.clip(d[v] + rng.normal(0, sd_rating, len(d)), 1, 10)
    for v in ("reuse", "cost"): d[v] = np.clip(d[v] * (1 + rng.normal(0, rel, len(d))), *UNIVERSE[v])
    cap = d.capability_m * (1 + rng.normal(0, rel, len(d))); d["height"] = np.clip(cap / d.required_m, 0, 3)
    return d
cases = list(CASE_LABEL)
base_sc = {c: scores(base, df[df.case == c]) for c in cases}
def metrics(sc_pert, c):
    b = base_sc[c]
    tau = kendalltau(b, sc_pert).statistic
    return (0.0 if np.isnan(tau) else tau), int(np.argmax(b) == np.argmax(sc_pert)), float(np.mean(np.abs(b - sc_pert)))
scen = {"MF breakpoints ±5% of range": lambda d: (FDSS(input_sets=perturb_sets(0.05)), d),
        "MF breakpoints ±10% of range": lambda d: (FDSS(input_sets=perturb_sets(0.10)), d),
        "Rule weights U(0.7,1)": lambda d: (FDSS(weights=rand_weights()), d),
        "Input noise (σ=0.5 rating pts, 10% others)": lambda d: (base, noisy(d)),
        "Classic max–min implication": lambda d: (FDSS(impl="min"), d),
        "All combined (MF ±5%, weights, inputs)": lambda d: (FDSS(input_sets=perturb_sets(0.05), weights=rand_weights()), noisy(d)),
        }
sens = {}
for name, fn in scen.items():
    N_ = 1 if name.startswith("Classic") else N
    acc = {c: [] for c in cases}
    for _ in range(N_):
        # one model draw per iteration applied to all cases (fair comparison)
        m, _d = fn(df)
        for c in cases:
            d = df[df.case == c]
            m_, d_ = (m, _d[_d.case == c]) if name.startswith(("Input", "All")) else (m, d)
            acc[c].append(metrics(scores(m_, d_), c))
    sens[name] = {c: dict(kendall_tau_mean=float(np.mean([a[0] for a in v])), top1_same=float(np.mean([a[1] for a in v])),
                          mean_abs_change=float(np.mean([a[2] for a in v]))) for c, v in acc.items()}
for dm in ("bisector",):
    m = FDSS(defuzz=dm); sens[f"Defuzzification: {dm}"] = {}
    for c in cases:
        d = df[df.case == c]; t, t1, mad = metrics(scores(m, d), c)
        sens[f"Defuzzification: {dm}"][c] = dict(kendall_tau_mean=t, top1_same=float(t1), mean_abs_change=mad)
R["sensitivity"] = sens; R["sensitivity_N"] = N
# baseline margins
R["margins"] = {c: sorted(base_sc[c])[::-1][:3] for c in cases}
R["base_scores"] = {c: [round(float(x), 2) for x in base_sc[c]] for c in cases}

# ---------- 5. AHP / TOPSIS comparison (feasible alternatives)
w, cr = ahp_weights(); R["ahp_weights"] = dict(zip(CRITERIA, np.round(w, 3).tolist())); R["ahp_CR"] = cr
cmp = {}
for c in cases:
    d = df[(df.case == c)].reset_index(drop=True); X = d[IN].values.astype(float)
    feas = d.feasible.values
    f = scores(base, d); a = ahp_scores(X[feas], w); t = topsis_scores(X[feas], w)
    ff = f[feas]; names = d.system[feas].tolist()
    def tau(x, y): return float(kendalltau(x, y).statistic) if len(x) > 2 else (1.0 if np.argsort(-x).tolist() == np.argsort(-y).tolist() else -1.0)
    cmp[c] = dict(systems=names, fdss=np.round(ff, 1).tolist(), ahp=np.round(a, 3).tolist(), topsis=np.round(t, 3).tolist(),
                  tau_fdss_ahp=tau(ff, a), tau_fdss_topsis=tau(ff, t), tau_ahp_topsis=tau(a, t),
                  top_fdss=names[int(np.argmax(ff))], top_ahp=names[int(np.argmax(a))], top_topsis=names[int(np.argmax(t))])
R["benchmark"] = cmp

# ---------- 6. missing-data robustness (self-consistency)
def mean_impute(X, mask):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]; obs = ~mask[:, j]
        col[mask[:, j]] = col[obs].mean() if obs.any() else np.nan
    return X
miss = {}
for p in (0.1, 0.2, 0.3):
    res_m = {"FDSS": [], "AHP": [], "TOPSIS": []}; top = {"FDSS": [], "AHP": [], "TOPSIS": []}
    for rep in range(400):
        for c in cases:
            d = df[df.case == c]; X = d[IN].values.astype(float)
            m = rng.random(X.shape) < p
            if m.all(0).any() or m.all(1).any(): continue
            bF = base_sc[c]; bA = ahp_scores(X, w); bT = topsis_scores(X, w)
            Xi = mean_impute(X, m)
            fF = np.array([base.evaluate(**{v: (None if m[i, j] else X[i, j]) for j, v in enumerate(IN)}) for i in range(len(X))])
            for k, (b, s) in {"FDSS": (bF, fF), "AHP": (bA, ahp_scores(Xi, w)), "TOPSIS": (bT, topsis_scores(Xi, w))}.items():
                t = kendalltau(b, s).statistic; res_m[k].append(0.0 if np.isnan(t) else t); top[k].append(int(np.argmax(b) == np.argmax(s)))
    miss[str(p)] = {k: dict(tau=float(np.mean(v)), top1=float(np.mean(top[k])), n=len(v)) for k, v in res_m.items()}
R["missing"] = miss

json.dump(R, open(OUT / "results.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in R if k not in ("sensitivity",)}, indent=1, default=float)[:6000])
print(json.dumps(R["sensitivity"], indent=1)[:5000])
