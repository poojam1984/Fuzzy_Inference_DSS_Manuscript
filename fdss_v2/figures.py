"""Generate manuscript figures (300 dpi) from results/ and the model."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).parent))
from fdss import INPUT_SETS, SUIT_SETS, LEVEL_SETS, UNIVERSE, mf, label_cutoffs
from cases import CASE_LABEL
HERE = Path(__file__).parent; FIG = HERE / "figures"; RES = HERE / "results"
R = json.load(open(RES / "results.json"))
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.labelsize": 9, "figure.dpi": 100, "savefig.dpi": 300})
def save(fig, name): fig.savefig(FIG / name, bbox_inches="tight", facecolor="white"); plt.close(fig)

# Fig 1: membership functions
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4)); axs = axs.ravel()
titles = {"safety": "(a) Safety performance (1–10)", "ease": "(b) Ease of installation (1–10)",
          "height": "(c) Height adequacy (capability ÷ required height)", "reuse": "(d) Reusability (cycles)",
          "cost": "(e) Operational cost (USD/h)"}
for ax, v in zip(axs, ["safety", "ease", "height", "reuse", "cost"]):
    x = np.linspace(*UNIVERSE[v], 600)
    for c, (n, p) in zip(OI, INPUT_SETS[v].items()): ax.plot(x, mf(x, p), color=c, lw=1.8, label=n)
    ax.set_title(titles[v], loc="left"); ax.set_ylim(-0.03, 1.08); ax.set_ylabel("Membership"); ax.legend(frameon=False, fontsize=7, loc={"height":"center","reuse":"upper center","cost":"upper center"}.get(v,"center right"), bbox_to_anchor={"height":(0.78,0.5)}.get(v))
ax = axs[5]; x = np.linspace(0, 100, 600)
for c, (n, p) in zip(OI, SUIT_SETS.items()): ax.plot(x, mf(x, p), color=c, lw=1.8, label=n)
for cut in R["cutoffs"]: ax.axvline(cut, color="#444", ls=":", lw=0.8)
ax.set_title("(f) Suitability output (0–100)", loc="left"); ax.set_ylim(-0.03, 1.08); ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5,-0.18), ncol=2)
fig.tight_layout(); save(fig, "fig1_membership_functions.png")

# Fig 2: architecture
fig, ax = plt.subplots(figsize=(7.4, 3.8)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 58)
def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="#333", lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.4)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1, color="#333"))
inp = "#E8F1FA"; st = "#FFF3D6"
box(1, 44, 22, 8, "Safety rating (1–10)", inp); box(1, 33, 22, 8, "Ease rating (1–10)", inp)
box(1, 22, 22, 8, "Height adequacy\n(capability ÷ required)", inp)
box(1, 8, 22, 8, "Reusability (cycles)", inp); box(1, -3, 22, 8, "Operational cost (USD/h)", inp)
box(32, 28, 21, 22, "Stage A\nTechnical performance\n20 rules", st)
box(32, -2, 21, 18, "Stage B\nLifecycle-economic\n9 rules", st)
box(62, 14, 16, 22, "Stage C\nSuitability\n9 rules", st)
box(84, 17, 15, 16, "Suitability\nscore (0–100),\nlabel and rule\ntrace", "#DDF3E8")
for y in (48, 37, 26): arr(23.5, y, 31.5, 39)
for y in (12, 1): arr(23.5, y, 31.5, 7)
arr(53.5, 39, 61.5, 28); arr(53.5, 7, 61.5, 22); arr(78.5, 25, 83.5, 25)
ax.text(57.5, 45, "technical score", fontsize=6.5, ha="center", style="italic", color="#555")
ax.text(57.5, 19, "lifecycle score", fontsize=6.5, ha="center", style="italic", color="#555")
ax.text(72, -1, "Feasibility filter: capability < required height\n→ alternative flagged infeasible", fontsize=6.8, color="#9a3b00", ha="center")
ax.text(50, 56, "Fuzzification → rule firing → product implication → max aggregation → centroid", ha="center", fontsize=7.4, style="italic")
ax.set_ylim(-5, 58); save(fig, "fig2_architecture.png")

# Fig 3: decision surface
S = np.load(RES / "surface.npz")
fig, ax = plt.subplots(figsize=(4.6, 3.6))
cs = ax.contourf(S["s"], S["h"], S["z"], levels=np.arange(0, 101, 10), cmap="viridis")
ax.contour(S["s"], S["h"], S["z"], levels=R["cutoffs"], colors="white", linewidths=0.8, linestyles="--")
ax.axhline(1.0, color="#D55E00", lw=1, ls="-"); ax.text(9.9, 1.04, "required height", color="#D55E00", ha="right", fontsize=7)
ax.set_xlabel("Safety rating (1–10)"); ax.set_ylabel("Height adequacy (ratio)")
cb = fig.colorbar(cs, ax=ax); cb.set_label("Suitability score")
save(fig, "fig3_decision_surface.png")

# Fig 4: scores by case
d = pd.read_csv(RES / "baseline_scores.csv"); systems = ["Fabric", "Timber", "Plastic", "Aluminium", "Steel", "Tunnel"]
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
for ax, c in zip(axs, CASE_LABEL):
    g = d[d.case == c].set_index("system").loc[systems]
    ax.bar(range(6), g.score, color=[OI[2] if f else "#BBBBBB" for f in g.feasible], edgecolor="#333", lw=0.6,
           hatch=None)
    for i, (sc, f) in enumerate(zip(g.score, g.feasible)):
        ax.text(i, sc + 1.5, f"{sc:.0f}" if f else "n/f", ha="center", fontsize=7)
    for cut in R["cutoffs"]: ax.axhline(cut, color="#444", ls=":", lw=0.7)
    ax.set_xticks(range(6)); ax.set_xticklabels(systems, rotation=45, ha="right", fontsize=7.5); ax.set_title(CASE_LABEL[c], loc="left"); ax.set_ylim(0, 100)
axs[0].set_ylabel("Suitability score (0–100)")
fig.text(0.5, -0.04, "Grey bars (n/f): not feasible, capability below required height. Dotted lines: label cut-offs.", ha="center", fontsize=7)
save(fig, "fig4_case_scores.png")

# Fig 5: sensitivity
sens = R["sensitivity"]; names = list(sens); cases = list(CASE_LABEL)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
y = np.arange(len(names))
for k, (ax, key, ttl) in enumerate(zip(axs, ["kendall_tau_mean", "top1_same"], ["Mean Kendall's τ vs baseline ranking", "Share of draws with unchanged top-ranked system"])):
    for j, c in enumerate(cases):
        ax.scatter([sens[n][c][key] for n in names], y + (j - 1) * 0.18, color=OI[j], s=22, label=CASE_LABEL[c], zorder=3)
    ax.set_title(ttl, loc="left", fontsize=8.5); ax.set_xlim(0, 1.05); ax.grid(axis="x", color="#ddd", lw=0.6)
axs[0].set_yticks(y); axs[0].set_yticklabels([n.replace("(", "\n(", 1) if len(n) > 30 else n for n in names], fontsize=7); axs[0].invert_yaxis()
axs[1].legend(frameon=False, fontsize=7, loc="lower left")
save(fig, "fig5_sensitivity.png")

# Fig 6: missing data
m = R["missing"]; ps = [0.1, 0.2, 0.3]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ax, key, ttl in zip(axs, ["tau", "top1"], ["Mean Kendall's τ vs own complete-data ranking", "Top-ranked system retained"]):
    for c, k in zip(OI, ["FDSS", "AHP", "TOPSIS"]):
        ax.plot([p * 100 for p in ps], [m[str(p)][k][key] for p in ps], "-o", color=c, label=k, lw=1.6, ms=4)
    ax.set_xlabel("Input cells deleted (%)"); ax.set_title(ttl, loc="left", fontsize=8.5); ax.set_ylim(0.4, 1.0); ax.set_xticks([10, 20, 30])
axs[0].legend(frameon=False, fontsize=7.5)
save(fig, "fig6_missing_data.png")
print("figures ok")
