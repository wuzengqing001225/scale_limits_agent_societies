"""Paper figures, PNAS-style. Reads data/, writes paper/figs/*.pdf.
Style: Wong colorblind palette, sans-serif, 7-8 pt, thin axes, panel letters,
no in-axes titles (descriptions live in captions), fonttype 42.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "..", "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

# ---- Wong palette ----
BLUE, VERM, GREEN, ORANGE, PURPLE, SKY, GREY = \
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#555555"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "figure.dpi": 200,
})

def panel(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")

def J(name): return json.load(open(os.path.join(DATA, name)))
def JL(name): return [json.loads(l) for l in open(os.path.join(DATA, name))]
def sv(path, fig):
    fig.savefig(os.path.join(FIGS, path), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def dd(dct):  # {"10": v} -> (Ns, vals) sorted
    ks = sorted(int(k) for k in dct)
    return ks, [dct[str(k)] for k in ks]

# ================= Fig 1: anchor pairs =================
A = J("anchors_data.json")
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15))
ax = axes[0]
x, y = dd(A["A_direct"]); ax.semilogx(x, y, "o-", color=VERM, label="direct reciprocity")
x, y = dd(A["A_network"]); ax.semilogx(x, y, "s-", color=BLUE, label="network reciprocity")
ax.set_xlabel(r"$N$"); ax.set_ylabel("cooperator share"); ax.set_ylim(-0.05, 1.08)
ax.legend(loc="center right"); panel(ax, "A")
ax = axes[1]
x, y = dd(A["B_local"]); ax.semilogx(x, y, "o-", color=VERM, label="local sampling")
x, y = dd(A["B_global"]); ax.semilogx(x, y, "s-", color=BLUE, label="global sampling")
ax.set_xlabel(r"$N$"); ax.set_ylabel("consensus rate"); ax.set_ylim(-0.05, 1.08)
ax.legend(loc="center right"); panel(ax, "B")
ax = axes[2]
x, y = dd(A["C"]["herd"]); ax.semilogx(x, y, "o-", color=VERM, label="sequential influence")
x, y = dd(A["C"]["indep"]); ax.semilogx(x, y, "s-", color=BLUE, label="independent votes")
ax.axhline(0.692, color=GREY, lw=0.7, ls=(0, (4, 3)))
ax.text(12, 0.705, "analytic plateau", fontsize=6, color=GREY)
ax.set_xlabel(r"$N$"); ax.set_ylabel("aggregation accuracy"); ax.set_ylim(0.6, 1.03)
ax.legend(loc="center right"); panel(ax, "C")
fig.tight_layout(w_pad=1.4); sv("fig_anchors.pdf", fig)

# ================= Fig 2: exponent =================
r3 = J("r3_results.json"); r3p = J("r3_prime_results.json")
ci = J("r3_prime_reinforced.json")["theta_0.3"]["boot_CI95"]
fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.25), sharey=True)
for ax, res, key, letter, note in [
        (axes[0], r3, "N_half", "A", "bounded memory"),
        (axes[1], r3p, "N_03", "B", "memory bound removed")]:
    rows = [r for r in res["rows"] if r["p_respond"] == 0.8 and r.get(key)]
    x = np.array([r["f_tau"] for r in rows], float)
    y = np.array([r[key] for r in rows], float)
    b, a = np.polyfit(np.log(x), np.log(y), 1)
    xs = np.geomspace(x.min() * 0.85, x.max() * 1.18, 40)
    ax.loglog(xs, np.exp(a) * xs**b, "-", color=BLUE, lw=1.0, zorder=1)
    ax.loglog(x, y, "o", color=BLUE, mfc="white", mew=1.1, zorder=2)
    ax.set_xlabel(r"$f\,\tau$")
    ax.text(0.05, 0.97, note, transform=ax.transAxes, va="top", fontsize=7, color=GREY)
    slope = res["primary_fit"]["slope"]
    txt = f"slope $= {slope:.3f}$"
    if letter == "B":
        txt += f"\n95% CI $[{ci[0]:.3f},\\,{ci[1]:.3f}]$"
    ax.text(0.05, 0.86, txt, transform=ax.transAxes, va="top", fontsize=7)
    panel(ax, letter)
axes[0].set_ylabel(r"crossover scale $N_c$")
fig.tight_layout(w_pad=1.2); sv("fig_exponent.pdf", fig)

# ================= Fig 3: single-term flips =================
r4 = J("r4_results.json"); r6 = J("r6_results.json")
fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.25), sharey=True)
ax = axes[0]
for var, c, mk, lab in [("diluted", VERM, "o", r"update step $\propto 1/N$"),
                        ("normalized", BLUE, "s", "per-event update")]:
    rows = sorted((r for r in r4["rows"] if r["variant"] == var), key=lambda r: r["N"])
    ax.semilogx([r["N"] for r in rows], [r["lift"] for r in rows], mk + "-", color=c, label=lab)
ax.axhline(0, color=GREY, lw=0.5)
ax.set_xlabel(r"$N$"); ax.set_ylabel("effect size"); ax.legend(loc="center right")
panel(ax, "A")
ax = axes[1]
for arm, c, mk, lab in [("organic", VERM, "o", "organic transmission"),
                        ("board", BLUE, "s", "public record")]:
    rows = sorted((r for r in r6["rows"] if r["arm"] == arm), key=lambda r: r["N"])
    ax.semilogx([r["N"] for r in rows], [r["lift"] for r in rows], mk + "-", color=c, label=lab)
ax.axhline(0, color=GREY, lw=0.5)
ax.set_xlabel(r"$N$"); ax.legend(loc="center right")
panel(ax, "B")
fig.tight_layout(w_pad=1.2); sv("fig_flips.pdf", fig)

# ================= Fig 4: R5 paths =================
r5 = J("r5_results.json"); Ns = [10, 100, 1000]
fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.25))
ax = axes[0]
ax.semilogx(Ns, [r5["coverage_global"][str(n)] for n in Ns], "o-", color=VERM, label="global routing")
ax.semilogx(Ns, [r5["coverage_local"][str(n)] for n in Ns], "s-", color=BLUE, label="local routing")
ax.set_xlabel(r"$N$"); ax.set_ylabel("coverage"); ax.set_ylim(-0.05, 1.02)
ax.legend(loc="center left"); panel(ax, "A")
ax = axes[1]
ax.semilogx(Ns, [r5["lift_global_refusal"][str(n)] for n in Ns], "D-", color=GREEN, label="refusal variant")
ax.semilogx(Ns, [r5["compliance_component_global"][str(n)] for n in Ns], "o-", color=ORANGE, label="compliance component")
ax.semilogx(Ns, [r5["lift_global"][str(n)] for n in Ns], "^-", color=GREY, label="aggregate")
ax.axhline(0, color=GREY, lw=0.5)
ax.set_xlabel(r"$N$"); ax.set_ylabel("effect size")
ax.legend(loc="upper right"); panel(ax, "B")
fig.tight_layout(w_pad=1.4); sv("fig_r5.pdf", fig)

# ================= Fig 5: dose lattice =================
rows = JL("p6_main_raw.jsonl")
ks = [0, 1, 5, 25]; Nss = [40, 200, 1000]
M = np.full((4, 3), np.nan)
for i, k in enumerate(ks):
    for j, n in enumerate(Nss):
        sel = [r for r in rows if r["k"] == k and r["N"] == n and r["action"] in ("give", "keep")]
        if sel: M[i, j] = np.mean([r["action"] == "give" for r in sel])
fig, ax = plt.subplots(figsize=(3.0, 2.55))
im = ax.imshow(M, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
ax.set_xticks(range(3), [str(n) for n in Nss])
ax.set_yticks(range(4), ["0 (baseline)", "1", "5", "25"])
ax.set_xlabel(r"stated population size $N$")
ax.set_ylabel(r"number of reports $k$")
for i in range(4):
    for j in range(3):
        col = "black" if M[i, j] > 0.78 else "white"
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7, color=col)
ax.text(1.0, -0.78, r"$\beta_{\log k}=+0.48\;(z=3.17)\quad\beta_{\log N}=0.00\;(z=0.00)$",
        ha="center", fontsize=6.5)
cb = fig.colorbar(im, shrink=0.85, pad=0.03); cb.set_label("give rate", fontsize=7)
cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.5)
fig.tight_layout(); sv("fig_dose.pdf", fig)

# ================= Fig 6: probe matrix =================
def rates_from(path, dedupe):
    best = {}
    for r in JL(path):
        if r.get("_header"): continue
        best[tuple(r.get(k) for k in dedupe)] = r
    out = {}
    for cell in set(r["cell"] for r in best.values()):
        sel = [r for r in best.values() if r["cell"] == cell and r["action"] in ("give", "keep")]
        if sel: out[cell] = np.mean([r["action"] == "give" for r in sel])
    return out
cells6 = [("P1_B", "no information"), ("P1_T", "targeted note"),
          ("P1_X", "third-party placebo"), ("P2_count_1000", r"count, $N{=}1000$"),
          ("P2_prop_8", r"proportion, $N{=}8$"), ("P2_prop_1000", r"proportion, $N{=}1000$")]
son = rates_from("probe_p1p2_raw.jsonl", ("cell", "i"))
opu = rates_from("p4_claude_opus_raw.jsonl", ("cell", "variant", "i"))
gpt = rates_from("p4_codex_gpt-5.5_raw.jsonl", ("cell", "variant", "i"))
def rates_n_from(path, dedupe):
    best = {}
    for r in JL(path):
        if r.get("_header"): continue
        best[tuple(r.get(k) for k in dedupe)] = r
    out = {}
    for cell in set(r["cell"] for r in best.values()):
        sel = [r for r in best.values() if r["cell"] == cell and r["action"] in ("give", "keep")]
        if sel: out[cell] = (np.mean([r["action"] == "give" for r in sel]), len(sel))
    return out
def wilson(p, n, z=1.96):
    d = 1 + z*z/n; c = p + z*z/(2*n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c-h)/d, (c+h)/d
son_n = rates_n_from("probe_p1p2_raw.jsonl", ("cell", "i"))
opu_n = rates_n_from("p4_claude_opus_raw.jsonl", ("cell", "variant", "i"))
gpt_n = rates_n_from("p4_codex_gpt-5.5_raw.jsonl", ("cell", "variant", "i"))
models = [("claude-sonnet-4-6", son_n, BLUE, "o"),
          ("claude-opus (alias)", opu_n, VERM, "s"),
          ("gpt-5.5", gpt_n, GREEN, "D")]
fig, ax = plt.subplots(figsize=(3.6, 2.8))
ypos = np.arange(len(cells6))[::-1]
off = {0: 0.22, 1: 0.0, 2: -0.22}
for j, (name, dct, col, mk) in enumerate(models):
    for i, (c, _) in enumerate(cells6):
        if c not in dct: continue
        p, n = dct[c]
        lo, hi = wilson(p, n)
        ax.plot([lo, hi], [ypos[i]+off[j]]*2, color=col, lw=1.1, solid_capstyle="butt")
        ax.plot(p, ypos[i]+off[j], mk, color=col, mfc="white", mew=1.0, ms=3.6,
                label=name if i == 0 else None)
ax.set_yticks(ypos, [lab for _, lab in cells6])
ax.set_xlabel("give rate"); ax.set_xlim(-0.03, 1.03)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3,
          columnspacing=0.9, handlelength=1.2)
fig.subplots_adjust(bottom=0.28, left=0.34)
sv("fig_probes.pdf", fig)

# ================= Fig 7: prospective (two panels: axelrod + DIV-1) =====
r7 = J("r7_results.json")
fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.4, 2.6),
                              gridspec_kw={"width_ratios": [1.15, 1]})
styles = {("II_Grudger", "G"): (BLUE, "o", "-", "pool II, per-capita budget"),
          ("II_Grudger", "S"): (BLUE, "o", (0, (4, 2.5)), "pool II, fixed match length"),
          ("I_WSLS", "G"): (VERM, "s", "-", "pool I, per-capita budget"),
          ("I_WSLS", "S"): (VERM, "s", (0, (4, 2.5)), "pool I, fixed match length")}
ax.axvspan(24, 48, color="#dfe8f3", alpha=0.9, lw=0, zorder=0)
ax.text(34, 1.05, "predicted\nflip window", fontsize=6, ha="center", color=GREY)
for (pool, arm), (c, mk, ls, lab) in styles.items():
    rows = sorted((r for r in r7["rows"] if r["pool"] == pool and r["arm"] == arm),
                  key=lambda r: r["N"])
    xs = [r["N"] for r in rows]; ys = [r["delta"] for r in rows]
    ax.semilogx(xs, ys, ls=ls, color=c, lw=1.0, zorder=2, label=lab)   # prediction
    ax.semilogx(xs, ys, mk, color=c, mfc="white", mew=1.1, ls="none", zorder=3)  # observed
ax.axhline(0, color=GREY, lw=0.5)
ax.set_xlabel(r"$N$"); ax.set_ylabel("payoff advantage per turn")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncols=2,
          columnspacing=1.0, handlelength=1.5)
panel(ax, "A")
# Panel B: sealed cross-architecture test (DIV-1)
d_r = J("div1_reasoner_summary.json")["rates"]
d_s = J("div1_sonnet_summary.json")["rates"]
Nsx = [8, 40, 200, 1000]
axb.semilogx(Nsx, [d_r[f"T_N{n}"]["give"] for n in Nsx], "o-", color=BLUE,
             lw=1.0, mfc="white", mew=1.1, label="graded arm, observed")
axb.semilogx(Nsx, [d_s[f"T_N{n}"]["give"] for n in Nsx], "s-", color=VERM,
             lw=1.0, mfc="white", mew=1.1, label="threshold arm, observed")
for n, pred in ((40, 0.396), (200, 0.682)):
    axb.plot([n, n], [pred-0.20, pred+0.20], color=BLUE, lw=3.5, alpha=0.25,
             solid_capstyle="butt", zorder=0)
    axb.plot(n, pred, "_", color=BLUE, ms=8, mew=1.4, zorder=1)
axb.text(90, 0.13, "sealed interior\npredictions $\\pm0.20$", fontsize=6,
         color=BLUE, ha="center")
axb.set_xlabel(r"stated population size $N$")
axb.set_ylabel("give rate"); axb.set_ylim(-0.05, 1.08)
axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncols=1,
           columnspacing=1.0, handlelength=1.5)
panel(axb, "B")
fig.subplots_adjust(bottom=0.38, wspace=0.42)
sv("fig_prospective.pdf", fig)

# ================= Fig 8: framework schematic =================
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
def box(x, y, w, h, text, fc="#eef3fb", ec=BLUE, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, color="#444", lw=0.9))
box(0.2, 4.4, 2.9, 1.3, "Mechanism implementation\n$\\Rightarrow$ structural quantities $q_j(N)$\n(re-encounter, coverage, monitoring,\ntime budget, update dilution)")
box(0.2, 2.4, 2.9, 1.3, "Agent population\n$\\Rightarrow$ single-decision probes\nmeasure gains $\\Delta_j$\n(effective inputs, formats)")
box(0.2, 0.4, 2.9, 1.3, "Observation plan\n$\\Rightarrow$ protocol checks\n(ceilings, extensive observables,\ntime budgets, final rounds)", fc="#fdf5e3", ec="#9a6b12")
box(4.0, 2.3, 2.3, 1.6, "Path table\n$E(N)=\\sum_j q_j(N)\\,\\Delta_j(N)$\n+ interactions", fc="#fff")
box(7.2, 3.6, 2.6, 1.3, "Per-path predictions\n(Level 1, curve class\n+ applicable range)", fc="#e9f6ee", ec="#1e7a46")
box(7.2, 1.7, 2.6, 1.3, "Aggregate $E(N)$ only if\nevery gain identified,\nelse `aggregate unidentified'", fc="#e9f6ee", ec="#1e7a46")
arrow(3.1, 5.0, 4.2, 3.6); arrow(3.1, 3.05, 4.0, 3.05); arrow(3.1, 1.1, 4.2, 2.4)
arrow(6.3, 3.3, 7.2, 4.1); arrow(6.3, 2.9, 7.2, 2.3)
ax.text(5.1, 0.55, "closed checklist fixed in advance; out-of-checklist explanation = audit failure;\npredictions logged before results, failures reported",
        ha="center", fontsize=6.2, color="#444")
sv("fig_framework.pdf", fig)

print("done:", sorted(os.listdir(FIGS)))
