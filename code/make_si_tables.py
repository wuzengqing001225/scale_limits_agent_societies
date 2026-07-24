"""Regenerates every numeric table in the paper's Supplementary Information
directly from the summary/analysis files in ../data (no hand-typed numbers).
Output: si_tables.tex fragments printed to stdout; see the SI source for how
they are included. Usage: python3 make_si_tables.py > si_tables.tex
"""
import json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
out = []
def lattice(title, label, path, order=("0", "1", "5", "25")):
    j = json.load(open(path)); r = j.get("rates") or j.get("cells")
    rows = []
    for k in order:
        cells = []
        for N in ("40", "200", "1000"):
            v = r.get(f"k{k}_N{N}")
            cells.append(f"{v['give']:.3f} ({v['n']})" if v else "--")
        rows.append(f"$k={k}$ & " + " & ".join(cells) + r" \\")
    out.append("\\begin{table}[h]\\centering\\small\n\\caption{%s}\n\\label{%s}\n\\begin{tabular}{@{}lccc@{}}\\toprule\n & $N=40$ & $N=200$ & $N=1000$ \\\\ \\midrule\n%s\n\\bottomrule\\end{tabular}\\end{table}" % (title, label, "\n".join(rows)))
lattice("Count-format dose lattice, primary engine.", "tab:si-p6", f"{D}/p6_main_summary.json")
for eng in ("reasoner", "gemini", "gpt55"):
    lattice(f"Count-format lattice, {eng}.", f"tab:si-p6c-{eng}", f"{D}/p6c_{eng}_summary.json")
for eng in ("sonnet", "reasoner"):
    lattice(f"Percentage-format lattice, {eng}.", f"tab:si-div1b-{eng}", f"{D}/div1b_{eng}_summary.json")
for eng in ("reasoner", "sonnet"):
    j = json.load(open(f"{D}/div1_{eng}_summary.json")); r = j["rates"]
    rows = []
    for c, lab in (("B", "No information"), ("T", "Report ($k=3$)")):
        cells = [f"{r[f'{c}_N{N}']['give']:.3f} ({r[f'{c}_N{N}']['n']})" for N in (8, 40, 200, 1000)]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out.append("\\begin{table}[h]\\centering\\small\n\\caption{Sealed test, %s.}\n\\begin{tabular}{@{}lcccc@{}}\\toprule\n & $N=8$ & $N=40$ & $N=200$ & $N=1000$ \\\\ \\midrule\n%s\n\\bottomrule\\end{tabular}\\end{table}" % (eng, "\n".join(rows)))
print("\n\n".join(out))

# ---- Part 3: baseline-adjusted models and prospectively logged tests ----
# (appended to stdout; split into si_tables_part3.tex for the SI source)
G = json.load(open(f"{D}/baseline_glm_results.json"))
_order = [("p6_main", "count", "claude-sonnet (primary)"),
          ("p6c_reasoner", "count", "deepseek-reasoner"),
          ("p6c_gemini", "count", "gemini-3.5-flash"),
          ("p6c_gpt55", "count", "gpt-5.5"),
          ("div1b_reasoner", "percentage", "deepseek-reasoner"),
          ("div1b_sonnet", "percentage", "claude-sonnet")]
_rows = []
for key, fmt, eng in _order:
    g = G[key]; c = g["coefficients"]
    def _cell(nm, c=c):
        e = c[nm]
        return f"${e['b']:+.2f}$ $[{e['CI95'][0]:.2f},{e['CI95'][1]:.2f}]$"
    _rows.append(f"{fmt} & {eng} & {g['method']} & {g['n']} & "
                 + " & ".join(_cell(nm) for nm in ("R", "R_logk", "logN", "R_logN")) + r" \\")
print("\n\n% ---- part3: baseline-adjusted GLM ----")
print("\\begin{table}[h]\\centering\\scriptsize\n\\caption{Baseline-adjusted logistic models.}\n\\label{tab:si-baseline-glm}\n\\begin{tabular}{@{}llllllll@{}}\\toprule\nFormat & Engine & Fit & $n$ & $R$ & $R\\log k$ & $\\log N$ & $R \\times \\log N$ \\\\ \\midrule\n" + "\n".join(_rows) + "\n\\bottomrule\\end{tabular}\\end{table}")

r7 = json.load(open(f"{D}/r7_results.json"))
_rows = []
for r in sorted(r7["rows"], key=lambda r: (r["pool"], r["arm"], r["N"])):
    arm = "per-capita budget" if r["arm"] == "G" else "fixed match length"
    _rows.append(f"{r['pool'].replace('_', ' ')} & {arm} & {r['N']} & ${r['delta']:+.3f}$ \\\\")
print("\n% ---- part3: R7 filed quantities ----")
print("\\begin{table}[h]\\centering\\scriptsize\n\\caption{Twenty registered quantities of the prospective third-party code test; filed and observed values coincide.}\n\\label{tab:si-r7}\n\\begin{tabular}{@{}llll@{}}\\toprule\nPool & Arm & $N$ & Payoff advantage per turn \\\\ \\midrule\n" + "\n".join(_rows) + "\n\\bottomrule\\end{tabular}\\end{table}")

_labels = {
 "monotone_rank": "Give rate strictly monotone in $N$",
 "anchor_N8_le_0.2": r"Anchor: give $\le 0.2$ at $N=8$",
 "anchor_N1000_ge_0.8": r"Anchor: give $\ge 0.8$ at $N=1000$",
 "mid_N40_pred_0.396_pm_0.20": r"Interior: $0.396 \pm 0.20$ at $N=40$",
 "mid_N200_pred_0.682_pm_0.20": r"Interior: $0.682 \pm 0.20$ at $N=200$",
 "divergence_rise_ge_0.4": r"Give-rate rise of at least $0.4$ across the tested range",
 "T_flat_spread_lt_0.15": r"Flatness: report-cell give-rate spread $< 0.15$",
 "Delta_flat_spread_lt_0.15": r"Flatness: spread of differences from same-run baselines $< 0.15$",
 "Delta_positive_all_N": r"Sanction direction: baseline-minus-report difference positive at every $N$",
 "flat_change_le_0.15": r"Flatness: end-to-end give-rate change $\le 0.15$",
}
d_r = json.load(open(f"{D}/div1_reasoner_summary.json"))["sealed_verdicts"]
d_s = json.load(open(f"{D}/div1_sonnet_summary.json"))["sealed_verdicts"]
_allk = list(dict.fromkeys(list(d_r.keys()) + list(d_s.keys())))
_rows = []
for k in _allk:
    lab = _labels.get(k, k.replace("_", r"\_"))
    m = lambda v: ("pass" if v[k] else "fail") if k in v else "--"
    _rows.append(f"{lab} & {m(d_r)} & {m(d_s)} \\\\")
print("\n% ---- part3: DIV-1 adjudication ----")
print("\\begin{table}[h]\\centering\\scriptsize\n\\caption{Per-criterion adjudication of the cross-architecture test.}\n\\label{tab:si-div1-adjud}\n\\begin{tabular}{@{}lll@{}}\\toprule\nRegistered criterion & Graded arm & Flatness arm \\\\ \\midrule\n" + "\n".join(_rows) + "\n\\bottomrule\\end{tabular}\\end{table}")

_crit = {"i_dose_sig": r"(i) dose effect $\beta_{\log k} > 0$, $|z| > 1.96$",
         "ii_CI_logN_contains_0": r"(ii) 95\% CI for $\beta_{\log N}$ contains $0$",
         "iii_antidiag_spread_gt_0.15": r"(iii) anti-diagonal give-rate spread $> 0.15$"}
GL = json.load(open(f"{D}/div_glm_results.json"))
_rows = []
for ck, clab in _crit.items():
    cells = ["pass" if GL[k]["sealed"][ck] else "fail"
             for k in ("p6c_gemini", "p6c_gpt55", "p6c_reasoner")]
    _rows.append(f"{clab} & " + " & ".join(cells) + r" \\")
print("\n% ---- part3: P6c criteria ----")
print("\\begin{table}[h]\\centering\\scriptsize\n\\caption{Registered criteria for the count-format lattice on further engines.}\n\\label{tab:si-p6c-criteria}\n\\begin{tabular}{@{}lccc@{}}\\toprule\nRegistered criterion & gemini-3.5-flash & gpt-5.5 & deepseek-reasoner \\\\ \\midrule\n" + "\n".join(_rows) + "\n\\bottomrule\\end{tabular}\\end{table}")
