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
