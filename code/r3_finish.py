"""Salvage R3: parse rows already printed to /tmp/r3_nc_scaling.log (process was
reaped by sandbox between calls), run the one missing config (p=0.4), fit, adjudicate
per the ORIGINAL preregistration (relative threshold 0.5*plateau).
"""
import ast, json, os, numpy as np
from r3_nc_scaling import measure_config

LOG = "/tmp/r3_nc_scaling.log"
rows = []
for line in open(LOG):
    line = line.strip()
    if line.startswith("{'f':"):
        rows.append(ast.literal_eval(line))
have = {(r["f"], r["tag_decay"], r["p_respond"]) for r in rows}
need = [dict(f=f, tag_decay=0.1, p_respond=0.8) for f in (1, 2, 3, 6, 10)] + \
       [dict(f=3, tag_decay=0.05, p_respond=0.8), dict(f=3, tag_decay=0.2, p_respond=0.8),
        dict(f=3, tag_decay=0.1, p_respond=0.4)]
for c in need:
    if (c["f"], c["tag_decay"], c["p_respond"]) not in have:
        print("running missing:", c, flush=True)
        rows.append(measure_config(**c))
prim = [r for r in rows if r["p_respond"] == 0.8 and r["N_half"]]
x = np.log([r["f_tau"] for r in prim]); y = np.log([r["N_half"] for r in prim])
b, a = np.polyfit(x, y, 1)
yhat = a + b * x
r2 = 1 - float(np.sum((y - yhat) ** 2)) / float(np.sum((y - y.mean()) ** 2))
pf = [r["N_half"] for r in rows if r["f_tau"] == 60.0 and r["p_respond"] == 0.8]
out = {"experiment": "R3_nc_scaling", "rows": rows,
       "primary_fit": {"slope": round(float(b), 3), "intercept": round(float(a), 3),
                        "R2": round(r2, 4), "n_configs": len(prim)},
       "R3_P_pass": bool(0.75 <= b <= 1.25 and r2 > 0.90),
       "product_form_check_ftau60": {"N_half_values": pf,
            "within_25pct": bool(max(pf) / min(pf) <= 1.25) if len(pf) == 2 else None},
       "secondary_p_effect": [r for r in rows if r["p_respond"] == 0.4]}
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "r3_results.json")
json.dump(out, open(p, "w"), indent=1)
print("slope=", round(float(b), 3), "R2=", round(r2, 4), "pass=", out["R3_P_pass"],
      "product_form=", out["product_form_check_ftau60"])
