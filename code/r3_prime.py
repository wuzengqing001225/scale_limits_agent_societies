"""R3': N_c quantitative prediction, amended design (registered in 实验计划.md
Amendment R3' BEFORE running). Changes vs R3: (1) no memory cap — dense boolean
tag matrix = M->inf limit of original mechanics, everything else identical
(random perfect matching, victim gossips to f others excluding self/keeper,
p_respond defect toward tagged, per-slot decay -> per-tag decay); (2) ABSOLUTE
threshold N_03 = N where informed_frac crosses 0.30.
Prediction R3'-P: slope of log N_03 on log(f*tau) in [0.75,1.25], R^2>0.90;
(f=6,tau=10) vs (f=3,tau=20) N_03 within 25%.
Chunked with resume (sandbox reaps background jobs): run per-config, then --fit.
"""
import numpy as np, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARTIAL = os.path.join(HERE, "..", "data", "r3p_partial.jsonl")
OUT = os.path.join(HERE, "..", "data", "r3_prime_results.json")
THETA = 0.30

def run_once(N, f, p_respond, tag_decay, rounds_total=150, burn=50, seed=0):
    rng = np.random.default_rng(seed)
    coop_prob = rng.uniform(0.4, 0.9, N)
    tags = np.zeros((N, N), dtype=bool)   # tags[holder, about]
    m = N - (N % 2)
    total_dec = informed_dec = 0
    for rnd in range(rounds_total):
        perm = rng.permutation(N)[:m]
        a, b_ = perm[::2], perm[1::2]
        n_pairs = len(a)
        a_tag, b_tag = tags[a, b_], tags[b_, a]
        if rnd >= burn:
            informed_dec += int(a_tag.sum()) + int(b_tag.sum())
            total_dec += 2 * n_pairs
        r_a, r_b = rng.random(n_pairs), rng.random(n_pairs)
        mv_a = np.where(a_tag, (r_a > p_respond), (r_a < coop_prob[a]))
        mv_b = np.where(b_tag, (r_b > p_respond), (r_b < coop_prob[b_]))
        for me_arr, other_arr, moves in [(a, b_, mv_a), (b_, a, mv_b)]:
            for idx in np.where(~moves)[0]:
                keeper, victim = me_arr[idx], other_arr[idx]
                cands = np.arange(N)
                cands = cands[(cands != victim) & (cands != keeper)]
                recip = rng.choice(cands, size=min(f, len(cands)), replace=False)
                tags[recip, keeper] = True
        ti, tj = np.nonzero(tags)
        if len(ti):
            drop = rng.random(len(ti)) < tag_decay
            tags[ti[drop], tj[drop]] = False
    return informed_dec / total_dec if total_dec else 0.0

CONFIGS = ([dict(f=f, tag_decay=0.1, p_respond=0.8) for f in (1, 2, 3, 6, 10)] +
           [dict(f=3, tag_decay=0.05, p_respond=0.8), dict(f=3, tag_decay=0.2, p_respond=0.8),
            dict(f=3, tag_decay=0.1, p_respond=0.4)])

def grid_for(c):
    tau = 1.0 / c["tag_decay"]
    N_pred = max(12, 1.17 * c["f"] * tau)  # keep_rate~0.35 -> N_03 ~ 0.35*f*tau/0.3
    g = sorted(set([10] + [max(10, int(round(N_pred * mfac))) for mfac in (0.25, 0.5, 1, 2, 4)]))
    return g

def key(c, N, s):
    return f"{c['f']}_{c['tag_decay']}_{c['p_respond']}_{N}_{s}"

def load_done():
    done = {}
    if os.path.exists(PARTIAL):
        for line in open(PARTIAL):
            d = json.loads(line); done[d["key"]] = d["informed"]
    return done

def run_config(ci, seeds=2):
    c = CONFIGS[ci]
    done = load_done()
    with open(PARTIAL, "a") as fo:
        for N in grid_for(c):
            for s in range(seeds):
                k = key(c, N, s)
                if k in done: continue
                v = run_once(N, c["f"], c["p_respond"], c["tag_decay"], seed=s)
                fo.write(json.dumps({"key": k, "f": c["f"], "tag_decay": c["tag_decay"],
                                      "p_respond": c["p_respond"], "N": N, "seed": s,
                                      "informed": round(v, 4)}) + "\n")
                fo.flush(); print(ci, c["f"], c["tag_decay"], N, s, round(v, 4), flush=True)

def crossing(vals):  # vals: {N: informed}; absolute threshold, log-interp
    ns = sorted(vals)
    for i in range(len(ns) - 1):
        v1, v2 = vals[ns[i]], vals[ns[i + 1]]
        if v1 >= THETA and v2 < THETA:
            ln = np.log(ns[i]) + (v1 - THETA) / (v1 - v2) * (np.log(ns[i + 1]) - np.log(ns[i]))
            return float(np.exp(ln))
    return None

def fit():
    done = load_done()
    recs = [json.loads(l) for l in open(PARTIAL)]
    rows = []
    for c in CONFIGS:
        sub = [r for r in recs if (r["f"], r["tag_decay"], r["p_respond"]) ==
               (c["f"], c["tag_decay"], c["p_respond"])]
        vals = {}
        for r in sub:
            vals.setdefault(r["N"], []).append(r["informed"])
        vals = {n: float(np.mean(v)) for n, v in vals.items()}
        tau = 1.0 / c["tag_decay"]
        rows.append({**c, "tau": tau, "f_tau": c["f"] * tau,
                     "curve": {str(n): round(vals[n], 4) for n in sorted(vals)},
                     "N_03": (round(crossing(vals), 2) if crossing(vals) else None)})
    prim = [r for r in rows if r["p_respond"] == 0.8 and r["N_03"]]
    x = np.log([r["f_tau"] for r in prim]); y = np.log([r["N_03"] for r in prim])
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - float(np.sum((y - (a + b * x)) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    pf = [r["N_03"] for r in rows if r["f_tau"] == 60.0 and r["p_respond"] == 0.8]
    out = {"experiment": "R3_prime", "theta": THETA, "rows": rows,
           "primary_fit": {"slope": round(float(b), 3), "intercept": round(float(a), 3),
                            "R2": round(r2, 4), "n_configs": len(prim)},
           "R3p_P_pass": bool(0.75 <= b <= 1.25 and r2 > 0.90),
           "product_form_ftau60": {"values": pf,
                "within_25pct": bool(max(pf) / min(pf) <= 1.25) if len(pf) == 2 else None},
           "secondary_p_effect": [r for r in rows if r["p_respond"] == 0.4]}
    json.dump(out, open(OUT, "w"), indent=1)
    print("R3' slope=", round(float(b), 3), "R2=", round(r2, 4), "pass=", out["R3p_P_pass"],
          "product60=", pf)

def fit_reinforced(thetas=(0.2, 0.3, 0.4), n_boot=1000):
    """Phase 2 (v4): slope per theta + bootstrap CI over seeds. Uses all seeds in partial."""
    import random as _rnd
    recs = [json.loads(l) for l in open(PARTIAL)]
    cfgs = sorted({(r["f"], r["tag_decay"], r["p_respond"]) for r in recs if r["p_respond"] == 0.8})
    # organize: vals[cfg][N] = list of per-seed informed
    V = {}
    for r in recs:
        c = (r["f"], r["tag_decay"], r["p_respond"])
        if r["p_respond"] != 0.8: continue
        V.setdefault(c, {}).setdefault(r["N"], []).append(r["informed"])
    def crossing_theta(vals, theta):
        ns = sorted(vals)
        for i in range(len(ns) - 1):
            v1, v2 = vals[ns[i]], vals[ns[i + 1]]
            if v1 >= theta and v2 < theta:
                ln = np.log(ns[i]) + (v1 - theta) / (v1 - v2) * (np.log(ns[i + 1]) - np.log(ns[i]))
                return float(np.exp(ln))
        return None
    def slope_from(means_by_cfg, theta):
        pts = []
        for c in cfgs:
            vals = {n: float(np.mean(v)) for n, v in means_by_cfg[c].items()}
            x = crossing_theta(vals, theta)
            if x: pts.append((c[0] / c[1], x))
        if len(pts) < 3: return None
        lx = np.log([a for a, _ in pts]); ly = np.log([b for _, b in pts])
        return float(np.polyfit(lx, ly, 1)[0])
    out = {"n_seeds": {str(c): min(len(v) for v in V[c].values()) for c in cfgs}}
    rng = _rnd.Random(7)
    for th in thetas:
        point = slope_from(V, th)
        boots = []
        for _ in range(n_boot):
            Vb = {c: {n: [rng.choice(v) for _ in v] for n, v in V[c].items()} for c in cfgs}
            s = slope_from(Vb, th)
            if s is not None: boots.append(s)
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
        out[f"theta_{th}"] = {"slope": round(point, 4) if point else None,
                               "boot_CI95": [round(lo, 4), round(hi, 4)], "n_boot_ok": len(boots)}
    p = os.path.join(HERE, "..", "data", "r3_prime_reinforced.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    if "--fit" in sys.argv:
        fit()
    elif "--reinforced" in sys.argv:
        fit_reinforced()
    else:
        n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 2
        for ci in [int(x) for x in sys.argv[1].split(",")]:
            run_config(ci, seeds=n_seeds)
