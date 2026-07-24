"""R3: quantitative N_c prediction (new held-out carrier).
Preregistered in 实验计划.md §2/R3 BEFORE running.
Mechanics identical to pair_d_rule_abm.run_gossip (M-cap, dedupe, per-keep loop);
only change: burn-in discarded from measurement (mechanics untouched).
Derivation (registered pre-data): tag circulation = branching process,
R0 ~ f*p_respond*tau/(N-1), plus seeding term ~ f*keep_rate*tau/N  =>  N_c ∝ f*tau.
Operationalization: plateau = informed(N=10); N_half = N where informed = 0.5*plateau
(log-N interpolation). Primary fit on p=0.8 configs: log(N_half) ~ log(f*tau),
slope in [0.75, 1.25], R^2 > 0.90.
Disclosure: default config (f=3, tau=10, p=0.8) seen in old runs (calibration only);
other configs are out-of-sample.
"""
import numpy as np, json, os

def run_gossip_window(N, f_gossip, p_respond, tag_decay, M=20,
                      rounds_total=150, burn=50, seed=0):
    rng = np.random.default_rng(seed)
    coop_prob = rng.uniform(0.4, 0.9, N)
    tag_ids = np.full((N, M), -1, dtype=np.int64)
    tag_ptr = np.zeros(N, dtype=np.int64)
    m = N - (N % 2)
    total_dec = informed_dec = 0
    for rnd in range(rounds_total):
        perm = rng.permutation(N)[:m]
        a, b_ = perm[::2], perm[1::2]
        n_pairs = len(a)
        def has_tag(x, y):
            return (tag_ids[x] == y[:, None]).any(axis=1)
        a_tag, b_tag = has_tag(a, b_), has_tag(b_, a)
        if rnd >= burn:
            informed_dec += int(a_tag.sum()) + int(b_tag.sum())
            total_dec += 2 * n_pairs
        r_a, r_b = rng.random(n_pairs), rng.random(n_pairs)
        mv_a = np.where(a_tag, (r_a > p_respond).astype(int), (r_a < coop_prob[a]).astype(int))
        mv_b = np.where(b_tag, (r_b > p_respond).astype(int), (r_b < coop_prob[b_]).astype(int))
        for side, me_arr, other_arr, moves in [(0, a, b_, mv_a), (1, b_, a, mv_b)]:
            for idx in np.where(moves == 0)[0]:
                keeper, victim = me_arr[idx], other_arr[idx]
                cands = np.arange(N)
                cands = cands[(cands != victim) & (cands != keeper)]
                recip = rng.choice(cands, size=min(f_gossip, len(cands)), replace=False)
                for r in recip:
                    if not np.any(tag_ids[r] == keeper):
                        s = tag_ptr[r]; tag_ids[r, s] = keeper; tag_ptr[r] = (s + 1) % M
        tag_ids[rng.random((N, M)) < tag_decay] = -1
    return informed_dec / total_dec if total_dec else 0.0

def measure_config(f, tag_decay, p_respond, seeds=2, cap_N=5000):
    tau = 1.0 / tag_decay
    N_pred = max(10, f * p_respond * tau)
    grid = sorted(set([10] + [max(10, int(round(N_pred * mfac))) for mfac in (0.25, 0.5, 1, 2, 4, 8)]))
    vals = {}
    def run_grid(ns):
        for N in ns:
            if N in vals: continue
            vals[N] = float(np.mean([run_gossip_window(N, f, p_respond, tag_decay, seed=s)
                                     for s in range(seeds)]))
    run_grid(grid)
    plateau = vals[grid[0]]
    target = 0.5 * plateau
    # adaptive extension if no crossing yet
    ext = 0
    while min(vals[n] for n in grid) > target and ext < 3 and grid[-1] * 2 <= cap_N:
        grid.append(grid[-1] * 2); run_grid([grid[-1]]); ext += 1
    grid = sorted(vals)
    N_half = None
    for i in range(len(grid) - 1):
        v1, v2 = vals[grid[i]], vals[grid[i + 1]]
        if v1 >= target and v2 < target:
            ln = np.log(grid[i]) + (v1 - target) / (v1 - v2) * (np.log(grid[i + 1]) - np.log(grid[i]))
            N_half = float(np.exp(ln)); break
    return {"f": f, "tag_decay": tag_decay, "tau": tau, "p_respond": p_respond,
            "N_pred_f_p_tau": round(f * p_respond * tau, 1), "f_tau": round(f * tau, 1),
            "plateau": round(plateau, 4), "target": round(target, 4),
            "curve": {str(n): round(vals[n], 4) for n in grid},
            "N_half": round(N_half, 2) if N_half else None}

if __name__ == "__main__":
    configs = ([dict(f=f, tag_decay=0.1, p_respond=0.8) for f in (1, 2, 3, 6, 10)] +
               [dict(f=3, tag_decay=0.05, p_respond=0.8), dict(f=3, tag_decay=0.2, p_respond=0.8),
                dict(f=3, tag_decay=0.1, p_respond=0.4)])
    rows = []
    for c in configs:
        r = measure_config(**c); rows.append(r); print(r, flush=True)
    # primary fit: p=0.8 configs only, log N_half ~ log(f*tau)
    prim = [r for r in rows if r["p_respond"] == 0.8 and r["N_half"]]
    x = np.log([r["f_tau"] for r in prim]); y = np.log([r["N_half"] for r in prim])
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {"experiment": "R3_nc_scaling", "rows": rows,
           "primary_fit": {"slope": round(float(b), 3), "intercept": round(float(a), 3),
                            "R2": round(r2, 4), "n_configs": len(prim)},
           "R3_P_pass": bool(0.75 <= b <= 1.25 and r2 > 0.90),
           "product_form_check": "configs (6,10) and (3,20) share f*tau=60: compare N_half",
           "secondary_p_effect": [r for r in rows if r["p_respond"] == 0.4]}
    p = os.path.join(os.path.dirname(__file__), "..", "data", "r3_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print("R3 done. slope=", round(float(b), 3), "R2=", round(r2, 4), "pass=", out["R3_P_pass"])
