"""R1: D2 mid-range rerun (remove ceiling artifact).
Preregistered in 实验计划.md §2/R1 BEFORE running.
Prediction R1-P: lift_D2 flat at mid-range: max|lift(Ni)-lift(Nj)| < 0.05.
"""
import numpy as np, json, os

def run_d2(N, rounds=50, generations=30, k=4, p_punish=0.2, punish_effect=0.01,
           reversion=0.01, seed=0):
    rng = np.random.default_rng(seed)
    coop_prob = rng.uniform(0.4, 0.9, N).copy()
    total_give = total_dec = total_punish = punish_ops = 0
    for _ in range(generations):
        for _ in range(rounds):
            moves = rng.random(N) < coop_prob
            total_dec += N; total_give += int(moves.sum())
            keepers = np.where(~moves)[0]
            if len(keepers):
                n_pun = rng.binomial(k, p_punish, size=len(keepers))
                coop_prob[keepers] = np.minimum(coop_prob[keepers] + n_pun * punish_effect, 0.99)
                punish_ops += k * len(keepers); total_punish += int(n_pun.sum())
            coop_prob = np.clip(coop_prob * (1 - reversion) + reversion * rng.uniform(0.4, 0.9, N), 0.3, 0.99)
    return total_give / total_dec, (total_punish / punish_ops if punish_ops else 0.0)

def run_base(N, rounds=50, generations=30, seed=0):
    rng = np.random.default_rng(seed)
    coop_prob = rng.uniform(0.4, 0.9, N)
    tg = td = 0
    for _ in range(generations * rounds):
        mv = rng.random(N) < coop_prob; td += N; tg += int(mv.sum())
    return tg / td

if __name__ == "__main__":
    out = {"experiment": "R1_d2_midrange", "target_band": [0.70, 0.88]}
    # calibration at N=100: pick first param set landing D2 give in band
    candidates = [dict(p_punish=0.2, punish_effect=0.010, reversion=0.01),
                  dict(p_punish=0.2, punish_effect=0.015, reversion=0.01),
                  dict(p_punish=0.2, punish_effect=0.005, reversion=0.01),
                  dict(p_punish=0.5, punish_effect=0.005, reversion=0.01)]
    chosen, calib = None, []
    for c in candidates:
        g = np.mean([run_d2(100, seed=s, **c)[0] for s in range(3)])
        calib.append({**c, "give_N100": round(float(g), 4)})
        if chosen is None and 0.70 <= g <= 0.88:
            chosen = c
    out["calibration"] = calib
    out["chosen"] = chosen
    if chosen is None:
        chosen = candidates[0]; out["chosen_fallback"] = True
    Ns = [10, 30, 100, 300, 1000, 3000]
    rows = []
    for N in Ns:
        seeds = 5
        d2 = [run_d2(N, seed=s, **chosen) for s in range(seeds)]
        base = np.mean([run_base(N, seed=s) for s in range(seeds)])
        give = float(np.mean([x[0] for x in d2]))
        prate = float(np.mean([x[1] for x in d2]))
        rows.append({"N": N, "base": round(float(base), 4), "d2": round(give, 4),
                     "lift": round(give - float(base), 4), "punish_rate": round(prate, 4)})
        print(rows[-1], flush=True)
    lifts = [r["lift"] for r in rows]
    spread = max(lifts) - min(lifts)
    out["rows"] = rows
    out["lift_spread"] = round(spread, 4)
    out["R1_P_pass"] = bool(spread < 0.05)
    p = os.path.join(os.path.dirname(__file__), "..", "data", "r1_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print("R1 done. spread=", spread, "pass=", out["R1_P_pass"])
