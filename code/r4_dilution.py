"""R4: causal confirmation of reputation-dilution -> N dependence.
Preregistered in 实验计划.md §2/R4 BEFORE running. Controlled REBUILD (original
Arch2 code absent from repo); the two variants differ ONLY in whether the
reputation update step is diluted by N:
  diluted:    rep[keeper] -= delta * k / N   (global board: k reports averaged into N-agent consensus)
  normalized: rep[keeper] -= delta           (per-incident step, N-free)
Predictions: R4-P1 diluted |lift(10)-lift(1000)| >= 0.10; R4-P2 normalized max|Δlift| < 0.05.
"""
import numpy as np, json, os

def run_rep(N, variant, rounds=50, generations=30, k=4, delta=0.15, rho=0.02,
            p_respond=0.8, theta=0.5, seed=0):
    rng = np.random.default_rng(seed)
    coop_prob = rng.uniform(0.4, 0.9, N)
    rep = np.ones(N)
    offsets = np.array([o for o in range(-(k // 2), k // 2 + 1) if o != 0])
    idx = np.arange(N)
    tg = td = 0
    step = delta * k / N if variant == "diluted" else delta
    for _ in range(generations):
        for _ in range(rounds):
            partner = (idx + rng.choice(offsets, N)) % N
            bad = rep[partner] < theta
            r = rng.random(N)
            moves = np.where(bad, r > p_respond, r < coop_prob)
            td += N; tg += int(moves.sum())
            keepers = idx[~moves]
            rep[keepers] = np.clip(rep[keepers] - step, 0.0, 1.0)
            rep = rep + rho * (1.0 - rep)
    return tg / td

def run_base(N, rounds=50, generations=30, seed=0):
    rng = np.random.default_rng(seed)
    coop_prob = rng.uniform(0.4, 0.9, N)
    tg = td = 0
    for _ in range(generations * rounds):
        mv = rng.random(N) < coop_prob; td += N; tg += int(mv.sum())
    return tg / td

if __name__ == "__main__":
    Ns = [10, 100, 1000]; seeds = 5
    out = {"experiment": "R4_dilution", "rows": []}
    lifts = {"diluted": {}, "normalized": {}}
    for N in Ns:
        base = float(np.mean([run_base(N, seed=s) for s in range(seeds)]))
        for var in ("diluted", "normalized"):
            g = float(np.mean([run_rep(N, var, seed=s) for s in range(seeds)]))
            lifts[var][N] = g - base
            out["rows"].append({"N": N, "variant": var, "base": round(base, 4),
                                 "give": round(g, 4), "lift": round(g - base, 4)})
            print(out["rows"][-1], flush=True)
    dil = abs(lifts["diluted"][10] - lifts["diluted"][1000])
    nrm = max(abs(lifts["normalized"][a] - lifts["normalized"][b])
              for a in Ns for b in Ns)
    out["diluted_delta_10_1000"] = round(dil, 4)
    out["normalized_max_delta"] = round(nrm, 4)
    out["R4_P1_pass"] = bool(dil >= 0.10)
    out["R4_P2_pass"] = bool(nrm < 0.05)
    p = os.path.join(os.path.dirname(__file__), "..", "data", "r4_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print("R4 done.", out["R4_P1_pass"], out["R4_P2_pass"])
