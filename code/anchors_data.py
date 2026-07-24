"""Regenerate anchor-pair curves (Pair A/B/C) with the ORIGINAL scripts'
functions and parameters, saving curves to JSON for the paper figure.
Resume-safe across chunks. Output: data/anchors_data.json
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coop_abm import run_wellmixed, run_network
from consensus_abm import run_consensus, run_schelling
from agg_abm import simulate, CKPTS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "anchors_data.json")

def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}

def save(d):
    json.dump(d, open(OUT, "w"), indent=1)

if __name__ == "__main__":
    d = load()
    # ---- Pair A ----
    if "A_direct" not in d:
        d["A_direct"] = {}
    for N in [10, 20, 40, 80, 160, 320, 1000]:
        k = str(N)
        if k in d["A_direct"]: continue
        seeds = 6 if N <= 320 else 3
        d["A_direct"][k] = round(float(np.mean(
            [run_wellmixed(N, seed=s)[0] for s in range(seeds)])), 4)
        save(d); print("A_direct", N, d["A_direct"][k], flush=True)
    if "A_network" not in d:
        d["A_network"] = {}
    for N in [10, 100, 1000, 10000, 100000]:
        k = str(N)
        if k in d["A_network"]: continue
        seeds = 3
        d["A_network"][k] = round(float(np.mean(
            [run_network(N, seed=s) for s in range(seeds)])), 4)
        save(d); print("A_network", N, d["A_network"][k], flush=True)
    # ---- Pair B ----
    for key, mode in [("B_local", "local"), ("B_global", "global")]:
        if key not in d: d[key] = {}
        for N in [10, 30, 100, 1000, 10000, 100000]:
            k = str(N)
            if k in d[key]: continue
            seeds = 8 if N <= 1000 else 3
            d[key][k] = round(float(np.mean(
                [run_consensus(N, mode, seed=s)[0] for s in range(seeds)])), 3)
            save(d); print(key, N, d[key][k], flush=True)
    # ---- Interventions (archive the historically reported rescues) ----
    if "A_intervention_fixed_partner" not in d:
        d["A_intervention_fixed_partner"] = {}
    for N in [40, 320, 1000]:
        k = str(N)
        if k in d["A_intervention_fixed_partner"]: continue
        d["A_intervention_fixed_partner"][k] = round(float(np.mean(
            [run_wellmixed(N, repeat_partner=True, seed=s)[0] for s in range(3)])), 4)
        save(d); print("A_interv", N, d["A_intervention_fixed_partner"][k], flush=True)
    if "B_intervention_eps" not in d:
        d["B_intervention_eps"] = {}
    for eps in [0.05, 0.2]:
        k = str(eps)
        if k in d["B_intervention_eps"]: continue
        d["B_intervention_eps"][k] = round(float(np.mean(
            [run_consensus(10000, "local", eps=eps, seed=s)[0] for s in range(5)])), 3)
        save(d); print("B_interv eps", eps, d["B_intervention_eps"][k], flush=True)
    if "C_intervention_lam" not in d:
        res = {}
        for lam in [0.0, 0.6, 0.9, 1.0]:
            a, _ = simulate(10001, lam=lam, runs=2000, seed=2)
            res[str(lam)] = round(float(a[10001]), 3)
        d["C_intervention_lam"] = res
        save(d); print("C_interv", res, flush=True)
    # ---- Pair C ----
    if "C" not in d:
        ai, _ = simulate(CKPTS[-1], lam=1.0, runs=2000)
        ah, _ = simulate(CKPTS[-1], lam=0.0, runs=2000, seed=1)
        d["C"] = {"indep": {str(n): round(float(ai[n]), 4) for n in CKPTS},
                  "herd": {str(n): round(float(ah[n]), 4) for n in CKPTS}}
        save(d); print("C done", flush=True)
    print("anchors_data complete:", {k: len(v) if isinstance(v, dict) else 1 for k, v in d.items()})
