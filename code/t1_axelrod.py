"""T1: third-party implementation transfer test on axelrod-python 4.13.1.
Registered in 实验计划.md §7/T1 Amendment BEFORE running. We only choose the
interaction schedule; strategy code is the community library's, untouched.
Arm G: per-capita budget B=50 -> turns t(N)=max(1, round(50/(N-1)))  (degrading shadow of future)
Arm S: t=10 fixed (survival twin)
Predictions: T1-P1 sign flip of Delta=TFT-Defector mean payoff/turn between N=20 and N=40 (arm G);
T1-P2 no flip in arm S; T1-P3 (exploratory) mixed-pool cooperation rate falls with N in G, ~flat in S.
"""
import axelrod as axl
import numpy as np, json, os, itertools

NS = [6, 10, 20, 40, 80]
B = 50

def turns_G(N): return max(1, round(B / (N - 1)))

def round_robin(players, turns, reps=1, seed=0):
    n = len(players)
    total = np.zeros(n); tturns = np.zeros(n)
    for rep in range(reps):
        for i, j in itertools.combinations(range(n), 2):
            p1, p2 = players[i].clone(), players[j].clone()
            m = axl.Match((p1, p2), turns=turns, seed=seed * 7919 + rep)
            m.play()
            s1, s2 = m.final_score()
            total[i] += s1; total[j] += s2
            tturns[i] += turns; tturns[j] += turns
    return total / tturns  # mean payoff per turn per player

def two_type_delta(N, turns):
    players = [axl.TitForTat() for _ in range(N // 2)] + [axl.Defector() for _ in range(N // 2)]
    pp = round_robin(players, turns)
    return float(pp[:N // 2].mean() - pp[N // 2:].mean())

def pool_coop_rate(N, turns, reps=3):
    kinds = [axl.TitForTat, axl.Defector, axl.Cooperator, axl.Grudger,
             axl.WinStayLoseShift, axl.Random]
    rates = []
    for rep in range(reps):
        players = [kinds[k % 6]() for k in range(N)]
        n = len(players); coop = 0; tot = 0
        for i, j in itertools.combinations(range(n), 2):
            p1, p2 = players[i].clone(), players[j].clone()
            m = axl.Match((p1, p2), turns=turns, seed=rep * 104729 + i * 31 + j)
            m.play()
            for a, b in m.result:
                coop += (a == axl.Action.C) + (b == axl.Action.C); tot += 2
        rates.append(coop / tot)
    return float(np.mean(rates))

if __name__ == "__main__":
    out = {"experiment": "T1_axelrod_transfer", "library": f"axelrod {axl.__version__}",
           "B": B, "rows": []}
    for N in NS:
        tg = turns_G(N)
        dG = two_type_delta(N, tg)
        dS = two_type_delta(N, 10)
        cG = pool_coop_rate(N, tg)
        cS = pool_coop_rate(N, 10)
        pred_sign = 1 if tg * (N - 2) - 2.5 * N > 0 else -1
        out["rows"].append({"N": N, "turns_G": tg, "delta_G": round(dG, 4),
                             "delta_S": round(dS, 4), "pred_sign_G": pred_sign,
                             "coop_pool_G": round(cG, 4), "coop_pool_S": round(cS, 4)})
        print(out["rows"][-1], flush=True)
    signs = [np.sign(r["delta_G"]) for r in out["rows"]]
    pred = [r["pred_sign_G"] for r in out["rows"]]
    out["T1_P1_pass"] = bool(signs[0] > 0 and signs[1] > 0 and signs[2] > 0 and
                              signs[3] < 0 and signs[4] < 0)
    out["T1_P1_matches_analytic"] = bool(all(int(s) == p for s, p in zip(signs, pred)))
    out["T1_P2_pass"] = bool(all(r["delta_S"] > 0 for r in out["rows"]))
    cg = [r["coop_pool_G"] for r in out["rows"]]; cs = [r["coop_pool_S"] for r in out["rows"]]
    out["T1_P3_direction"] = {"G_falls": bool(cg[-1] < cg[0] - 0.05),
                               "S_flat": bool(abs(cs[-1] - cs[0]) < 0.05)}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "t1_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print({k: out[k] for k in out if k.startswith("T1_")})
