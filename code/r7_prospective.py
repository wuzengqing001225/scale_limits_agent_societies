"""R7: TRUE prospective test on unmodified third-party code (axelrod 4.13.1).
Blind predictions were logged in 外部检验B1b_R7.md BEFORE this script ran.
Frozen spec: pools I (WSLS/ALLD) and II (Grudger/ALLD), 50/50;
arm G: t(N)=max(1,round(60/(N-1))); arm S: t=6; N in {6,12,24,48,96};
metric: per-turn payoff difference Delta(N) = coop-type mean - ALLD mean;
secondary: pool cooperation rate. Deterministic strategies, 1 repetition.
"""
import axelrod as axl
import numpy as np, json, os, itertools

NS = [6, 12, 24, 48, 96]; B = 60

def turns_G(N): return max(1, round(B / (N - 1)))

def run_pool(coop_cls, N, turns):
    n = N // 2
    players = [coop_cls() for _ in range(n)] + [axl.Defector() for _ in range(n)]
    total = np.zeros(N); tt = np.zeros(N); coop_moves = 0; all_moves = 0
    for i, j in itertools.combinations(range(N), 2):
        p1, p2 = players[i].clone(), players[j].clone()
        m = axl.Match((p1, p2), turns=turns)
        m.play()
        s1, s2 = m.final_score()
        total[i] += s1; total[j] += s2
        tt[i] += turns; tt[j] += turns
        for a, b in m.result:
            coop_moves += (a == axl.Action.C) + (b == axl.Action.C); all_moves += 2
    per_turn = total / tt
    return (float(per_turn[:n].mean() - per_turn[n:].mean()),
            float(coop_moves / all_moves))

if __name__ == "__main__":
    out = {"experiment": "R7_prospective", "library": f"axelrod {axl.__version__}",
           "B": B, "rows": []}
    for pool_name, cls in [("I_WSLS", axl.WinStayLoseShift), ("II_Grudger", axl.Grudger)]:
        for arm, tf in [("G", turns_G), ("S", lambda N: 6)]:
            for N in NS:
                t = tf(N)
                d, cr = run_pool(cls, N, t)
                out["rows"].append({"pool": pool_name, "arm": arm, "N": N, "turns": t,
                                     "delta": round(d, 4), "coop_rate": round(cr, 4)})
                print(out["rows"][-1], flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "r7_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print("saved", p)
