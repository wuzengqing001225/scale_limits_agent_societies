"""R6: institutional-coverage demonstration (organic gossip vs public board).
Registered in 实验计划.md §7/R6 BEFORE running.
Fixed coop_prob (no evolution), multiplicative sanction (informed -> give w.p. coop*0.2),
tag decay 0.1, 4 tags per keep. Arms differ ONLY in coverage generator:
  organic: victim sends tags to 4 random others  (coverage ~ 1/N)
  board:   every keep visible to ALL             (coverage ≡ 1)
Predictions: R6-P1 organic |lift(1000)| <= 0.2*|lift(10)|; R6-P2 board max|Δlift| < 0.05.
"""
import numpy as np, json, os

def run(N, arm, rounds=1500, p_respond=0.8, tag_decay=0.1, emit=4, seed=0):
    rng = np.random.default_rng(seed)
    coop = rng.uniform(0.4, 0.9, N)
    tags = np.zeros((N, N), dtype=bool)   # tags[holder, about]
    m = N - (N % 2)
    give_ct = dec_ct = inf_ct = 0
    for _ in range(rounds):
        perm = rng.permutation(N)[:m]
        a, b_ = perm[::2], perm[1::2]
        for me, other in ((a, b_), (b_, a)):
            has = tags[me, other] if arm != "none" else np.zeros(len(me), bool)
            p = np.where(has, coop[me] * (1 - p_respond), coop[me])
            gv = rng.random(len(me)) < p
            give_ct += int(gv.sum()); dec_ct += len(me); inf_ct += int(has.sum())
            keepers = me[~gv]
            if arm == "organic" and len(keepers):
                recips = (keepers[:, None] + 1 + rng.integers(0, N - 1, size=(len(keepers), emit))) % N
                tags[recips.ravel(), np.repeat(keepers, emit)] = True
            elif arm == "board" and len(keepers):
                tags[:, keepers] = True
        if arm != "none":
            ti, tj = np.nonzero(tags)
            if len(ti):
                drop = rng.random(len(ti)) < tag_decay
                tags[ti[drop], tj[drop]] = False
    return give_ct / dec_ct, inf_ct / dec_ct

PARTIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "r6_partial.jsonl")

if __name__ == "__main__":
    Ns = [10, 100, 1000]; seeds = 5
    done = {}
    if os.path.exists(PARTIAL):
        for line in open(PARTIAL):
            d = json.loads(line); done[(d["N"], d["arm"], d["seed"])] = d
    with open(PARTIAL, "a") as f:
        for N in Ns:
            for arm in ("none", "organic", "board"):
                for s in range(seeds):
                    if (N, arm, s) in done: continue
                    g, cov = run(N, arm, seed=s)
                    d = {"N": N, "arm": arm, "seed": s, "give": round(g, 5), "cov": round(cov, 5)}
                    done[(N, arm, s)] = d
                    f.write(json.dumps(d) + "\n"); f.flush()
                    print(d, flush=True)
    # finalize
    def mean(N, arm, key="give"):
        vs = [d[key] for (n, a, s), d in done.items() if n == N and a == arm]
        return float(np.mean(vs)) if vs else None
    out = {"experiment": "R6_board_vs_organic", "rows": []}
    L = {"organic": {}, "board": {}}
    complete = True
    for N in Ns:
        base = mean(N, "none")
        for arm in ("organic", "board"):
            g, cov = mean(N, arm), mean(N, arm, "cov")
            if g is None or base is None: complete = False; continue
            L[arm][N] = g - base
            out["rows"].append({"N": N, "arm": arm, "base": round(base, 4), "give": round(g, 4),
                                 "lift": round(g - base, 4), "coverage": round(cov, 4)})
    if complete:
        out["R6_P1_pass"] = bool(abs(L["organic"][1000]) <= 0.2 * abs(L["organic"][10]))
        out["R6_P2_pass"] = bool(max(abs(L["board"][a] - L["board"][b]) for a in Ns for b in Ns) < 0.05)
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "r6_results.json")
        json.dump(out, open(p, "w"), indent=1)
        print("R6:", {k: out[k] for k in out if k.endswith("_pass")})
    else:
        print("R6: partial, rerun to continue")
