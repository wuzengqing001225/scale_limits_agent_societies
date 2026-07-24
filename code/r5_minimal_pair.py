"""R5 (+R2 components): minimal pair — arms differ ONLY in information routing.
Preregistered in 实验计划.md §2/R5, §2/R2 BEFORE running.
Shared by both arms: ring topology (donor faces random one of k=4 ring neighbors),
donation payoffs (give: -1 donor / +3 recipient), response rule (defect w.p.
p_respond toward tagged partner), evolutionary learning (payoff-roulette resampling
of coop_prob + Gaussian mutation, identity=position so tags persist), tag decay 0.1
(dense matrix, no M-cap — declared deviation), 4 tags emitted per keep in both arms.
Difference:
  local  arm: tags go to keeper's 4 ring neighbors (= keeper's future partners)
  global arm: victim sends tags to 4 uniform random agents (excluding keeper)
Variant (R2-P3): global arm with response = refusal (no transfer, both get 0).
Predictions: R5-P1 global |lift|(1000) <= 0.2*|lift|(10); R5-P2 local max|Δlift|<0.05;
R5-P3 coverage: global decays >=10x, local varies <2x; R2-P1/P2 decay of total and
compliance components in global arm; sign NOT preregistered.
"""
import numpy as np, json, os

def run_arm(N, arm, rounds=50, generations=30, k=4, p_respond=0.8, tag_decay=0.1,
            emit=4, refusal=False, seed=0, b=3.0, c=1.0, mut=0.05,
            floor=0.05, cap=0.95, measure_from_gen=15):
    rng = np.random.default_rng(seed)
    coop = rng.uniform(0.2, 0.8, N)
    tags = np.zeros((N, N), dtype=bool)
    offsets = np.array([o for o in range(-(k // 2), k // 2 + 1) if o != 0])
    idx = np.arange(N)
    st = dict(played=0, give=0, inf_dec=0, inf_give=0, uninf_dec=0, uninf_give=0)
    for gen in range(generations):
        payoff = np.zeros(N)
        for _ in range(rounds):
            partner = (idx + rng.choice(offsets, N)) % N
            has = tags[idx, partner]
            r1, r2 = rng.random(N), rng.random(N)
            if refusal:
                refuse = has & (r1 < p_respond)
                played = ~refuse
                give = played & (r2 < coop)
            else:
                played = np.ones(N, dtype=bool)
                # Amendment 2026-07-12(a): multiplicative response (sanction stays a
                # sanction regardless of evolved baseline): tagged -> give w.p. coop*(1-p_respond)
                give = r1 < np.where(has, coop * (1.0 - p_respond), coop)
            payoff -= c * give
            np.add.at(payoff, partner[give], b)
            if gen >= measure_from_gen:
                st["played"] += int(played.sum()); st["give"] += int(give.sum())
                st["inf_dec"] += int((has & played).sum()); st["inf_give"] += int((has & give).sum())
                st["uninf_dec"] += int((~has & played).sum()); st["uninf_give"] += int((~has & give).sum())
            keep_mask = played & ~give
            keepers = idx[keep_mask]
            if arm != "none" and len(keepers):
                if arm == "local":
                    for off in offsets:
                        tags[(keepers + off) % N, keepers] = True
                elif arm == "global":
                    recips = (keepers[:, None] + 1 + rng.integers(0, N - 1, size=(len(keepers), emit))) % N
                    tags[recips.ravel(), np.repeat(keepers, emit)] = True
            if arm != "none":
                ti, tj = np.nonzero(tags)
                if len(ti):
                    drop = rng.random(len(ti)) < tag_decay
                    tags[ti[drop], tj[drop]] = False
        fit = payoff - payoff.min() + 1.0
        parents = rng.choice(N, N, p=fit / fit.sum())
        coop = np.clip(coop[parents] + rng.normal(0, mut, N), floor, cap)
    res = {"give": st["give"] / st["played"] if st["played"] else None,
           "coverage": st["inf_dec"] / (st["inf_dec"] + st["uninf_dec"]) if (st["inf_dec"] + st["uninf_dec"]) else 0.0,
           "give_informed": st["inf_give"] / st["inf_dec"] if st["inf_dec"] else None,
           "give_uninformed": st["uninf_give"] / st["uninf_dec"] if st["uninf_dec"] else None}
    return res

PARTIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "r5_partial.jsonl")

def run_chunk(Ns, seeds=3, arms=None):
    combos = [("none", False), ("local", False), ("global", False), ("global", True)]
    if arms:
        combos = [c for c in combos if c[0] in arms]
    done = set()
    if os.path.exists(PARTIAL):
        for line in open(PARTIAL):
            d = json.loads(line); done.add((d["N"], d["arm"], d["refusal"], d["seed"]))
    with open(PARTIAL, "a") as f:
        for N in Ns:
            for arm, refusal in combos:
                for s in range(seeds):
                    if (N, arm, refusal, s) in done: continue
                    r = run_arm(N, arm, refusal=refusal, seed=s)
                    f.write(json.dumps({"N": N, "arm": arm, "refusal": refusal, "seed": s, **r}) + "\n")
                    f.flush(); print(N, arm, refusal, s, round(r["give"], 3), flush=True)

def finalize(Ns=(10, 100, 1000)):
    recs = [json.loads(l) for l in open(PARTIAL)]
    agg, rows = {}, []
    for N in Ns:
        for arm, refusal in [("none", False), ("local", False), ("global", False), ("global", True)]:
            rs = [r for r in recs if r["N"] == N and r["arm"] == arm and r["refusal"] == refusal]
            mean = {k2: (float(np.mean([r[k2] for r in rs if r.get(k2) is not None]))
                          if any(r.get(k2) is not None for r in rs) else None)
                    for k2 in ("give", "coverage", "give_informed", "give_uninformed")}
            agg[(N, arm, refusal)] = mean
            rows.append({"N": N, "arm": arm, "refusal": refusal, "n_seeds": len(rs),
                         **{k2: (round(v, 4) if v is not None else None) for k2, v in mean.items()}})
    def lift(N, arm, refusal=False):
        return agg[(N, arm, refusal)]["give"] - agg[(N, "none", False)]["give"]
    out = {"experiment": "R5_minimal_pair_plus_R2",
           "amendment": "2026-07-12(a): multiplicative sanction response; interpretability floor |lift(10)|>=0.05 for ratio verdicts",
           "rows": rows}
    Ns = list(Ns)
    lg = {N: lift(N, "global") for N in Ns}
    ll = {N: lift(N, "local") for N in Ns}
    lgr = {N: lift(N, "global", True) for N in Ns}
    comp = {N: agg[(N, "global", False)]["give_uninformed"] - agg[(N, "none", False)]["give"] for N in Ns}
    cov_g = {N: agg[(N, "global", False)]["coverage"] for N in Ns}
    cov_l = {N: agg[(N, "local", False)]["coverage"] for N in Ns}
    out["lift_global"] = {str(N): round(v, 4) for N, v in lg.items()}
    out["lift_local"] = {str(N): round(v, 4) for N, v in ll.items()}
    out["lift_global_refusal"] = {str(N): round(v, 4) for N, v in lgr.items()}
    out["compliance_component_global"] = {str(N): round(v, 4) for N, v in comp.items()}
    out["coverage_global"] = {str(N): round(v, 4) for N, v in cov_g.items()}
    out["coverage_local"] = {str(N): round(v, 4) for N, v in cov_l.items()}
    FLOOR = 0.05
    def ratio_verdict(l10, l1000):
        if abs(l10) < FLOOR: return "INCONCLUSIVE(floor)"
        return bool(abs(l1000) <= 0.2 * abs(l10))
    out["R5_P1_pass"] = ratio_verdict(lg[10], lg[1000])
    out["R5_P2_pass"] = bool(max(abs(ll[a] - ll[b]) for a in Ns for b in Ns) < 0.05)
    out["R5_P3_pass"] = bool(cov_g[10] / max(cov_g[1000], 1e-9) >= 10 and
                              max(cov_l.values()) / max(min(cov_l.values()), 1e-9) < 2)
    out["R2_P1_pass"] = out["R5_P1_pass"]
    out["R2_P2_pass"] = ratio_verdict(comp[10], comp[1000])
    out["R2_P3_pass"] = ratio_verdict(lgr[10], lgr[1000])
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "r5_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print("R5 done.", {k: out[k] for k in out if k.endswith("_pass")})

if __name__ == "__main__":
    import sys
    if "--finalize" in sys.argv:
        finalize()
    elif "--retest" in sys.argv:
        # R5-P2 retest amendment (2026-07-14): grid from N=30 per C3 causal
        # neighbourhood condition; local arm + baseline; 6 seeds.
        run_chunk([30, 100, 300, 1000], seeds=6, arms=("none", "local"))
        recs = [json.loads(l) for l in open(PARTIAL)]
        Ns = [30, 100, 300, 1000]
        lifts = {}
        for N in Ns:
            base = {r["seed"]: r["give"] for r in recs if r["N"] == N and r["arm"] == "none"}
            ls = [r["give"] - base[r["seed"]] for r in recs
                  if r["N"] == N and r["arm"] == "local" and not r["refusal"] and r["seed"] in base]
            lifts[N] = float(np.mean(ls))
        spread = max(lifts.values()) - min(lifts.values())
        out = {"experiment": "R5_P2_retest", "grid": Ns, "seeds": 6,
               "lift_local": {str(k): round(v, 4) for k, v in lifts.items()},
               "spread": round(spread, 4), "R5_P2_retest_pass": bool(spread < 0.05)}
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "r5_p2_retest.json")
        json.dump(out, open(p, "w"), indent=1)
        print(json.dumps(out, indent=1))
    else:
        ns = [int(x) for x in sys.argv[sys.argv.index("--ns") + 1].split(",")] if "--ns" in sys.argv else [10, 100, 1000]
        run_chunk(ns)
