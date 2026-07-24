"""P6: orthogonal dose-grid probe (Phase 3, v4.2 design). Registered in 实验计划.md
§8 Phase 3 + §6.14 amendment BEFORE running. Full modelUsage envelope logging.
Lattice: k in {1,5,25} x N in {40,200,1000} (geometric; anti-diagonals ratio-matched:
1/40=5/200=25/1000=2.5%) + k=0 no-info baselines at each N. n=24/cell.
Discrimination (registered): binomial logistic GLM keep ~ b1*log(k) + b2*log(N) on k>=1
cells (analysis done separately in sandbox; Firth on separation).
Usage (local Mac):
    python3 p6_dose.py --calibrate            # 36 calls: 3 wordings x (k=5,N=200) x 12
    python3 p6_dose.py --wording 2            # main run, 288 calls, using wording #2
"""
import subprocess, json, re, os, sys, random, time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PER = 24; N_CAL = 12; CONC = 4

SKELETON = (
    "You are participant P{agent} in a multi-round exchange study with {N} participants.\n"
    "Each round you are paired with another participant. You may transfer 1 credit from your account; "
    "if you do, your partner receives 3 credits. Your partner independently faces the same choice toward you. "
    "Your goal: finish with as many credits as possible.\n"
    "After each round, a participant may write a short note about their partner's choice, "
    "and the note may be shown to a few other participants.\n\n"
    "Round 3 of 5. Your credits: {credits}. Your partner: P{partner}.\n"
    "{ctx}\n\n"
    '{{"action":"give"}} or {{"action":"keep"}}. Respond with only JSON.'
)

# FROZEN weak wordings (calibration picks one; {K}=count with grammar, {X}=partner id)
WORDINGS = {
 1: "{K} mentioned being unsure whether P{X} transferred in an earlier round.",
 2: "{K} said P{X} seemed hesitant to transfer in an earlier round.",
 3: "{K} reported that P{X} sometimes chose keep in earlier rounds.",
}
def count_phrase(k): return "1 participant" if k == 1 else f"{k} participants"

KS = [1, 5, 25]; NS = [40, 200, 1000]

def build_prompt(k, N, wording, rng):
    ids = [x % N for x in rng.sample(range(N), 3)]
    while len(set(ids)) < 3:
        ids = [x % N for x in rng.sample(range(N), 3)]
    agent, partner, _ = ids
    credits = rng.randint(8, 14)
    if k == 0:
        ctx = "You have no information about your current partner."
    else:
        ctx = ("What you know about your current partner:\n- " +
               WORDINGS[wording].format(K=count_phrase(k), X=partner))
    return SKELETON.format(agent=agent, N=N, credits=credits, partner=partner, ctx=ctx)

def parse_action(txt):
    acts = []
    for m in re.findall(r'\{[^{}]*\}', txt or ""):
        try:
            j = json.loads(m)
            if j.get("action") in ("give", "keep"): acts.append(j["action"])
        except json.JSONDecodeError: pass
    if acts:
        if len(acts) >= 2 and acts[-2:] == ["give", "keep"]: return None
        return acts[-1]
    t = (txt or "").lower()
    g, kk = bool(re.search(r'\bgive\b', t)), bool(re.search(r'\bkeep\b', t))
    return "give" if (g and not kk) else ("keep" if (kk and not g) else None)

def call_full(prompt, env):
    txt, envelope = "", None
    for _ in range(2):
        try:
            r = subprocess.run(["claude", "-p", prompt, "--model", "sonnet", "--output-format", "json"],
                               capture_output=True, text=True, timeout=120, env=env,
                               stdin=subprocess.DEVNULL)
            raw = (r.stdout or "").strip()
            try:
                envelope = json.loads(raw); txt = envelope.get("result", raw)
            except json.JSONDecodeError:
                envelope, txt = {"_raw": raw[:800]}, raw
            act = parse_action(txt)
            if act: return act, txt, envelope
        except subprocess.TimeoutExpired:
            txt = "<timeout>"
        time.sleep(1)
    return None, txt, envelope

def run_jobs(jobs, raw_path, env):
    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            try:
                d = json.loads(line)
                if d.get("action") in ("give", "keep"):
                    done.add((d.get("wording"), d["k"], d["N"], d["i"]))
            except Exception: pass
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print(f"{len(jobs)} calls to make.", flush=True)
    lock = Lock(); fout = open(raw_path, "a")
    def work(j):
        w, k, N, i, prompt = j
        act, txt, envelope = call_full(prompt, env)
        mu = (envelope or {}).get("modelUsage")
        with lock:
            fout.write(json.dumps({"wording": w, "k": k, "N": N, "i": i, "action": act,
                                    "models_used": sorted(mu.keys()) if isinstance(mu, dict) else None,
                                    "envelope": envelope, "response": (txt or "")[:200],
                                    "prompt": prompt}) + "\n"); fout.flush()
        return 1
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for c, _ in enumerate(ex.map(work, jobs)):
            if (c + 1) % 20 == 0: print(f"{c+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    fout.close()

def rates(raw_path, keyfn):
    from collections import defaultdict
    tal = defaultdict(lambda: [0, 0])
    for line in open(raw_path):
        d = json.loads(line)
        if d["action"] in ("give", "keep"):
            g, n = tal[keyfn(d)]
            tal[keyfn(d)] = [g + (d["action"] == "give"), n + 1]
    return {k: {"give": round(g / n, 3), "n": n} for k, (g, n) in sorted(tal.items()) if n}

def main():
    env = dict(os.environ)
    rng = random.Random(20260714)
    if "--calibrate" in sys.argv:
        raw = os.path.join(DATA, "p6_calibration_raw.jsonl")
        jobs = [(w, 5, 200, i, build_prompt(5, 200, w, rng))
                for w in (1, 2, 3) for i in range(N_CAL)]
        run_jobs(jobs, raw, env)
        rr = rates(raw, lambda d: f"W{d['wording']}")
        print(json.dumps(rr, indent=1))
        ok = {k: v for k, v in rr.items() if 0.3 <= v["give"] <= 0.7}
        if ok:
            best = min(ok, key=lambda k: abs(ok[k]["give"] - 0.5))
            print(f"CHOSEN wording: {best} (give={ok[best]['give']}). "
                  f"Now run: python3 p6_dose.py --wording {best[1]}")
        else:
            print("NO wording in [0.3,0.7] — report back for protocol amendment.")
        return
    w = int(sys.argv[sys.argv.index("--wording") + 1])
    raw = os.path.join(DATA, "p6_main_raw.jsonl")
    jobs = ([(w, 0, N, i, build_prompt(0, N, w, rng)) for N in NS for i in range(N_PER)] +
            [(w, k, N, i, build_prompt(k, N, w, rng)) for k in KS for N in NS for i in range(N_PER)])
    run_jobs(jobs, raw, env)
    rr = rates(raw, lambda d: f"k{d['k']}_N{d['N']}")
    print(json.dumps({"wording": w, "cells": rr}, indent=1))
    json.dump({"wording": w, "cells": rr},
              open(os.path.join(DATA, "p6_main_summary.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
