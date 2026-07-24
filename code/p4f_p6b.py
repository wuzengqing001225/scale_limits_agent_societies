"""P4-F (V4 flip factor separation) + P6b (k=0->1 jump wording retest).
Registered in 实验计划.md §10 BEFORE running; wordings below are FROZEN there.
Both classificatory/exploratory — no main-claim pass/fail. Full modelUsage logging.
Usage (local Mac, logged-in claude CLI):
    python3 p4f_p6b.py --p4f                  # 96 calls, sonnet
    python3 p4f_p6b.py --p6b                  # 216 calls, sonnet
    python3 p4f_p6b.py --p4f --model opus     # opus arm (§10 v5 追加, registered)
    python3 p4f_p6b.py --p6b --model opus
Resume-safe: rerun the same command to continue. Outputs in ../data/
(sonnet keeps original filenames; other models get a _<model> suffix).
"""
import subprocess, json, re, os, sys, random, time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PER = 24; CONC = 6

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

# ---- P4-F: FROZEN 2x2 wordings (rate-led vs count-led) x (accusatory vs neutral) ----
P4F = {
 "F_RA": "37.5% of participants (3 of 8) have reported P{X} for not transferring in earlier rounds.",
 "F_RN": "37.5% of participants (3 of 8) say that P{X} did not transfer in earlier rounds.",
 "F_CA": "3 of the 8 participants (37.5%) have reported P{X} for not transferring in earlier rounds.",
 "F_CN": "3 of the 8 participants (37.5%) say that P{X} did not transfer in earlier rounds.",
}

# ---- P6b: wordings FROZEN at original P6 registration (W1, W2); W3 was the main run ----
P6B_WORDINGS = {
 1: "{K} mentioned being unsure whether P{X} transferred in an earlier round.",
 2: "{K} said P{X} seemed hesitant to transfer in an earlier round.",
}
NS = [40, 200, 1000]

def count_phrase(k): return "1 participant" if k == 1 else f"{k} participants"

def build_p4f(cellkey, rng):
    N = 8
    ids = [x % N for x in rng.sample(range(8), 3)]
    while len(set(ids)) < 3:
        ids = [x % N for x in rng.sample(range(8), 3)]
    agent, partner, _ = ids
    credits = rng.randint(8, 14)
    ctx = ("What you know about your current partner:\n- " +
           P4F[cellkey].format(X=partner))
    return SKELETON.format(agent=agent, N=N, credits=credits, partner=partner, ctx=ctx)

def build_p6b(k, N, w, rng):
    ids = [x % N for x in rng.sample(range(N), 3)]
    while len(set(ids)) < 3:
        ids = [x % N for x in rng.sample(range(N), 3)]
    agent, partner, _ = ids
    credits = rng.randint(8, 14)
    if k == 0:
        ctx = "You have no information about your current partner."
    else:
        ctx = ("What you know about your current partner:\n- " +
               P6B_WORDINGS[w].format(K=count_phrase(k), X=partner))
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

def call_full(prompt, env, model="sonnet"):
    txt, envelope = "", None
    for _ in range(2):
        try:
            r = subprocess.run(["claude", "-p", prompt, "--model", model,
                                "--output-format", "json"],
                               capture_output=True, text=True, timeout=240, env=env,
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

def run_jobs(jobs, raw_path, env, keyfields, model="sonnet"):
    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            try:
                d = json.loads(line)
                if d.get("action") in ("give", "keep"):
                    done.add(tuple(d[f] for f in keyfields))
            except Exception: pass
    jobs = [j for j in jobs if tuple(j[0][f] for f in keyfields) not in done]
    print(f"{len(jobs)} calls to make.", flush=True)
    lock = Lock(); fout = open(raw_path, "a")
    def work(j):
        meta, prompt = j
        act, txt, envelope = call_full(prompt, env, model)
        mu = (envelope or {}).get("modelUsage")
        row = dict(meta)
        row.update({"action": act,
                    "models_used": sorted(mu.keys()) if isinstance(mu, dict) else None,
                    "envelope": envelope, "response": (txt or "")[:200],
                    "prompt": prompt})
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
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
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "sonnet"
    sfx = "" if model == "sonnet" else f"_{model}"
    if "--p4f" in sys.argv:
        rng = random.Random(20260719)
        raw = os.path.join(DATA, f"p4f{sfx}_raw.jsonl")
        jobs = [({"cell": c, "i": i}, build_p4f(c, rng))
                for c in ["F_RA", "F_RN", "F_CA", "F_CN"] for i in range(N_PER)]
        run_jobs(jobs, raw, env, ("cell", "i"), model)
        rr = rates(raw, lambda d: d["cell"])
        out = {"experiment": "P4F_factor_separation", "model": model, "cells": rr,
               "registered_readings": "see 实验计划.md §10 (a)-(d) / opus arm (i)-(iii)"}
        json.dump(out, open(os.path.join(DATA, f"p4f{sfx}_summary.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
    elif "--p6b" in sys.argv:
        rng = random.Random(20260720)
        raw = os.path.join(DATA, f"p6b{sfx}_raw.jsonl")
        jobs = ([({"wording": 0, "k": 0, "N": N, "i": i}, build_p6b(0, N, None, rng))
                 for N in NS for i in range(N_PER)] +
                [({"wording": w, "k": 1, "N": N, "i": i}, build_p6b(1, N, w, rng))
                 for w in (1, 2) for N in NS for i in range(N_PER)])
        run_jobs(jobs, raw, env, ("wording", "k", "N", "i"), model)
        rr = rates(raw, lambda d: f"w{d['wording']}_k{d['k']}_N{d['N']}")
        pooled = rates(raw, lambda d: f"w{d['wording']}_k{d['k']}")
        out = {"experiment": "P6b_jump_wording_retest", "model": model,
               "cells": rr, "pooled": pooled,
               "registered_readings": "Delta=give(k1,w)-give(k0); see 实验计划.md §10 (a)-(c) / opus arm"}
        json.dump(out, open(os.path.join(DATA, f"p6b{sfx}_summary.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
    else:
        print("use --p4f or --p6b [--model opus]")

if __name__ == "__main__":
    main()
