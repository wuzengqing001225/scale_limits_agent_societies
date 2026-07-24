"""P1R: confirmatory rerun of the five most critical probe cells with FULL model
envelope logging (v4 audit round 2 requirement). Registered in 实验计划.md §8 BEFORE running.
Old P1/P2 data relabeled "initial discovery (indirect model attribution)"; this run is
the confirmatory evidence with exact model IDs.
Cells (V1 wording, frozen, identical strings to original probes):
  P1_B(no info) / P1_T(targeted) / P1_X(placebo) / P2_count_8 / P2_count_1000
Predictions (registered): give(B)>=0.8; give(T)<=0.2; give(X)>=0.8;
  give(count_8)<=0.2; give(count_1000)<=0.2; |count_8 - count_1000| < 10pp.
Usage (local Mac):  python3 probe_confirm.py          (~120 calls, sonnet)
"""
import subprocess, json, re, os, sys, random, time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PER = 24; CONC = 4
RAW = os.path.join(DATA, "p1r_confirm_raw.jsonl")

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
CELLS = ["P1_B", "P1_T", "P1_X", "P2_count_8", "P2_count_1000"]

def build_prompt(cell, rng):
    N = 32 if cell.startswith("P1") else int(cell.rsplit("_", 1)[1])
    ids = [x % N for x in rng.sample(range(max(N, 8)), 4)]
    while len(set(ids)) < 4:
        ids = [x % N for x in rng.sample(range(max(N, 8)), 4)]
    agent, partner, a, z = ids
    credits = rng.randint(8, 14)
    if cell == "P1_T":
        ctx = ("What you know about your current partner:\n"
               f"- Note from P{a}: P{partner} chose keep against me in an earlier round.")
    elif cell == "P1_B":
        ctx = "You have no information about your current partner."
    elif cell == "P1_X":
        ctx = ("What you know:\n"
               f"- Note from P{a}: P{z} chose keep against me in an earlier round.\n"
               "You have no information about your current partner.")
    else:
        ctx = ("What you know about your current partner:\n"
               f"- 3 participants have reported that P{partner} chose keep against them in earlier rounds.")
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
    g, k = bool(re.search(r'\bgive\b', t)), bool(re.search(r'\bkeep\b', t))
    return "give" if (g and not k) else ("keep" if (k and not g) else None)

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
                envelope, txt = {"_raw": raw[:1000]}, raw
            act = parse_action(txt)
            if act: return act, txt, envelope
        except subprocess.TimeoutExpired:
            txt = "<timeout>"
        time.sleep(1)
    return None, txt, envelope

def main():
    env = dict(os.environ)
    try:
        ver = subprocess.run(["claude", "--version"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        ver = "unknown"
    done = set()
    if os.path.exists(RAW):
        for line in open(RAW):
            try:
                d = json.loads(line)
                if d.get("action") in ("give", "keep"): done.add((d["cell"], d["i"]))
            except Exception: pass
    rng = random.Random(20260713)
    jobs = [(c, i, build_prompt(c, rng)) for c in CELLS for i in range(N_PER)]
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"P1R: {len(jobs)} calls. CLI: {ver}", flush=True)
    lock = Lock(); fout = open(RAW, "a")
    if not done:
        fout.write(json.dumps({"_header": True, "cli_version": ver,
                                "date": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
    def work(j):
        cell, i, prompt = j
        act, txt, envelope = call_full(prompt, env)
        mu = (envelope or {}).get("modelUsage")
        with lock:
            fout.write(json.dumps({"cell": cell, "i": i, "action": act,
                                    "models_used": sorted(mu.keys()) if isinstance(mu, dict) else None,
                                    "envelope": envelope, "response": (txt or "")[:200],
                                    "prompt": prompt}) + "\n"); fout.flush()
        return 1
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for k, _ in enumerate(ex.map(work, jobs)):
            if (k + 1) % 20 == 0: print(f"{k+1}/{len(jobs)}", flush=True)
    fout.close()
    # summary + registered verdicts
    from collections import defaultdict, Counter
    tal = defaultdict(lambda: [0, 0]); models = Counter()
    for line in open(RAW):
        d = json.loads(line)
        if d.get("_header"): continue
        if d["action"] in ("give", "keep"):
            tal[d["cell"]][0] += d["action"] == "give"; tal[d["cell"]][1] += 1
        for m in (d.get("models_used") or []): models[m] += 1
    rates = {c: round(g / n, 3) for c, (g, n) in tal.items() if n}
    B = {"P1_B": ("ge", .8), "P1_T": ("le", .2), "P1_X": ("ge", .8),
         "P2_count_8": ("le", .2), "P2_count_1000": ("le", .2)}
    verdicts = {c: (rates[c] >= b if op == "ge" else rates[c] <= b)
                for c, (op, b) in B.items() if c in rates}
    verdicts["denominator_neglect_repl"] = (abs(rates.get("P2_count_8", 0) -
                                                 rates.get("P2_count_1000", 0)) < 0.10)
    out = {"experiment": "P1R_confirmatory", "cli_version": ver, "rates": rates,
           "models_used_counts": dict(models), "verdicts": verdicts,
           "P1R_pass_all": all(verdicts.values())}
    json.dump(out, open(os.path.join(DATA, "p1r_confirm_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
