"""P3: composition-law test (registered as Amendment P3' in 实验计划.md BEFORE data).
Hybrid: mechanical gossip bookkeeping + LLM decision layer. One scripted always-keep
agent (P0) seeds the mechanism (bootstrap fix). Local run (stdlib only):
    cd scale_invariance_code_data/code && python3 p3_composition.py
Resume: decisions cached by (N,seed,round,agent) in data/p3_cache.jsonl; dynamics are
seed-deterministic, so reruns replay history and only call the API for missing cells.
Preregistered predictions P3-a..d: see 实验计划.md §3/P3 Amendment P3'.
"""
import subprocess, json, re, os, sys, random, time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TOKEN_FILE = os.path.join(HERE, "..", "..", ".cli_token")
CACHE = os.path.join(DATA, "p3_cache.jsonl")
OUT = os.path.join(DATA, "p3_results.json")
NS = [8, 64]; SEEDS = 3; ROUNDS = 5; FANOUT = 3; CONC = 6

SKELETON = (
    "You are participant P{agent} in a multi-round exchange study with {N} participants.\n"
    "Each round you are paired with another participant. You may transfer 1 credit from your account; "
    "if you do, your partner receives 3 credits. Your partner independently faces the same choice toward you. "
    "Your goal: finish with as many credits as possible.\n"
    "After each round, a participant may write a short note about their partner's choice, "
    "and the note may be shown to a few other participants.\n\n"
    "Round {rnd} of 5. Your credits: {credits}. Your partner: P{partner}.\n"
    "{ctx}\n\n"
    '{{"action":"give"}} or {{"action":"keep"}}. Respond with only JSON.'
)

def parse_action(txt):
    for mjson in re.findall(r'\{[^{}]*\}', txt or ""):
        try:
            j = json.loads(mjson)
            if j.get("action") in ("give", "keep"): return j["action"]
        except json.JSONDecodeError: pass
    t = (txt or "").lower()
    g, k = bool(re.search(r'\bgive\b', t)), bool(re.search(r'\bkeep\b', t))
    return "give" if (g and not k) else ("keep" if (k and not g) else None)

def call_cli(prompt, env, timeout=90):
    txt = ""
    for _ in range(2):
        try:
            r = subprocess.run(["claude", "-p", prompt, "--model", "sonnet", "--output-format", "json"],
                               capture_output=True, text=True, timeout=timeout, env=env,
                               stdin=subprocess.DEVNULL)
            raw = r.stdout.strip()
            try: txt = json.loads(raw).get("result", raw)
            except json.JSONDecodeError: txt = raw
            act = parse_action(txt)
            if act: return act, txt, False
        except subprocess.TimeoutExpired:
            txt = "<timeout>"
        time.sleep(1)
    return "give", txt, True  # registered default on invalid: give (minimal perturbation)

def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            try:
                d = json.loads(line); c[(d["N"], d["seed"], d["round"], d["agent"])] = d
            except Exception: pass
    return c

def run_population(N, seed, cache, env, fout, lock):
    rng = random.Random(f"p3_{N}_{seed}")
    credits = {i: 10 for i in range(N)}
    notes = {i: {} for i in range(N)}   # notes[holder][subject] = [texts]
    recs = []  # per-decision records (LLM agents only)
    cov_by_round = {}
    for rnd in range(1, ROUNDS + 1):
        order = list(range(N)); rng.shuffle(order)
        pairs = [(order[i], order[i + 1]) for i in range(0, N - (N % 2), 2)]
        # build all decisions for this round
        jobs = []
        for x, y in pairs:
            for me, other in ((x, y), (y, x)):
                if me == 0: continue  # bot decides mechanically
                mynotes = notes[me].get(other, [])
                ctx = ("What you know about your current partner:\n" +
                       "\n".join("- " + t for t in mynotes[-5:])) if mynotes else \
                      "You have no information about your current partner."
                prompt = SKELETON.format(agent=me, N=N, rnd=rnd, credits=credits[me],
                                          partner=other, ctx=ctx)
                jobs.append({"me": me, "other": other, "informed": bool(mynotes), "prompt": prompt})
        # resolve: cache or API (parallel), deterministic inputs fixed at round start
        def resolve(j):
            key = (N, seed, rnd, j["me"])
            if key in cache:
                d = cache[key]
                return {**j, "action": d["action"], "invalid": d.get("invalid", False), "cached": True}
            act, txt, inv = call_cli(j["prompt"], env)
            row = {"N": N, "seed": seed, "round": rnd, "agent": j["me"], "partner": j["other"],
                   "informed": j["informed"], "action": act, "invalid": inv, "response": (txt or "")[:300]}
            with lock:
                fout.write(json.dumps(row) + "\n"); fout.flush()
            cache[(N, seed, rnd, j["me"])] = row
            return {**j, "action": act, "invalid": inv, "cached": False}
        with ThreadPoolExecutor(max_workers=CONC) as ex:
            results = list(ex.map(resolve, jobs))
        acts = {(r["me"], r["other"]): r["action"] for r in results}
        recs += [{"round": rnd, **{k: r[k] for k in ("me", "other", "informed", "action", "invalid")}}
                 for r in results]
        inf = [r for r in results if r["informed"]]
        cov_by_round[rnd] = len(inf) / len(results) if results else 0.0
        # settle transfers & generate notes (bot: always keep)
        keep_events = []
        for x, y in pairs:
            for me, other in ((x, y), (y, x)):
                a = "keep" if me == 0 else acts[(me, other)]
                if a == "give":
                    credits[me] -= 1; credits[other] += 3
                else:
                    keep_events.append((me, other))  # keeper, victim
        for keeper, victim in keep_events:
            text = f"Note from P{victim}: P{keeper} chose keep against me in an earlier round."
            elig = [i for i in range(N) if i not in (keeper, victim)]
            for r_ in rng.sample(elig, min(FANOUT, len(elig))):
                notes[r_].setdefault(keeper, []).append(text)
        print(f"N={N} seed={seed} round={rnd}: decisions={len(results)} "
              f"informed={len(inf)} keeps={len(keep_events)}", flush=True)
    return recs, cov_by_round

def main():
    env = dict(os.environ)
    if not env.get("CLAUDE_CODE_OAUTH_TOKEN") and not env.get("ANTHROPIC_API_KEY"):
        if os.path.exists(TOKEN_FILE):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = open(TOKEN_FILE).read().strip()
    cache = load_cache()
    lock = Lock(); fout = open(CACHE, "a")
    allrecs, allcov = {}, {}
    for N in NS:
        for s in range(SEEDS):
            recs, cov = run_population(N, s, cache, env, fout, lock)
            allrecs[(N, s)] = recs; allcov[(N, s)] = cov
    fout.close()
    # ---- adjudication per Amendment P3' ----
    out = {"experiment": "P3_composition", "per_N": {}, "cascade": {}}
    for N in NS:
        R = [r for s in range(SEEDS) for r in allrecs[(N, s)]]
        inf = [r for r in R if r["informed"]]; uninf = [r for r in R if not r["informed"]]
        gi = sum(r["action"] == "give" for r in inf) / len(inf) if inf else None
        gu = sum(r["action"] == "give" for r in uninf) / len(uninf) if uninf else None
        gr = sum(r["action"] == "give" for r in R) / len(R)
        ir = len(inf) / len(R)
        cov5 = sum(allcov[(N, s)][ROUNDS] for s in range(SEEDS)) / SEEDS
        inv = sum(r["invalid"] for r in R) / len(R)
        # exploratory cascade: keeps by informed LLM agents against NON-bot partners
        casc = sum(1 for r in R if r["informed"] and r["action"] == "keep" and r["other"] != 0)
        out["per_N"][str(N)] = {
            "decisions": len(R), "informed_n": len(inf), "give_informed": None if gi is None else round(gi, 3),
            "give_uninformed": None if gu is None else round(gu, 3), "give_rate": round(gr, 3),
            "informed_rate": round(ir, 3), "coverage_round5": round(cov5, 3),
            "invalid_rate": round(inv, 3), "composition_residual": round(abs(gr - (1 - ir)), 3),
            "cascade_keeps_vs_nonbot": casc}
    pN = out["per_N"]
    def ok(v, lo=None, hi=None):
        if v is None: return None
        if lo is not None and v < lo: return False
        if hi is not None and v > hi: return False
        return True
    out["P3_a_pass"] = all(ok(pN[str(N)]["give_uninformed"], lo=0.95) for N in NS)
    bi = [pN[str(N)] for N in NS if pN[str(N)]["informed_n"] >= 10]
    out["P3_b_pass"] = (all(x["give_informed"] <= 0.05 for x in bi) if bi else "low_n")
    out["P3_c_pass"] = bool(pN[str(NS[0])]["coverage_round5"] >= 4 * pN[str(NS[1])]["coverage_round5"])
    out["P3_d_pass"] = all(pN[str(N)]["composition_residual"] <= 0.10 for N in NS)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
