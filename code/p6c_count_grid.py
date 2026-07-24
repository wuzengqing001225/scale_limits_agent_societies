"""P6c: count-format lattice across architectures (registered 实验计划.md P6c,
2026-07-20, BEFORE data; confirmatory with sealed per-engine criteria:
(i) b_logk > 0 significant, (ii) 95% CI of b_logN contains 0,
(iii) anti-diagonal (same 2.5%) NOT equal, spread > 0.15).
Wordings FROZEN at registration: V1 count family (no denominator mentioned) +
P6 weak wordings W1-W3 for the calibration step.
Usage (user machine; calibrate first, then lattice with the chosen wording):
    export DEEPSEEK_API_KEY=...   # reasoner
    python3 p6c_count_grid.py --engine reasoner --calibrate
    python3 p6c_count_grid.py --engine reasoner --wording V1   # or W1/W2/W3
    export GEMINI_API_KEY=...     # gemini
    python3 p6c_count_grid.py --engine gemini --calibrate
    python3 p6c_count_grid.py --engine gemini --wording ...
    python3 p6c_count_grid.py --engine gpt55 --calibrate       # codex CLI
    python3 p6c_count_grid.py --engine gpt55 --wording ...
Resume-safe. Calibration rule (registered): pick wording with give(k5,N200) in
[0.3, 0.7] nearest 0.5; if none, nearest overall and report saturation.
"""
import json, os, re, sys, random, time, subprocess, tempfile
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PER = 24; N_CAL = 12
KS = [1, 5, 25]; NS = [40, 200, 1000]

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
B_WORDING = "You have no information about your current partner."

def count_phrase(k): return "1 participant" if k == 1 else f"{k} participants"

WORDINGS = {
 "V1": "{K} {HV} reported that P{X} chose keep against them in earlier rounds.",
 "W1": "{K} mentioned being unsure whether P{X} transferred in an earlier round.",
 "W2": "{K} said P{X} seemed hesitant to transfer in an earlier round.",
 "W3": "{K} reported that P{X} sometimes chose keep in earlier rounds.",
}

def t_wording(w, k, X):
    hv = "has" if k == 1 else "have"
    return WORDINGS[w].format(K=count_phrase(k), HV=hv, X=X)

def build_prompt(w, k, N, rng):
    ids = [x % N for x in rng.sample(range(N), 3)]
    while len(set(ids)) < 3:
        ids = [x % N for x in rng.sample(range(N), 3)]
    agent, partner, _ = ids
    credits = rng.randint(8, 14)
    if k == 0:
        ctx = B_WORDING
    else:
        ctx = ("What you know about your current partner:\n- " + t_wording(w, k, partner))
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

def call_openai_compat(prompt, base, model, key, max_tokens):
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                env = json.loads(r.read())
            txt = env["choices"][0]["message"].get("content") or ""
            return parse_action(txt), txt, env.get("model")
        except Exception as e:
            if attempt == 5:
                return None, f"<{type(e).__name__}: {str(e)[:200]}>", None
            time.sleep(40 * (attempt + 1) if "429" in str(e) else 4 * (attempt + 1))

def call_codex(prompt):
    for _ in range(2):
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False); tmp.close()
        try:
            r = subprocess.run(["codex", "exec", "--skip-git-repo-check", "-m", "gpt-5.5",
                                "--output-last-message", tmp.name, prompt],
                               capture_output=True, text=True, timeout=300,
                               stdin=subprocess.DEVNULL)
            txt = ""
            if os.path.exists(tmp.name):
                try: txt = open(tmp.name).read().strip()
                except OSError: txt = ""
            if not txt:
                txt = re.sub(r'\x1b\[[0-9;]*m', '', (r.stdout or "").strip())
            act = parse_action(txt)
            if act:
                return act, txt, "gpt-5.5(codex)"
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            txt = f"<{type(e).__name__}>"
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass
        time.sleep(2)
    return None, txt, None

def get_caller(engine):
    if engine == "reasoner":
        key = os.environ.get("DEEPSEEK_API_KEY") or sys.exit("set DEEPSEEK_API_KEY")
        return lambda p: call_openai_compat(p, "https://api.deepseek.com/v1",
                                            "deepseek-reasoner", key, 8192), 4
    if engine == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or sys.exit("set GEMINI_API_KEY")
        return lambda p: call_openai_compat(
            p, "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-3.5-flash", key, 2048), 4
    if engine == "gpt55":
        return lambda p: call_codex(p), 4
    sys.exit("--engine reasoner|gemini|gpt55")

def run_jobs(jobs, raw_path, caller, conc, keyfields):
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
        act, txt, mid = caller(prompt)
        row = dict(meta)
        row.update({"action": act, "resolved_model": mid,
                    "response": (txt or "")[:200], "prompt": prompt})
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
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
    engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else sys.exit("--engine required")
    caller, conc = get_caller(engine)
    rng = random.Random(20260722)
    if "--calibrate" in sys.argv:
        raw = os.path.join(DATA, f"p6c_{engine}_cal_raw.jsonl")
        jobs = [({"wording": w, "k": 5, "N": 200, "i": i}, build_prompt(w, 5, 200, rng))
                for w in ["V1", "W1", "W2", "W3"] for i in range(N_CAL)]
        run_jobs(jobs, raw, caller, conc, ("wording", "i"))
        rr = rates(raw, lambda d: d["wording"])
        print(json.dumps(rr, indent=1))
        ok = {k: v for k, v in rr.items() if 0.3 <= v["give"] <= 0.7}
        if ok:
            best = min(ok, key=lambda k: abs(ok[k]["give"] - 0.5))
            print(f"CHOSEN wording: {best} (give={ok[best]['give']}). "
                  f"Run: python3 p6c_count_grid.py --engine {engine} --wording {best}")
        else:
            best = min(rr, key=lambda k: abs(rr[k]["give"] - 0.5)) if rr else None
            print(f"NO wording in [0.3,0.7]; nearest is {best} — saturation to be "
                  f"reported honestly per registration.")
        return
    w = sys.argv[sys.argv.index("--wording") + 1]
    raw = os.path.join(DATA, f"p6c_{engine}_raw.jsonl")
    jobs = ([({"wording": w, "k": 0, "N": N, "i": i}, build_prompt(w, 0, N, rng))
             for N in NS for i in range(N_PER)] +
            [({"wording": w, "k": k, "N": N, "i": i}, build_prompt(w, k, N, rng))
             for k in KS for N in NS for i in range(N_PER)])
    run_jobs(jobs, raw, caller, conc, ("wording", "k", "N", "i"))
    rr = rates(raw, lambda d: f"k{d['k']}_N{d['N']}")
    anti = {c: rr.get(c, {}).get("give") for c in ["k1_N40", "k5_N200", "k25_N1000"]}
    out = {"experiment": "P6c_count_grid", "engine": engine, "wording": w,
           "rates": rr, "anti_diagonal_2.5pct": anti,
           "sealed_criteria": "(i) b_logk>0 sig; (ii) CI(b_logN) contains 0; "
                              "(iii) anti-diagonal spread > 0.15 — GLM in sandbox"}
    json.dump(out, open(os.path.join(DATA, f"p6c_{engine}_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
