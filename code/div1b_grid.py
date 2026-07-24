"""DIV-1b: proportion x group-size unconfounding grid (registered 实验计划.md
DIV-1b, 2026-07-20, BEFORE data; classificatory, no favoured outcome).
Same lattice as P6 (k in {1,5,25} x N in {40,200,1000} + k=0 baselines), but the
report wording states the explicit proportion. Analysis (registered, mirrors
P6): GLM keep ~ b1*log k + b2*log N, contrast H0 b1=-b2, anti-diagonal equality.
Usage (user machine, both arms same day):
    python3 div1b_grid.py --engine sonnet          # 288 calls via claude CLI
    export DEEPSEEK_API_KEY=...
    python3 div1b_grid.py --engine reasoner        # 288 calls via DeepSeek API
Resume-safe; outputs ../data/div1b_<engine>_raw.jsonl + _summary.json.
GLM computed separately in sandbox (statsmodels); summary here reports rates only.
"""
import json, os, re, sys, random, time, subprocess
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PER = 24
KS = [1, 5, 25]; NS = [40, 200, 1000]

def pct_str(k, N):
    v = 100.0 * k / N
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return s

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

def t_wording(k, N, X):
    verb = "has" if k == 1 else "have"
    return (f"{k} of the {N} participants ({pct_str(k, N)}%) {verb} reported that "
            f"P{X} chose keep against them in earlier rounds.")

def build_prompt(k, N, rng):
    ids = [x % N for x in rng.sample(range(N), 3)]
    while len(set(ids)) < 3:
        ids = [x % N for x in rng.sample(range(N), 3)]
    agent, partner, _ = ids
    credits = rng.randint(8, 14)
    if k == 0:
        ctx = B_WORDING
    else:
        ctx = ("What you know about your current partner:\n- " + t_wording(k, N, partner))
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

def call_deepseek(prompt, key):
    body = json.dumps({"model": "deepseek-reasoner", "max_tokens": 8192,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions", data=body,
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

def call_cli(prompt, env):
    txt = ""
    for _ in range(2):
        try:
            r = subprocess.run(["claude", "-p", prompt, "--model", "sonnet",
                                "--output-format", "json"],
                               capture_output=True, text=True, timeout=240, env=env,
                               stdin=subprocess.DEVNULL)
            raw = (r.stdout or "").strip()
            try:
                envelope = json.loads(raw); txt = envelope.get("result", raw)
            except json.JSONDecodeError:
                envelope, txt = {}, raw
            act = parse_action(txt)
            if act:
                mu = envelope.get("modelUsage") if isinstance(envelope, dict) else None
                return act, txt, sorted(mu.keys()) if isinstance(mu, dict) else None
        except subprocess.TimeoutExpired:
            txt = "<timeout>"
        time.sleep(1)
    return None, txt, None

def main():
    engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else sys.exit("--engine sonnet|reasoner")
    raw_path = os.path.join(DATA, f"div1b_{engine}_raw.jsonl")
    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            try:
                d = json.loads(line)
                if d.get("action") in ("give", "keep"): done.add((d["k"], d["N"], d["i"]))
            except Exception: pass
    rng = random.Random(20260721)
    jobs = ([(0, N, i, build_prompt(0, N, rng)) for N in NS for i in range(N_PER)] +
            [(k, N, i, build_prompt(k, N, rng)) for k in KS for N in NS for i in range(N_PER)])
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} calls (engine={engine}).", flush=True)
    lock = Lock(); fout = open(raw_path, "a")
    if engine == "reasoner":
        key = os.environ.get("DEEPSEEK_API_KEY") or sys.exit("set DEEPSEEK_API_KEY")
        def work(j):
            k, N, i, prompt = j
            act, txt, mid = call_deepseek(prompt, key)
            with lock:
                fout.write(json.dumps({"k": k, "N": N, "i": i, "action": act,
                                       "resolved_model": mid,
                                       "response": (txt or "")[:200], "prompt": prompt}) + "\n")
                fout.flush()
        conc = 4
    else:
        env = dict(os.environ)
        def work(j):
            k, N, i, prompt = j
            act, txt, mid = call_cli(prompt, env)
            with lock:
                fout.write(json.dumps({"k": k, "N": N, "i": i, "action": act,
                                       "models_used": mid,
                                       "response": (txt or "")[:200], "prompt": prompt}) + "\n")
                fout.flush()
        conc = 6
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for c, _ in enumerate(ex.map(work, jobs)):
            if (c + 1) % 20 == 0: print(f"{c+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    fout.close()
    from collections import defaultdict
    tal = defaultdict(lambda: [0, 0])
    for line in open(raw_path):
        d = json.loads(line)
        if d["action"] in ("give", "keep"):
            g, n = tal[(d["k"], d["N"])]
            tal[(d["k"], d["N"])] = [g + (d["action"] == "give"), n + 1]
    rates = {f"k{k}_N{N}": {"give": round(g / n, 3), "n": n}
             for (k, N), (g, n) in sorted(tal.items()) if n}
    anti = {f"k{k}_N{N}": rates.get(f"k{k}_N{N}", {}).get("give")
            for k, N in [(1, 40), (5, 200), (25, 1000)]}
    out = {"experiment": "DIV1b_unconfound_grid", "engine": engine, "rates": rates,
           "anti_diagonal_2.5pct": anti,
           "registered_analysis": "GLM in sandbox; signatures (a)-(d) per 实验计划.md DIV-1b"}
    json.dump(out, open(os.path.join(DATA, f"div1b_{engine}_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
