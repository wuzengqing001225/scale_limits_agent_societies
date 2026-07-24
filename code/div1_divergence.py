"""DIV-1: cross-architecture divergence prediction experiment (registered
实验计划.md §10/DIV-1 BEFORE data; predictions sealed, including interpolated
mid-points give(T,40)=0.396±0.20, give(T,200)=0.682±0.20, N_c≈71).
Design: k=3 fixed, N in {8,40,200,1000}; cells {B baseline, T proportion-format
targeted} x N x n=24 = 192 calls per engine.
Engines:
    reasoner  -> DeepSeek API (DEEPSEEK_API_KEY), deepseek-reasoner, max_tokens 8192
    sonnet    -> local claude CLI, --model sonnet, full envelope logging
Usage (user machine):
    export DEEPSEEK_API_KEY=...
    python3 div1_divergence.py --engine reasoner
    python3 div1_divergence.py --engine sonnet
Resume-safe. Within-run comparisons only (snapshot discipline).
"""
import json, os, re, sys, random, time, subprocess
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PER = 24
NS = [8, 40, 200, 1000]
PCT = {8: "37.5", 40: "7.5", 200: "1.5", 1000: "0.3"}

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
T_WORDING = ("3 of the {N} participants ({pct}%) have reported that P{X} chose keep "
             "against them in earlier rounds.")
B_WORDING = "You have no information about your current partner."

def build_prompt(cell, N, rng):
    ids = rng.sample(range(max(N, 8)), 3)
    agent, partner, _ = [x % N for x in ids]
    while agent == partner:
        agent, partner, _ = [x % N for x in rng.sample(range(max(N, 8)), 3)]
    credits = rng.randint(8, 14)
    if cell == "B":
        ctx = B_WORDING
    else:
        ctx = ("What you know about your current partner:\n- " +
               T_WORDING.format(N=N, pct=PCT[N], X=partner))
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
            return parse_action(txt), txt, env.get("model"), env.get("usage")
        except Exception as e:
            if attempt == 5:
                return None, f"<{type(e).__name__}: {str(e)[:200]}>", None, None
            time.sleep(40 * (attempt + 1) if "429" in str(e) else 4 * (attempt + 1))

def call_cli(prompt, env):
    txt, envelope = "", None
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
                envelope, txt = {"_raw": raw[:600]}, raw
            act = parse_action(txt)
            if act:
                mu = envelope.get("modelUsage") if isinstance(envelope, dict) else None
                mid = sorted(mu.keys()) if isinstance(mu, dict) else None
                return act, txt, mid, None
        except subprocess.TimeoutExpired:
            txt = "<timeout>"
        time.sleep(1)
    return None, txt, None, None

def main():
    engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else sys.exit("--engine reasoner|sonnet")
    raw_path = os.path.join(DATA, f"div1_{engine}_raw.jsonl")
    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            try:
                d = json.loads(line)
                if d.get("action") in ("give", "keep"): done.add((d["cell"], d["N"], d["i"]))
            except Exception: pass
    rng = random.Random(20260720)
    jobs = [(c, N, i, build_prompt(c, N, rng)) for c in ["B", "T"] for N in NS
            for i in range(N_PER)]
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} calls (engine={engine}).", flush=True)
    lock = Lock(); fout = open(raw_path, "a")
    if engine == "reasoner":
        key = os.environ.get("DEEPSEEK_API_KEY") or sys.exit("set DEEPSEEK_API_KEY")
        def work(j):
            cell, N, i, prompt = j
            act, txt, mid, usage = call_deepseek(prompt, key)
            with lock:
                fout.write(json.dumps({"cell": cell, "N": N, "i": i, "action": act,
                                       "resolved_model": mid, "usage": usage,
                                       "response": (txt or "")[:200], "prompt": prompt}) + "\n")
                fout.flush()
        conc = 4
    else:
        env = dict(os.environ)
        def work(j):
            cell, N, i, prompt = j
            act, txt, mid, _ = call_cli(prompt, env)
            with lock:
                fout.write(json.dumps({"cell": cell, "N": N, "i": i, "action": act,
                                       "models_used": mid,
                                       "response": (txt or "")[:200], "prompt": prompt}) + "\n")
                fout.flush()
        conc = 6
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for k_, _ in enumerate(ex.map(work, jobs)):
            if (k_ + 1) % 20 == 0: print(f"{k_+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    fout.close()
    # summary + sealed verdicts
    from collections import defaultdict
    tal = defaultdict(lambda: [0, 0])
    for line in open(raw_path):
        d = json.loads(line)
        if d["action"] in ("give", "keep"):
            g, n = tal[(d["cell"], d["N"])]
            tal[(d["cell"], d["N"])] = [g + (d["action"] == "give"), n + 1]
    rates = {f"{c}_N{N}": {"give": round(g / n, 3), "n": n}
             for (c, N), (g, n) in sorted(tal.items()) if n}
    def gv(c, N):
        g, n = tal.get((c, N), (0, 0))
        return g / n if n else None
    T = [gv("T", N) for N in NS]; B = [gv("B", N) for N in NS]
    verdict = {}
    if all(v is not None for v in T + B):
        D = [b - t for b, t in zip(B, T)]
        if engine == "reasoner":
            verdict = {
                "monotone_rank": T[0] < T[1] < T[2] < T[3],
                "anchor_N8_le_0.2": T[0] <= 0.2, "anchor_N1000_ge_0.8": T[3] >= 0.8,
                "mid_N40_pred_0.396_pm_0.20": abs(T[1] - 0.396) <= 0.20,
                "mid_N200_pred_0.682_pm_0.20": abs(T[2] - 0.682) <= 0.20,
                "divergence_rise_ge_0.4": (T[3] - T[0]) >= 0.4}
        else:
            verdict = {
                "T_flat_spread_lt_0.15": (max(T) - min(T)) < 0.15,
                "Delta_flat_spread_lt_0.15": (max(D) - min(D)) < 0.15,
                "Delta_positive_all_N": all(d > 0 for d in D),
                "flat_change_le_0.15": abs(T[3] - T[0]) <= 0.15}
    out = {"experiment": "DIV1_divergence", "engine": engine, "rates": rates,
           "Delta_by_N": {f"N{N}": (round(gv('B',N)-gv('T',N),3) if gv('B',N) is not None and gv('T',N) is not None else None) for N in NS},
           "sealed_verdicts": verdict,
           "registration": "实验计划.md DIV-1 (2026-07-20)"}
    json.dump(out, open(os.path.join(DATA, f"div1_{engine}_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
