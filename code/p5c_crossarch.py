"""P5c: cross-architecture replication (registered 实验计划.md §10, 2026-07-19).
6 cells x V1 frozen wording x n=24 = 144 calls per engine.
Engines (OpenAI-compatible chat completions):
    deepseek-chat, deepseek-reasoner  -> DEEPSEEK_API_KEY, https://api.deepseek.com/v1
    gemini (e.g. gemini-2.5-flash)    -> GEMINI_API_KEY,
        https://generativelanguage.googleapis.com/v1beta/openai/
Temperature: provider default (not set). Response `model` field logged.
Usage (user machine):
    export DEEPSEEK_API_KEY=...
    python3 p5c_crossarch.py --engine deepseek-chat
    python3 p5c_crossarch.py --engine deepseek-reasoner
    export GEMINI_API_KEY=...
    python3 p5c_crossarch.py --engine gemini   # 默认 gemini-3.5-flash
Resume-safe; outputs ../data/p5c_<engine>_raw.jsonl + _summary.json.
"""
import json, os, re, sys, random, time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
N_PER = 24

ENGINES = {
    "deepseek-chat":     {"base": "https://api.deepseek.com/v1", "key": "DEEPSEEK_API_KEY",
                          "model": "deepseek-chat"},
    "deepseek-reasoner": {"base": "https://api.deepseek.com/v1", "key": "DEEPSEEK_API_KEY",
                          "model": "deepseek-reasoner", "max_tokens": 8192},
    "gemini":            {"base": "https://generativelanguage.googleapis.com/v1beta/openai",
                          "key": "GEMINI_API_KEY", "model": "gemini-3.5-flash",
                          "sleep": 1.0},
}

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
# V1 frozen wordings (identical to P4/P5 scripts)
NOTE_T = "Note from P{A}: P{S} chose keep against me in an earlier round."
NO_INFO = "You have no information about your current partner."
COUNT = "3 participants have reported that P{P} chose keep against them in earlier rounds."
PROP8 = "3 of the 8 participants (37.5%) have reported that P{P} chose keep against them in earlier rounds."
PROP1000 = "3 of the 1000 participants (0.3%) have reported that P{P} chose keep against them in earlier rounds."

CELLS = ["P1_T", "P1_B", "P1_X", "P2_count_1000", "P2_prop_8", "P2_prop_1000"]

def build_prompt(cell, rng):
    N = 32 if cell.startswith("P1") else int(cell.rsplit("_", 1)[1])
    ids = rng.sample(range(max(N, 8)), 4)
    agent, partner, a, z = [x % N for x in ids]
    while len({agent, partner, a, z}) < 4:
        agent, partner, a, z = [x % N for x in rng.sample(range(max(N, 8)), 4)]
    credits = rng.randint(8, 14)
    hdr = "What you know about your current partner:\n- "
    if cell == "P1_T":
        ctx = hdr + NOTE_T.format(A=a, S=partner)
    elif cell == "P1_B":
        ctx = NO_INFO
    elif cell == "P1_X":
        ctx = "What you know:\n- " + NOTE_T.format(A=a, S=z) + "\n" + NO_INFO
    elif cell == "P2_count_1000":
        ctx = hdr + COUNT.format(P=partner)
    elif cell == "P2_prop_8":
        ctx = hdr + PROP8.format(P=partner)
    elif cell == "P2_prop_1000":
        ctx = hdr + PROP1000.format(P=partner)
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

def call_api(prompt, cfg, key):
    body = json.dumps({"model": cfg["model"], "max_tokens": cfg.get("max_tokens", 2048),
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(cfg["base"].rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                env = json.loads(r.read())
            msg = env["choices"][0]["message"]
            txt = msg.get("content") or ""
            return parse_action(txt), txt, env.get("model"), env.get("usage")
        except Exception as e:
            is_429 = "429" in str(e)
            if attempt == 5:
                return None, f"<{type(e).__name__}: {str(e)[:200]}>", None, None
            time.sleep(40 * (attempt + 1) if is_429 else 4 * (attempt + 1))

def main():
    eng = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else sys.exit("--engine required")
    cfg = dict(ENGINES[eng])
    if "--api-model" in sys.argv:
        cfg["model"] = sys.argv[sys.argv.index("--api-model") + 1]
    key = os.environ.get(cfg["key"]) or sys.exit(f"set {cfg['key']}")
    raw_path = os.path.join(DATA, f"p5c_{eng}_raw.jsonl")
    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            try:
                d = json.loads(line)
                if d.get("action") in ("give", "keep"): done.add((d["cell"], d["i"]))
            except Exception: pass
    rng = random.Random(20260719)
    jobs = [(c, i, build_prompt(c, rng)) for c in CELLS for i in range(N_PER)]
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"{len(jobs)} calls (engine={eng}, model={cfg['model']}).", flush=True)
    with open(raw_path, "a") as fout:
        for k_, (cell, i, prompt) in enumerate(jobs):
            act, txt, mid, usage = call_api(prompt, cfg, key)
            fout.write(json.dumps({"cell": cell, "i": i, "action": act, "engine": eng,
                                   "resolved_model": mid, "usage": usage,
                                   "response": (txt or "")[:300], "prompt": prompt}) + "\n")
            fout.flush()
            if (k_ + 1) % 20 == 0: print(f"{k_+1}/{len(jobs)}", flush=True)
            time.sleep(cfg.get("sleep", 0.5))
    from collections import defaultdict
    tal = defaultdict(lambda: [0, 0]); models = set(); invalid = 0
    for line in open(raw_path):
        d = json.loads(line)
        if d["action"] in ("give", "keep"):
            g, n = tal[d["cell"]]
            tal[d["cell"]] = [g + (d["action"] == "give"), n + 1]
        else:
            invalid += 1
        if d.get("resolved_model"): models.add(d["resolved_model"])
    rates = {c: {"give": round(g / n, 3), "n": n} for c, (g, n) in sorted(tal.items()) if n}
    def r(c): return rates.get(c, {}).get("give")
    cls = {"conditional_structure": (r("P1_B") is not None and r("P1_T") is not None
           and r("P1_B") >= 0.8 and r("P1_T") <= 0.2)}
    if r("P2_prop_8") is not None:
        p8, p1000 = r("P2_prop_8"), r("P2_prop_1000")
        if p8 >= 0.8: cls["proportion_type"] = "discount"
        elif p8 <= 0.2 and (p1000 is not None and p1000 <= 0.2): cls["proportion_type"] = "testimony_as_evidence"
        elif p8 <= 0.2 and (p1000 is not None and p1000 >= 0.8): cls["proportion_type"] = "TRUE_PROPORTION_SENSITIVE"
        else: cls["proportion_type"] = "intermediate_report_as_is"
    out = {"experiment": "P5c_crossarch", "engine": eng, "model_requested": cfg["model"],
           "models_resolved": sorted(models), "rates": rates, "invalid": invalid,
           "classification": cls,
           "registered_readings": "see 实验计划.md §10 P5c (1)-(4)"}
    json.dump(out, open(os.path.join(DATA, f"p5c_{eng}_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
