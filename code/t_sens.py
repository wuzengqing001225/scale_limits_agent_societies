"""Temperature sensitivity probe (registered 实验计划.md, amendment 2026-07-19).
Cells P1_T / P1_B / P2_count_1000 x temperature {0, 0.7, 1.0} x n=24 = 216 calls.
Model pinned to API string claude-sonnet-5; response body `model` field logged.
Registered prediction: step DIRECTION preserved at every temperature.
Snapshot rule: within-run comparisons only.
Usage (user machine):
    export ANTHROPIC_API_KEY=...
    python3 t_sens.py
"""
import json, os, re, sys, random, time
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MODEL = "claude-sonnet-5"
N_PER = 24
TEMPS = [0.0, 0.7, 1.0]

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
NOTE_T = "Note from P{A}: P{S} chose keep against me in an earlier round."
NO_INFO = "You have no information about your current partner."
COUNT = "3 participants have reported that P{P} chose keep against them in earlier rounds."

def build_prompt(cell, rng):
    N = 32 if cell.startswith("P1") else 1000
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
    elif cell == "P2_count_1000":
        ctx = hdr + COUNT.format(P=partner)
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

def call_api(prompt, temp, key):
    body = json.dumps({"model": MODEL, "max_tokens": 64, "temperature": temp,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                env = json.loads(r.read())
            txt = "".join(b.get("text", "") for b in env.get("content", []))
            return parse_action(txt), txt, env.get("model"), env.get("usage")
        except urllib.error.HTTPError as e:
            body = ""
            try: body = e.read().decode()[:300]
            except Exception: pass
            if e.code == 400 or attempt == 3:   # 400 = param rejected, no point retrying
                return None, f"<HTTP {e.code}: {body}>", None, None
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            if attempt == 3:
                return None, f"<{type(e).__name__}: {str(e)[:150]}>", None, None
            time.sleep(3 * (attempt + 1))

def call_gemini(prompt, temp, key):
    body = json.dumps({"model": "gemini-3.5-flash", "max_tokens": 2048, "temperature": temp,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        data=body, headers={"Authorization": f"Bearer {key}",
                            "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                env = json.loads(r.read())
            txt = env["choices"][0]["message"].get("content") or ""
            return parse_action(txt), txt, env.get("model"), env.get("usage")
        except urllib.error.HTTPError as e:
            body_txt = ""
            try: body_txt = e.read().decode()[:300]
            except Exception: pass
            if e.code == 400 or attempt == 5:
                return None, f"<HTTP {e.code}: {body_txt}>", None, None
            time.sleep(40 * (attempt + 1) if e.code == 429 else 4 * (attempt + 1))
        except Exception as e:
            if attempt == 5:
                return None, f"<{type(e).__name__}: {str(e)[:150]}>", None, None
            time.sleep(4 * (attempt + 1))

def main():
    engine = "gemini" if "--engine" in sys.argv and \
        sys.argv[sys.argv.index("--engine") + 1] == "gemini" else "anthropic"
    if engine == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or sys.exit("set GEMINI_API_KEY")
        caller, tag = call_gemini, "t_sens_gemini"
    else:
        key = os.environ.get("ANTHROPIC_API_KEY") or sys.exit("set ANTHROPIC_API_KEY")
        caller, tag = call_api, "t_sens"
    if "--diag" in sys.argv:   # one t=0 call, print full error body
        rng = random.Random(1)
        act, txt, mid, usage = caller(build_prompt("P1_B", rng), 0.0, key)
        print("action:", act, "| model:", mid, "\n", (txt or "")[:400])
        return
    raw_path = os.path.join(DATA, f"{tag}_raw.jsonl")
    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            try:
                d = json.loads(line)
                if d.get("action") in ("give", "keep"):
                    done.add((d["cell"], d["temp"], d["i"]))
            except Exception: pass
    rng = random.Random(20260719)
    jobs = [(c, t, i, build_prompt(c, rng)) for c in ["P1_T", "P1_B", "P2_count_1000"]
            for t in TEMPS for i in range(N_PER)]
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} calls (engine={engine}).", flush=True)
    with open(raw_path, "a") as fout:
        for k_, (cell, temp, i, prompt) in enumerate(jobs):
            act, txt, mid, usage = caller(prompt, temp, key)
            fout.write(json.dumps({"cell": cell, "temp": temp, "i": i, "action": act,
                                   "resolved_model": mid, "usage": usage,
                                   "response": (txt or "")[:200], "prompt": prompt}) + "\n")
            fout.flush()
            if (k_ + 1) % 20 == 0: print(f"{k_+1}/{len(jobs)}", flush=True)
            time.sleep(0.3)
    from collections import defaultdict
    tal = defaultdict(lambda: [0, 0])
    models = set()
    for line in open(raw_path):
        d = json.loads(line)
        if d["action"] in ("give", "keep"):
            g, n = tal[(d["cell"], d["temp"])]
            tal[(d["cell"], d["temp"])] = [g + (d["action"] == "give"), n + 1]
        if d.get("resolved_model"): models.add(d["resolved_model"])
    rates = {f"{c}_t{t}": {"give": round(g / n, 3), "n": n}
             for (c, t), (g, n) in sorted(tal.items()) if n}
    # registered verdict: step direction preserved at every temperature
    verdict = {}
    for t in TEMPS:
        b = tal.get(("P1_B", t), (0, 0)); s = tal.get(("P1_T", t), (0, 0))
        c = tal.get(("P2_count_1000", t), (0, 0))
        if min(b[1], s[1], c[1]) == 0: continue
        verdict[f"t{t}"] = {"baseline_give": round(b[0]/b[1], 3),
                            "targeted_give": round(s[0]/s[1], 3),
                            "count1000_give": round(c[0]/c[1], 3),
                            "direction_preserved": (b[0]/b[1] > s[0]/s[1]) and (b[0]/b[1] > c[0]/c[1])}
    out = {"experiment": "T_sensitivity", "engine": engine,
           "model_requested": MODEL if engine == "anthropic" else "gemini-3.5-flash",
           "models_resolved": sorted(models), "rates": rates, "verdict": verdict}
    json.dump(out, open(os.path.join(DATA, f"{tag}_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
