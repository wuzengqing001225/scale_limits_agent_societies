"""P1 (Δ-probe + placebo) and P2 (denominator probe) via local Claude Code CLI.
Preregistered in 实验计划.md §3 BEFORE running. Model: sonnet (claude -p).
Auth: env CLAUDE_CODE_OAUTH_TOKEN, or token file ../../.cli_token (project root).
Usage: python3 probe_p1p2.py [--n 48] [--dry]  (--dry prints one prompt per cell, no calls)
"""
import subprocess, json, re, os, sys, random, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TOKEN_FILE = os.path.join(HERE, "..", "..", ".cli_token")
N_PER_CELL = 48
CONC = 6

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

def make_ids(rng, N, k=4):
    ids = rng.sample(range(N), k)
    return ids  # distinct ids

def build_prompt(cell, rng):
    if cell.startswith("P1"):
        N = 32
    else:
        N = int(cell.rsplit("_", 1)[1])
    agent, partner, a, z = make_ids(rng, max(N, 8))
    agent, partner, a, z = agent % N, partner % N, a % N, z % N
    # re-draw until distinct within N
    while len({agent, partner, a, z}) < 4:
        agent, partner, a, z = [x % N for x in rng.sample(range(N), 4)]
    credits = rng.randint(8, 14)
    if cell == "P1_T":
        ctx = ("What you know about your current partner:\n"
               f"- Note from P{a}: P{partner} chose keep against me in an earlier round.")
    elif cell == "P1_B" or cell.startswith("P2_base"):
        ctx = "You have no information about your current partner."
    elif cell == "P1_X":
        ctx = ("What you know:\n"
               f"- Note from P{a}: P{z} chose keep against me in an earlier round.\n"
               "You have no information about your current partner.")
    elif cell.startswith("P2_count"):
        ctx = ("What you know about your current partner:\n"
               f"- 3 participants have reported that P{partner} chose keep against them in earlier rounds.")
    elif cell.startswith("P2_prop"):
        pct = "37.5" if N == 8 else "0.3"
        ctx = ("What you know about your current partner:\n"
               f"- 3 of the {N} participants ({pct}%) have reported that P{partner} chose keep "
               "against them in earlier rounds.")
    else:
        raise ValueError(cell)
    return SKELETON.format(agent=agent, N=N, credits=credits, partner=partner, ctx=ctx)

def parse_action(txt):
    for mjson in re.findall(r'\{[^{}]*\}', txt):
        try:
            j = json.loads(mjson)
            if j.get("action") in ("give", "keep"):
                return j["action"]
        except json.JSONDecodeError:
            pass
    t = txt.lower()
    g = bool(re.search(r'\bgive\b', t)); k = bool(re.search(r'\bkeep\b', t))
    if g and not k: return "give"
    if k and not g: return "keep"
    return None

def call_cli(prompt, env, timeout=90):
    for attempt in range(2):
        try:
            r = subprocess.run(
                ["claude", "-p", prompt, "--model", "sonnet", "--output-format", "json"],
                capture_output=True, text=True, timeout=timeout, env=env,
                stdin=subprocess.DEVNULL)
            raw = r.stdout.strip()
            try:
                env_json = json.loads(raw)
                txt = env_json.get("result", raw)
                model = env_json.get("modelUsage") or env_json.get("model") or ""
            except json.JSONDecodeError:
                txt, model = raw, ""
            act = parse_action(txt or "")
            if act:
                return act, txt, str(model)[:200]
        except subprocess.TimeoutExpired:
            txt = "<timeout>"
        time.sleep(1)
    return None, txt if 'txt' in dir() else "", ""

def main():
    n = N_PER_CELL
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    dry = "--dry" in sys.argv
    cells = ["P1_T", "P1_B", "P1_X",
             "P2_count_8", "P2_count_1000", "P2_prop_8", "P2_prop_1000",
             "P2_base_8", "P2_base_1000"]
    rng = random.Random(42)
    jobs = []
    for cell in cells:
        for i in range(n):
            jobs.append((cell, i, build_prompt(cell, rng)))
    if dry:
        seen = set()
        for cell, i, p in jobs:
            if cell not in seen:
                seen.add(cell); print("=" * 20, cell, "=" * 20); print(p, "\n")
        return
    env = dict(os.environ)
    if not env.get("CLAUDE_CODE_OAUTH_TOKEN") and not env.get("ANTHROPIC_API_KEY"):
        if os.path.exists(TOKEN_FILE):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = open(TOKEN_FILE).read().strip()
        else:
            print("No auth: set CLAUDE_CODE_OAUTH_TOKEN or create .cli_token in project root.")
            sys.exit(1)
    raw_path = os.path.join(DATA, "probe_p1p2_raw.jsonl")
    done_keys = set()
    if os.path.exists(raw_path):  # resume support
        for line in open(raw_path):
            try:
                d = json.loads(line); done_keys.add((d["cell"], d["i"]))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1]) not in done_keys]
    print(f"{len(jobs)} calls to make (resume skipped {len(done_keys)}).", flush=True)
    lock_write = __import__("threading").Lock()
    fout = open(raw_path, "a")
    def work(job):
        cell, i, prompt = job
        act, txt, model = call_cli(prompt, env)
        with lock_write:
            fout.write(json.dumps({"cell": cell, "i": i, "action": act,
                                    "response": txt[:500], "model": model,
                                    "prompt": prompt}) + "\n")
            fout.flush()
        return cell, act
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for k, (cell, act) in enumerate(ex.map(work, jobs)):
            if (k + 1) % 20 == 0:
                print(f"{k+1}/{len(jobs)} elapsed {time.time()-t0:.0f}s", flush=True)
    fout.close()
    summarize()

def summarize():
    from collections import defaultdict
    raw_path = os.path.join(DATA, "probe_p1p2_raw.jsonl")
    tally = defaultdict(lambda: {"give": 0, "keep": 0, "invalid": 0})
    for line in open(raw_path):
        d = json.loads(line)
        c = tally[d["cell"]]
        if d["action"] in ("give", "keep"):
            c[d["action"]] += 1
        else:
            c["invalid"] += 1
    rates = {}
    for cell, c in tally.items():
        valid = c["give"] + c["keep"]
        rates[cell] = {"give_rate": round(c["give"] / valid, 3) if valid else None,
                       "n_valid": valid, "invalid": c["invalid"]}
    g = lambda cell: rates[cell]["give_rate"]
    out = {"experiment": "P1_P2_probes", "rates": rates}
    try:
        d_t = g("P1_B") - g("P1_T"); d_x = g("P1_B") - g("P1_X")
        out["P1"] = {"delta_targeted": round(d_t, 3), "delta_placebo": round(d_x, 3),
                     "P1_P1_pass_mechanism_activatable": bool(d_t >= 0.10),
                     "P1_P2_class": ("targeted" if d_t > 0 and d_x / d_t <= 0.3 else
                                      "generalized" if d_t > 0 and d_x / d_t >= 0.7 else
                                      "mixed" if d_t > 0 else "n/a")}
        dc8 = g("P2_base_8") - g("P2_count_8"); dc1k = g("P2_base_1000") - g("P2_count_1000")
        dp8 = g("P2_base_8") - g("P2_prop_8"); dp1k = g("P2_base_1000") - g("P2_prop_1000")
        out["P2"] = {"delta_count_N8": round(dc8, 3), "delta_count_N1000": round(dc1k, 3),
                     "delta_prop_N8": round(dp8, 3), "delta_prop_N1000": round(dp1k, 3),
                     "P2_P1_denominator_neglect": bool(abs(dc1k - dc8) < 0.10),
                     "P2_P2_salience_restores_stats": bool((dc1k - dp1k) >= 0.15)}
    except (TypeError, ZeroDivisionError):
        out["error"] = "missing cells or zero valid responses"
    p = os.path.join(DATA, "probe_p1p2_summary.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    if "--summarize" in sys.argv:
        summarize()
    else:
        main()
