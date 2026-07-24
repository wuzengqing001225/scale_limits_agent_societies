"""P4 (paraphrase robustness) + P5 (cross-model) probes. Registered in 实验计划.md §7
BEFORE running; paraphrase texts below are FROZEN at registration.
Auth: uses the locally logged-in CLI directly (NO token file needed).
Usage (local Mac):
    python3 probe_p4_paraphrase.py                                   # P4: claude sonnet, variants 1-5
    python3 probe_p4_paraphrase.py --model opus --variants 1         # P5: claude opus
    python3 probe_p4_paraphrase.py --engine codex --model gpt-4o --variants 1   # P5: GPT-4o via Codex CLI
    python3 probe_p4_paraphrase.py --engine codex --model gpt-4o --selftest     # 1 call, prints raw output
Resume-safe; raw JSONL + summary JSON in ../data/, tagged by engine_model.
"""
import subprocess, json, re, os, sys, random, time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TOKEN_FILE = os.path.join(HERE, "..", "..", ".cli_token")
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

# ---- FROZEN paraphrase variants (V1 = original wording used in P1/P2) ----
NOTE_T = {  # targeted negative note about subject S, author A
 1: 'Note from P{A}: P{S} chose keep against me in an earlier round.',
 2: 'Message from P{A}: "P{S} kept the credit instead of transferring when paired with me."',
 3: 'You received word from P{A} that P{S} decided not to transfer in a previous pairing.',
 4: 'P{A} reports: in an earlier round, P{S} took the keep option against them.',
 5: 'A note passed to you says P{S} withheld the transfer from P{A} last time they were paired.'}
NO_INFO = {
 1: 'You have no information about your current partner.',
 2: 'Nothing is known to you about this partner.',
 3: 'You have not received any notes about your current partner.',
 4: 'No information about P{P} is available to you.',
 5: 'You know nothing about the participant you are paired with.'}
COUNT = {
 1: '3 participants have reported that P{P} chose keep against them in earlier rounds.',
 2: 'Three separate participants say P{P} kept rather than transferred when paired with them.',
 3: 'You have received reports from 3 participants that P{P} did not transfer in earlier rounds.',
 4: 'P{P} has been reported by 3 participants for choosing keep against them.',
 5: 'Notes from 3 different participants indicate P{P} withheld transfers in previous rounds.'}
PROP8 = {
 1: '3 of the 8 participants (37.5%) have reported that P{P} chose keep against them in earlier rounds.',
 2: 'Out of the 8 participants in this study, 3 (37.5%) say P{P} kept rather than transferred with them.',
 3: "Reports about P{P} choosing keep have come from 3 of the study's 8 participants - 37.5%.",
 4: '37.5% of participants (3 of 8) have reported P{P} for not transferring in earlier rounds.',
 5: 'A 37.5% share of the 8 participants (3 people) report that P{P} withheld transfers.'}
PROP1000 = {  # P5b amendment cell (frozen 2026-07-12, before P5b data)
 1: '3 of the 1000 participants (0.3%) have reported that P{P} chose keep against them in earlier rounds.',
 2: 'Out of the 1000 participants in this study, 3 (0.3%) say P{P} kept rather than transferred with them.',
 3: "Reports about P{P} choosing keep have come from 3 of the study's 1000 participants - 0.3%.",
 4: '0.3% of participants (3 of 1000) have reported P{P} for not transferring in earlier rounds.',
 5: 'A 0.3% share of the 1000 participants (3 people) report that P{P} withheld transfers.'}

CELLS = ["P1_T", "P1_B", "P1_X", "P2_count_1000", "P2_prop_8"]
# "P2_prop_1000" is the P5b add-on cell: enabled via --extra-prop1000, reported
# rate-only (classificatory, no pass/fail bound; both readings preregistered in plan §7/P5b).

def build_prompt(cell, v, rng):
    N = 32 if cell.startswith("P1") else int(cell.rsplit("_", 1)[1])
    ids = rng.sample(range(max(N, 8)), 4)
    agent, partner, a, z = [x % N for x in ids]
    while len({agent, partner, a, z}) < 4:
        agent, partner, a, z = [x % N for x in rng.sample(range(max(N, 8)), 4)]
    credits = rng.randint(8, 14)
    hdr = "What you know about your current partner:\n- "
    if cell == "P1_T":
        ctx = hdr + NOTE_T[v].format(A=a, S=partner)
    elif cell == "P1_B":
        ctx = NO_INFO[v].format(P=partner)
    elif cell == "P1_X":
        ctx = ("What you know:\n- " + NOTE_T[v].format(A=a, S=z) + "\n" +
               NO_INFO[v].format(P=partner))
    elif cell == "P2_count_1000":
        ctx = hdr + COUNT[v].format(P=partner)
    elif cell == "P2_prop_8":
        ctx = hdr + PROP8[v].format(P=partner)
    elif cell == "P2_prop_1000":
        ctx = hdr + PROP1000[v].format(P=partner)
    return SKELETON.format(agent=agent, N=N, credits=credits, partner=partner, ctx=ctx)

def parse_action(txt):
    # scan matches in REVERSE: if a CLI echoes the prompt, the prompt's literal
    # {"action":"give"} / {"action":"keep"} pair appears early; the model's real
    # answer comes last. If the tail is exactly the echoed option pair, no answer.
    acts = []
    for m in re.findall(r'\{[^{}]*\}', txt or ""):
        try:
            j = json.loads(m)
            if j.get("action") in ("give", "keep"): acts.append(j["action"])
        except json.JSONDecodeError: pass
    if acts:
        if len(acts) >= 2 and acts[-2:] == ["give", "keep"]:
            return None  # echoed option pair only
        return acts[-1]
    t = (txt or "").lower()
    g, k = bool(re.search(r'\bgive\b', t)), bool(re.search(r'\bkeep\b', t))
    return "give" if (g and not k) else ("keep" if (k and not g) else None)

def _ansi(s): return re.sub(r'\x1b\[[0-9;]*m', '', s or "")

def _dbg(attempts, cmd, rc, out, err):
    attempts.append({
        "cmd": " ".join(cmd[:5]) + " …", "rc": rc,
        "stdout_head": (out or "")[:300], "stdout_tail": (out or "")[-300:],
        "stderr_tail": (err or "")[-900:]})

def call_cli(prompt, model, env, engine="claude"):
    import tempfile
    txt = ""
    attempts = []
    for _ in range(2):
        if engine == "claude":
            cmds = [["claude", "-p", prompt, "--model", model, "--output-format", "json"]]
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
            tmp.close()
            cmds = [["codex", "exec", "--skip-git-repo-check", "-m", model,
                     "--output-last-message", tmp.name, prompt],
                    ["codex", "exec", "--skip-git-repo-check", "-m", model, prompt]]
        for cmd in cmds:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                                   env=env, stdin=subprocess.DEVNULL)
                raw, err = _ansi((r.stdout or "").strip()), _ansi((r.stderr or "").strip())
                _dbg(attempts, cmd, r.returncode, raw, err)
                if engine == "claude":
                    try: txt = json.loads(raw).get("result", raw)
                    except json.JSONDecodeError: txt = raw
                else:
                    txt = ""
                    if "--output-last-message" in cmd and os.path.exists(tmp.name):
                        try: txt = open(tmp.name).read().strip()
                        except OSError: txt = ""
                    if not txt:
                        txt = raw
                act = parse_action(txt)
                if not act and engine == "codex":  # last resort: answer amid stderr log
                    act = parse_action(err)
                    if act: txt = err
                if act:
                    if engine == "codex":
                        try: os.unlink(tmp.name)
                        except OSError: pass
                    return act, txt, attempts
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                txt = f"<{type(e).__name__}>"
                _dbg(attempts, cmd, None, "", txt)
        if engine == "codex":
            try: os.unlink(tmp.name)
            except OSError: pass
        time.sleep(2)
    return None, txt, attempts

def main():
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "sonnet"
    engine = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else "claude"
    conc = int(sys.argv[sys.argv.index("--conc") + 1]) if "--conc" in sys.argv else CONC
    variants = ([int(x) for x in sys.argv[sys.argv.index("--variants") + 1].split(",")]
                if "--variants" in sys.argv else [1, 2, 3, 4, 5])
    env = dict(os.environ)  # use locally logged-in CLI auth as-is (no token injection)
    if "--selftest" in sys.argv:
        rng = random.Random(1)
        cell = "P2_prop_1000" if "--extra-prop1000" in sys.argv else "P1_B"
        p = build_prompt(cell, 1, rng)
        act, txt, attempts = call_cli(p, model, env, engine)
        print("cell:", cell, "parsed action:", act,
              "\n--- final text (first 800 chars) ---\n", (txt or "")[:800])
        print("\n--- per-template attempts (cmd / returncode / stdout / stderr) ---")
        for a in attempts:
            print(json.dumps(a, ensure_ascii=False, indent=1))
        return
    tag = f"p4_{engine}_{model}"
    raw_path = os.path.join(DATA, f"{tag}_raw.jsonl")
    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            try:
                d = json.loads(line)
                if d.get("action") in ("give", "keep"):  # failed rows get retried
                    done.add((d["cell"], d["variant"], d["i"]))
            except Exception: pass
    rng = random.Random(2026)
    cells = CELLS + (["P2_prop_1000"] if "--extra-prop1000" in sys.argv else [])
    jobs = [(c, v, i, build_prompt(c, v, rng)) for c in cells for v in [1, 2, 3, 4, 5]
            for i in range(N_PER)]           # build ALL for stable rng, filter after
    jobs = [j for j in jobs if j[1] in variants and (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} calls (model={model}).", flush=True)
    lock = Lock(); fout = open(raw_path, "a")
    def work(j):
        cell, v, i, prompt = j
        act, txt, attempts = call_cli(prompt, model, env, engine)
        row = {"cell": cell, "variant": v, "i": i, "action": act,
               "model": model, "engine": engine, "response": (txt or "")[:300],
               "prompt": prompt}
        if act is None and attempts:  # per-row failure diagnostics
            row["fail_rc"] = attempts[-1]["rc"]
            row["fail_stderr_tail"] = attempts[-1]["stderr_tail"][-400:]
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
        return 1
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for k, _ in enumerate(ex.map(work, jobs)):
            if (k + 1) % 25 == 0: print(f"{k+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    fout.close()
    # summary + registered verdicts (dedupe by (cell,variant,i), keep last = retried rows win)
    from collections import defaultdict
    best = {}
    for line in open(raw_path):
        d = json.loads(line)
        best[(d["cell"], d["variant"], d["i"])] = d
    tal = defaultdict(lambda: [0, 0])
    inv_by_cell = defaultdict(int)
    for d in best.values():
        if d["action"] in ("give", "keep"):
            tal[(d["cell"], d["variant"])][0] += d["action"] == "give"
            tal[(d["cell"], d["variant"])][1] += 1
        else:
            inv_by_cell[f"{d['cell']}_V{d['variant']}"] += 1
    rates = {f"{c}_V{v}": round(g / n, 3) for (c, v), (g, n) in sorted(tal.items()) if n}
    BOUND = {"P1_T": ("le", 0.2), "P1_B": ("ge", 0.8), "P1_X": ("ge", 0.8),
             "P2_count_1000": ("le", 0.2), "P2_prop_8": ("ge", 0.8)}
    verdict = {}
    for c in CELLS:
        oks = []
        for v in [1, 2, 3, 4, 5]:
            g, n = tal.get((c, v), (0, 0))
            if n == 0: continue
            r = g / n
            op, b = BOUND[c]
            oks.append(r <= b if op == "le" else r >= b)
        verdict[c] = {"variants_pass": sum(oks), "variants_run": len(oks),
                      "P4_pass_ge4of5": (sum(oks) >= 4) if len(oks) == 5 else None}
    out = {"experiment": "P4_P5_paraphrase_crossmodel", "model": model, "engine": engine,
           "rates": rates, "invalid_by_cell": dict(inv_by_cell), "verdicts": verdict}
    p = os.path.join(DATA, f"{tag}_summary.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
