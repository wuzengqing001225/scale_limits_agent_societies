"""baseline_glm.py -- Baseline-adjusted report x log N interaction models.

Logistic model over all lattice decisions including the k=0 no-information
baselines:

    keep ~ 1 + R + R*log(k) + log(N) + R*log(N)

where R = 1[k >= 1] indicates report presence, R*log(k) carries the dose
among report cells (0 at k=0), logs are natural, y is keep=1. The
coefficient of interest is the interaction R x log(N): the change in
stated-size dependence attributable to the presence of reports, over and
above any drift of the uninformed baseline with stated size. Wald 95%
intervals. Firth (Jeffreys-penalised) fits are used where the plain GLM
separates (div1b_sonnet, p6c_gemini), matching the protocol used for the
registered per-lattice GLMs in div_glm.py.

Outputs ../data/baseline_glm_results.json with full coefficient vectors.
"""
import json, os
import numpy as np
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

NAMES = ["intercept", "R", "R_logk", "logN", "R_logN"]

DATASETS = [
    # key, file, format, engine label, firth
    ("p6_main", "p6_main_raw.jsonl", "count", "claude-sonnet (primary)", False),
    ("p6c_reasoner", "p6c_reasoner_raw.jsonl", "count", "deepseek-reasoner", False),
    ("p6c_gemini", "p6c_gemini_raw.jsonl", "count", "gemini-3.5-flash", True),
    ("p6c_gpt55", "p6c_gpt55_raw.jsonl", "count", "gpt-5.5", False),
    ("div1b_reasoner", "div1b_reasoner_raw.jsonl", "percentage", "deepseek-reasoner", False),
    ("div1b_sonnet", "div1b_sonnet_raw.jsonl", "percentage", "claude-sonnet", True),
]


def rows(path):
    best = {}
    for l in open(path):
        d = json.loads(l)
        if d.get("_header"):
            continue
        best[(d["k"], d["N"], d["i"])] = d
    return [(d["k"], d["N"], 1.0 if d["action"] == "keep" else 0.0)
            for d in best.values() if d["action"] in ("give", "keep")]


def design(rs):
    X, y = [], []
    for k, N, keep in rs:
        R = 1.0 if k >= 1 else 0.0
        lk = np.log(k) if k >= 1 else 0.0
        X.append([1.0, R, R * lk, np.log(N), R * np.log(N)])
        y.append(keep)
    return np.array(X), np.array(y)


def glm_fit(X, y):
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return np.asarray(m.params), np.asarray(m.bse), "GLM"


def firth_fit(X, y):
    beta = np.zeros(X.shape[1])
    Iinv = None
    for _ in range(400):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = p * (1 - p)
        I = X.T @ (X * W[:, None])
        Iinv = np.linalg.inv(I)
        Xs = X * np.sqrt(W)[:, None]
        H = Xs @ Iinv @ Xs.T
        U = X.T @ (y - p + np.diag(H) * (0.5 - p))
        step = Iinv @ U
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta, np.sqrt(np.diag(Iinv)), "Firth"


def main():
    out = {}
    for key, fname, fmt, engine, firth in DATASETS:
        X, y = design(rows(os.path.join(DATA, fname)))
        beta, se, method = (firth_fit if firth else glm_fit)(X, y)
        coefs = {}
        for j, nm in enumerate(NAMES):
            b, s = float(beta[j]), float(se[j])
            coefs[nm] = {"b": round(b, 3), "se": round(s, 3),
                         "CI95": [round(b - 1.96 * s, 3), round(b + 1.96 * s, 3)]}
        out[key] = {"format": fmt, "engine": engine, "method": method,
                    "n": int(len(y)), "coefficients": coefs,
                    "interaction_R_logN": coefs["R_logN"]}
    path = os.path.join(DATA, "baseline_glm_results.json")
    json.dump(out, open(path, "w"), indent=1)
    for k, v in out.items():
        c = v["interaction_R_logN"]
        print(f"{k:16s} {v['format']:10s} {v['method']:5s} "
              f"b={c['b']:+.2f} CI={c['CI95']}")


if __name__ == "__main__":
    main()
