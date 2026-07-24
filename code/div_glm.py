"""div_glm.py — DIV-1b 与 P6c 的注册 GLM/Firth 复算（代码入库义务，2026-07-22 补齐）。
输出 ../data/div_glm_results.json。y 编码：keep=1（与注册文本 "GLM keep ~ ..." 一致）。
Firth = Jeffreys 罚 Newton（分离时按协议使用）；对比检验 H0: b1 = -b2。
"""
import json, os
import numpy as np
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

def load_xy(path):
    best = {}
    for l in open(path):
        d = json.loads(l)
        best[(d["k"], d["N"], d["i"])] = d
    X, y = [], []
    tal = {}
    for d in best.values():
        if d["action"] in ("give", "keep"):
            g, n = tal.get((d["k"], d["N"]), (0, 0))
            tal[(d["k"], d["N"])] = (g + (d["action"] == "give"), n + 1)
            if d["k"] >= 1:
                X.append([1.0, np.log(d["k"]), np.log(d["N"])])
                y.append(1.0 if d["action"] == "keep" else 0.0)
    return np.array(X), np.array(y), tal

def glm(X, y):
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    c = np.array([0, 1, 1.0])
    est = c @ m.params; se = float(np.sqrt(c @ m.cov_params() @ c))
    return {"b_logk": round(float(m.params[1]), 3), "se_logk": round(float(m.bse[1]), 3),
            "b_logN": round(float(m.params[2]), 3), "se_logN": round(float(m.bse[2]), 3),
            "contrast_b1_plus_b2": round(float(est), 3), "contrast_se": round(se, 3)}

def firth(X, y):
    beta = np.zeros(X.shape[1])
    for _ in range(200):
        p = 1 / (1 + np.exp(-(X @ beta))); W = p * (1 - p)
        I = X.T @ (X * W[:, None]); Iinv = np.linalg.inv(I)
        H = (X * np.sqrt(W)[:, None]) @ Iinv @ (X * np.sqrt(W)[:, None]).T
        U = X.T @ (y - p + np.diag(H) * (0.5 - p))
        step = Iinv @ U; beta += step
        if np.max(np.abs(step)) < 1e-8: break
    cov = Iinv
    c = np.array([0, 1, 1.0])
    return {"b_logk": round(float(beta[1]), 3), "se_logk": round(float(np.sqrt(cov[1, 1])), 3),
            "b_logN": round(float(beta[2]), 3), "se_logN": round(float(np.sqrt(cov[2, 2])), 3),
            "contrast_b1_plus_b2": round(float(c @ beta), 3),
            "contrast_se": round(float(np.sqrt(c @ cov @ c)), 3), "method": "Firth"}

def anti(tal):
    cells = [(1, 40), (5, 200), (25, 1000)]
    vals = [round(tal[c][0] / tal[c][1], 3) for c in cells if c in tal and tal[c][1]]
    return {"cells_2.5pct": vals, "spread": round(max(vals) - min(vals), 3) if vals else None}

def main():
    out = {}
    X, y, tal = load_xy(os.path.join(DATA, "div1b_reasoner_raw.jsonl"))
    out["div1b_reasoner"] = {**glm(X, y), **anti(tal)}
    X, y, tal = load_xy(os.path.join(DATA, "div1b_sonnet_raw.jsonl"))
    out["div1b_sonnet"] = {**firth(X, y), **anti(tal)}   # 完全分离 → Firth（协议）
    for eng in ["reasoner", "gemini", "gpt55"]:
        X, y, tal = load_xy(os.path.join(DATA, f"p6c_{eng}_raw.jsonl"))
        g = glm(X, y)
        lo = g["b_logN"] - 1.96 * g["se_logN"]; hi = g["b_logN"] + 1.96 * g["se_logN"]
        g["CI_logN"] = [round(lo, 3), round(hi, 3)]
        g["sealed"] = {"i_dose_sig": abs(g["b_logk"] / g["se_logk"]) > 1.96 and g["b_logk"] > 0,
                       "ii_CI_logN_contains_0": lo < 0 < hi,
                       "iii_antidiag_spread_gt_0.15": anti(tal)["spread"] > 0.15}
        out[f"p6c_{eng}"] = {**g, **anti(tal)}
    path = os.path.join(DATA, "div_glm_results.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
