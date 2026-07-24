"""pr_analysis.py — 同行评议观察性研究，冻结口径分析（实验计划.md §9 v2.5）。
仅开发半（holdout_split_2025.json dev 领域）；验证半留待最终口径一次运行。
实现常量（本文件首次运行前固定，未经结果调参）：
  CREATION_EPS_MS = 120000   # edit.tcdate 距 note.tcdate <2min 视为创建记录，不计修订
  REVIEW_DDL  = 2024-11-04 12:00 UTC (AoE 11-03 末)      = 1730721600000
  FEEDBACK_T0 = 2024-10-15 00:00 UTC = 1728950400000
  FEEDBACK_T1 = 2024-11-13 12:00 UTC = 1731499200000
  DISCUSS_T1  = 2024-11-28 12:00 UTC = 1732795200000
描述性纪律：2025 年一切修订统计含 ~50% agent 诱导成分（Thakkar），不作因果解读。
用法：python3 pr_analysis.py --dev-2025 | --tmlr-control | --all
"""
import json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PR = os.path.join(HERE, "..", "data", "pr_extract")

CREATION_EPS_MS = 120000
REVIEW_DDL = 1730721600000
FEEDBACK_T0 = 1728950400000
FEEDBACK_T1 = 1731499200000
DISCUSS_T1 = 1732708800000  # 2024-11-27 12:00 UTC（冻结讨论期 11-13..26 AoE 末；2026-07-22 审计更正，原值 1732795200000 多一天）
DAY = 86400000

def area_of(sub):
    c = sub.get("content") or {}
    pa = c.get("primary_area")
    return pa.get("value") if isinstance(pa, dict) else pa

def load_2025_dev(half="dev"):
    split = json.load(open(os.path.join(PR, "holdout_split_2025.json")))
    dev = set(split[half])
    forum_area = {}
    for l in open(os.path.join(PR, "iclr2025", "submissions.jsonl")):
        d = json.loads(l)
        a = area_of(d)
        if a in dev:
            forum_area[d["id"]] = a
    return dev, forum_area

def dev_2025(half="dev"):
    dev, forum_area = load_2025_dev(half)
    # reviews in dev forums
    rev = {}
    for l in open(os.path.join(PR, "iclr2025", "reviews.jsonl")):
        d = json.loads(l)
        if d.get("forum") in forum_area:
            rev[d["id"]] = {"tcdate": d["tcdate"], "forum": d["forum"],
                            "area": forum_area[d["forum"]],
                            "sig": (d.get("signatures") or [""])[0]}
    # reviewer edits per frozen definition
    rev_edits = defaultdict(list)   # review_id -> [edit tcdate]
    sys_edits = 0
    for l in open(os.path.join(PR, "iclr2025", "edits.jsonl")):
        e = json.loads(l)
        rid = e.get("_review_id")
        if rid not in rev:
            continue
        sig = (e.get("signatures") or [""])[0]
        et = e.get("tcdate")
        if et is None:
            continue
        # 冻结定义从严实现：须为该 review 本人签名（2026-07-22 收紧；全量验证子串判定与本人签名判定 0 差异）
        if "Reviewer_" in sig and sig == rev[rid]["sig"]:
            if et > rev[rid]["tcdate"] + CREATION_EPS_MS:
                rev_edits[rid].append(et)
        elif "Reviewer_" not in sig:
            sys_edits += 1
    n_rev = len(rev)
    out = {"n_dev_areas": len(dev), "n_dev_forums": len(forum_area),
           "n_dev_reviews": n_rev, "system_edit_records_excluded": sys_edits,
           "caveat": "2025 revision stats include ~50% agent-induced component (Thakkar); descriptive only"}
    # M1 revision rate
    revised = {r for r in rev_edits if rev_edits[r]}
    out["M1_revision_rate"] = round(len(revised) / n_rev, 4)
    # M2 first-revision delay (days)
    delays = sorted((min(rev_edits[r]) - rev[r]["tcdate"]) / DAY for r in revised)
    if delays:
        q = lambda p: round(delays[int(p * (len(delays) - 1))], 2)
        out["M2_first_revision_delay_days"] = {"n": len(delays), "median": q(.5),
                                               "q25": q(.25), "q75": q(.75)}
    # M3 revision timing vs institutional windows
    win = Counter()
    for r in revised:
        for ts in rev_edits[r]:
            if ts <= REVIEW_DDL: win["before_review_ddl"] += 1
            elif ts <= FEEDBACK_T1: win["feedback_window_post_ddl"] += 1
            elif ts <= DISCUSS_T1: win["discussion_period"] += 1
            else: win["after_discussion"] += 1
    out["M3_reviewer_edit_timing"] = dict(win)
    # S1 deadline clustering of review submission (tcdate)
    tot = pre = h72 = h24 = late = 0
    for r in rev.values():
        t = r["tcdate"]; tot += 1
        if t <= REVIEW_DDL:
            pre += 1
            if t > REVIEW_DDL - 72 * 3600000: h72 += 1
            if t > REVIEW_DDL - 24 * 3600000: h24 += 1
        else:
            late += 1
    out["S1_clustering"] = {"n": tot, "share_last72h_of_predeadline": round(h72 / pre, 4),
                            "share_last24h_of_predeadline": round(h24 / pre, 4),
                            "late_share": round(late / tot, 4)}
    # S2 per-area clustering vs demand pressure (dev areas)
    area_stats = {}
    for a in sorted({r["area"] for r in rev.values()}):
        rs = [r for r in rev.values() if r["area"] == a]
        pre_a = [r for r in rs if r["tcdate"] <= REVIEW_DDL]
        c72 = sum(1 for r in pre_a if r["tcdate"] > REVIEW_DDL - 72 * 3600000)
        area_stats[a] = {"n_reviews": len(rs),
                         "n_submissions": sum(1 for f, aa in forum_area.items() if aa == a),
                         "clust72": round(c72 / len(pre_a), 4) if pre_a else None}
    out["S2_area_table"] = area_stats
    xs = [(v["n_submissions"], v["clust72"]) for v in area_stats.values() if v["clust72"] is not None]
    # Spearman（scipy，正确处理并列；2026-07-22 审计更正——手写版无 tie 处理）
    from scipy import stats as _st
    rho, pval = _st.spearmanr([x for x, _ in xs], [y for _, y in xs])
    out["S2_spearman_subs_vs_clust72"] = {"rho": round(float(rho), 3),
                                          "p": round(float(pval), 3), "n_areas": len(xs)}
    # D1/D2
    total_dev_subs = len(forum_area)
    out["D_demand"] = {a: {"D1_submissions": v["n_submissions"],
                           "D2_share_of_dev": round(v["n_submissions"] / total_dev_subs, 4)}
                       for a, v in area_stats.items()}
    return out

def tmlr_control():
    ts = []
    for l in open(os.path.join(PR, "tmlr", "reviews.jsonl")):
        d = json.loads(l)
        if d.get("tcdate"): ts.append(d["tcdate"])
    ts.sort()
    # max share of reviews in any 7-day window (global synchronization metric)
    import bisect
    best = 0
    for t in ts[:: max(1, len(ts) // 2000)]:
        j = bisect.bisect_right(ts, t + 7 * DAY)
        i = bisect.bisect_left(ts, t)
        best = max(best, (j - i) / len(ts))
    return {"n_reviews": len(ts), "max_7day_window_share": round(best, 4),
            "span_days": round((ts[-1] - ts[0]) / DAY, 0)}

def icl2025_window_share():
    ts = []
    for l in open(os.path.join(PR, "iclr2025", "reviews.jsonl")):
        d = json.loads(l)
        if d.get("tcdate"): ts.append(d["tcdate"])
    ts.sort()
    import bisect
    best = 0
    for t in ts[:: max(1, len(ts) // 2000)]:
        j = bisect.bisect_right(ts, t + 7 * DAY)
        i = bisect.bisect_left(ts, t)
        best = max(best, (j - i) / len(ts))
    return {"n_reviews": len(ts), "max_7day_window_share": round(best, 4)}

DDL_2024 = 1698840000000   # 2023-11-01 12:00 UTC (评审窗口止 10-31 AoE；Guide 另有"due Oct 30"表述，歧义入档)
REL_2024 = 1699574400000   # 2023-11-10 00:00 UTC 发布（2026-07-22 审计更正，原值为 11-11）
DISC_END_2024 = 1700740800000  # 2023-11-23 12:00 UTC
DDL_2026 = 1762084800000   # 2025-11-02 12:00 UTC (评审窗口止 11-01 AoE)
W_CLEAN_END = 1762819200000    # 2025-11-11 00:00 UTC（干净窗口 ≤11-10）
W_FUZZY_END = 1764288000000    # 2025-11-28 00:00 UTC（模糊窗口 11-11..11-27）

def _words(content):
    n = 0
    for v in (content or {}).values():
        s = v.get("value") if isinstance(v, dict) else v
        if isinstance(s, str): n += len(s.split())
    return n

def m4_dev_2025():
    dev, forum_area = load_2025_dev()
    base = {}
    for l in open(os.path.join(PR, "iclr2025", "reviews.jsonl")):
        d = json.loads(l)
        if d.get("forum") in forum_area:
            base[d["id"]] = {"tcdate": d["tcdate"], "content": d.get("content") or {}}
    ed = defaultdict(list)
    for l in open(os.path.join(PR, "iclr2025", "edits.jsonl")):
        e = json.loads(l)
        rid = e.get("_review_id")
        if rid in base and "Reviewer_" in (e.get("signatures") or [""])[0] \
           and e.get("tcdate") and e["tcdate"] > base[rid]["tcdate"] + CREATION_EPS_MS:
            c = (e.get("note") or {}).get("content")
            if c: ed[rid].append((e["tcdate"], c))
    # 实现注记（2026-07-22）：公开流不含发布前原始版本（创建 edit 不公开、reviews.jsonl
    # content 为最终态），故"相对原始版的修订幅度"公开不可测——M4 收窄为相邻公开
    # edit 间的词数变化（仅 ≥2 次审稿人 edit 的评审可测），范围如实注记。
    deltas = []
    multi = 0
    for rid, lst in ed.items():
        lst = sorted(lst)
        if len(lst) < 2: continue
        multi += 1
        for (t0, c0), (t1, c1) in zip(lst, lst[1:]):
            common = set(c0) | set(c1)
            old = _words({k: c0.get(k) for k in common})
            new = _words({k: c1.get(k, c0.get(k)) for k in common})
            deltas.append(new - old)
    deltas.sort()
    n = len(deltas)
    if not n:
        return {"n_multi_edit_reviews": multi, "n_inter_edit_deltas": 0}
    q = lambda p: deltas[int(p * (n - 1))]
    return {"n_multi_edit_reviews": multi, "n_inter_edit_deltas": n,
            "scope_note": "inter-edit only; magnitude vs pre-publication original unmeasurable publicly",
            "signed_delta_words": {"median": q(.5), "q25": q(.25), "q75": q(.75)},
            "share_growing": round(sum(1 for d in deltas if d > 0) / n, 4),
            "mean_abs_delta": round(sum(abs(d) for d in deltas) / n, 1)}

def year_2024():
    rev = {}
    for l in open(os.path.join(PR, "iclr2024", "reviews.jsonl")):
        d = json.loads(l)
        rev[d["id"]] = d["tcdate"]
    rev_edits = defaultdict(list)
    for l in open(os.path.join(PR, "iclr2024", "edits.jsonl")):
        e = json.loads(l)
        rid = e.get("_review_id")
        if rid in rev and "Reviewer_" in (e.get("signatures") or [""])[0] \
           and e.get("tcdate") and e["tcdate"] > rev[rid] + CREATION_EPS_MS:
            rev_edits[rid].append(e["tcdate"])
    n = len(rev)
    revised = {r for r in rev_edits if rev_edits[r]}
    delays = sorted((min(rev_edits[r]) - rev[r]) / DAY for r in revised)
    q = lambda p: round(delays[int(p * (len(delays) - 1))], 2)
    win = Counter()
    for r in revised:
        for ts in rev_edits[r]:
            if ts <= DDL_2024: win["before_ddl"] += 1
            elif ts <= REL_2024: win["ddl_to_release"] += 1
            elif ts <= DISC_END_2024: win["discussion"] += 1
            else: win["after"] += 1
    tot = pre = h72 = h24 = late = 0
    for t in rev.values():
        tot += 1
        if t <= DDL_2024:
            pre += 1
            if t > DDL_2024 - 72 * 3600000: h72 += 1
            if t > DDL_2024 - 24 * 3600000: h24 += 1
        else: late += 1
    return {"n_reviews": n, "M1": round(len(revised) / n, 4),
            "M2_median_days": q(.5),
            "M3": dict(win),
            "S1": {"share_last72h": round(h72 / pre, 4), "share_last24h": round(h24 / pre, 4),
                   "late_share": round(late / tot, 4)},
            "note": "2024 无反馈 agent；讨论期 11-10..11-22"}

def case_2026():
    rev = {}
    for l in open(os.path.join(PR, "iclr2026", "reviews.jsonl")):
        d = json.loads(l)
        rev[d["id"]] = d["tcdate"]
    win_rev = Counter(); win_sys = Counter(); daily_sys = Counter(); daily_rev = Counter()
    for l in open(os.path.join(PR, "iclr2026", "edits.jsonl")):
        e = json.loads(l)
        rid = e.get("_review_id"); et = e.get("tcdate")
        if rid not in rev or not et: continue
        sig = (e.get("signatures") or [""])[0]
        is_rev = "Reviewer_" in sig and et > rev[rid] + CREATION_EPS_MS
        w = ("clean" if et < W_CLEAN_END else "fuzzy" if et < W_FUZZY_END else "post")
        day = int((et - W_CLEAN_END) // DAY)
        if is_rev:
            win_rev[w] += 1
            if -10 <= day <= 40: daily_rev[day] += 1
        elif "Reviewer_" not in sig:
            win_sys[w] += 1
            if -10 <= day <= 40: daily_sys[day] += 1
    tot = pre = h72 = h24 = 0
    for t in rev.values():
        tot += 1
        if t <= DDL_2026:
            pre += 1
            if t > DDL_2026 - 72 * 3600000: h72 += 1
            if t > DDL_2026 - 24 * 3600000: h24 += 1
    return {"n_reviews": tot,
            "S1": {"share_last72h": round(h72 / pre, 4), "share_last24h": round(h24 / pre, 4),
                   "late_share": round((tot - pre) / tot, 4)},
            "reviewer_edits_by_window": dict(win_rev),
            "system_edits_by_window": dict(win_sys),
            "daily_reviewer_edits_day0_is_1111": {str(k): v for k, v in sorted(daily_rev.items())},
            "daily_system_edits_day0_is_1111": {str(k): v for k, v in sorted(daily_sys.items())}}

def main():
    out = {}
    if "--dev-2025" in sys.argv or "--all" in sys.argv:
        out["dev_2025"] = dev_2025("dev")
    if "--val-2025" in sys.argv:   # 最终口径一次运行（封存裁决 A1/A2/A3-val）
        v = dev_2025("val")
        rho = v["S2_spearman_subs_vs_clust72"]["rho"]
        v["sealed_adjudication"] = {
            "A1_S1_last72h_ge_0.27": v["S1_clustering"]["share_last72h_of_predeadline"] >= 0.27,
            "A2_S2_rho_positive": rho > 0,
            "A3_M1_in_band_0.22_0.32": 0.22 <= v["M1_revision_rate"] <= 0.32}
        out["val_2025_FINAL"] = v
    if "--tmlr-control" in sys.argv or "--all" in sys.argv:
        out["tmlr"] = tmlr_control()
        out["iclr2025_all_for_window_contrast"] = icl2025_window_share()
    if "--m4" in sys.argv or "--all" in sys.argv:
        out["M4_dev_2025"] = m4_dev_2025()
    if "--y2024" in sys.argv or "--all" in sys.argv:
        out["year_2024"] = year_2024()
    if "--case2026" in sys.argv or "--all" in sys.argv:
        out["case_2026"] = case_2026()
    path = os.path.join(PR, "pr_analysis_val.json" if "--val-2025" in sys.argv
                        else "pr_analysis_dev.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
