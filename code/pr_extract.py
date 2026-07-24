#!/usr/bin/env python3
"""pr_extract.py — OpenReview 观察性研究取数脚本（只取数，不分析）。

冻结依据：《实验计划.md》§9 v2.5（2026-07-19）。
纪律：凭证仅经环境变量（OPENREVIEW_USER / OPENREVIEW_PASS）；≤2 req/s；
429 指数退避；JSONL 断点续跑；溯源日志；不计算任何内容性统计。

用法（在本地、认证可用的网络下运行）：
  export OPENREVIEW_USER=...; export OPENREVIEW_PASS=...
  python pr_extract.py --venue iclr2025 --stage submissions --smoke   # 烟测
  python pr_extract.py --venue iclr2025 --stage submissions
  python pr_extract.py --venue iclr2025 --stage reviews
  python pr_extract.py --venue iclr2025 --stage edits
  python pr_extract.py --venue iclr2025 --stage reconcile
  # venue ∈ {iclr2024, iclr2025, iclr2026, tmlr}；stage 按序跑。
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone

import openreview

BASE = "https://api2.openreview.net"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pr_extract")

VENUES = {
    "iclr2024": {"sub_inv": "ICLR.cc/2024/Conference/-/Submission",
                 "review_suffix": "/-/Official_Review"},
    "iclr2025": {"sub_inv": "ICLR.cc/2025/Conference/-/Submission",
                 "review_suffix": "/-/Official_Review"},
    "iclr2026": {"sub_inv": "ICLR.cc/2026/Conference/-/Submission",
                 "review_suffix": "/-/Official_Review"},
    "tmlr":     {"sub_inv": "TMLR/-/Submission",
                 "review_suffix": "/-/Review"},
}

# 对账锚（§9 v2.5 冻结；reconcile 阶段只比数量，不看内容）
ANCHORS = {
    "iclr2025": {"submissions_total": 11672, "non_desk_rejected": 11553,
                 "reviews_thakkar": 44831},
    "iclr2026": {"valid_submissions_official": 19525, "desk_rejected_official": 779,
                 "desk_rejected_api_prior": 853, "withdrawn_official": 5042},
    "tmlr": {"papers_approx": 4026},
}

RATE_SLEEP = float(os.environ.get("PR_RATE_SLEEP", "0.5"))  # 全局礼貌限速 ≤2 req/s（注册纪律）；并发多进程时用 PR_RATE_SLEEP 调大使总和 ≤2/s
MAX_RETRY = 6


def now():
    return datetime.now(timezone.utc).isoformat()


def paths(venue):
    d = os.path.join(DATA, venue)
    os.makedirs(d, exist_ok=True)
    return {
        "submissions": os.path.join(d, "submissions.jsonl"),
        "reviews": os.path.join(d, "reviews.jsonl"),
        "edits": os.path.join(d, "edits.jsonl"),
        "done_reviews": os.path.join(d, "done_forums.txt"),
        "done_edits": os.path.join(d, "done_review_ids.txt"),
        "provenance": os.path.join(d, "provenance.jsonl"),
        "status": os.path.join(d, "status_counts.json"),
    }


def log_prov(p, stage, params, n):
    with open(p["provenance"], "a") as f:
        f.write(json.dumps({"t": now(), "stage": stage, "params": params,
                            "n_results": n,
                            "client": getattr(openreview, "__version__", "unknown")},
                           ensure_ascii=False) + "\n")


def get_client():
    user, pw = os.environ.get("OPENREVIEW_USER"), os.environ.get("OPENREVIEW_PASS")
    if not user or not pw:
        sys.exit("请先设置 OPENREVIEW_USER / OPENREVIEW_PASS 环境变量（凭证不得写入文件）")
    return openreview.api.OpenReviewClient(baseurl=BASE, username=user, password=pw)


def with_retry(fn, desc):
    delay = 2.0
    for i in range(MAX_RETRY):
        try:
            return fn()
        except Exception as e:
            msg = str(e)[:200]
            if i == MAX_RETRY - 1:
                raise
            print(f"  retry {i+1}/{MAX_RETRY} after error on {desc}: {msg}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def note_json(n):
    """to_json() drops the trusted timestamps (tcdate/tmdate) and edit-level
    dates; pull them from object attributes explicitly (发现于 2026-07-20 对账，
    这些是冻结设计的关键字段——cdate/mdate 可被设置，不作真实时间依据)."""
    try:
        j = n.to_json()
    except Exception:
        j = json.loads(json.dumps(n, default=lambda o: getattr(o, "__dict__", str(o))))
    for attr in ("cdate", "tcdate", "mdate", "tmdate", "odate", "pdate"):
        v = getattr(n, attr, None)
        if v is not None and j.get(attr) is None:
            j[attr] = v
    return j


def classify_status(note):
    """七类状态：仅依据 venueid/venue 字符串，模式层判断。"""
    c = note.get("content", {}) if isinstance(note, dict) else {}
    vid = (c.get("venueid") or {}).get("value", "") if isinstance(c.get("venueid"), dict) else ""
    ven = (c.get("venue") or {}).get("value", "") if isinstance(c.get("venue"), dict) else ""
    s = (vid + " " + ven).lower()
    if "withdrawn" in s:
        return "withdrawn"
    if "desk" in s and "reject" in s:
        return "desk_rejected"
    if "reject" in s:
        return "rejected_public"
    if any(k in s for k in ("oral", "poster", "spotlight", "accept")) or s.strip() == "tmlr tmlr":
        return "accepted"
    if not s.strip():
        return "undetermined"
    return "active_or_other"


def stage_submissions(client, venue, p, smoke):
    cfg = VENUES[venue]
    print(f"[{now()}] submissions: invitation={cfg['sub_inv']}")
    notes = with_retry(lambda: client.get_all_notes(invitation=cfg["sub_inv"]),
                       "get_all_notes(submissions)")
    if smoke:
        notes = notes[:20]
    seen, counts = set(), {}
    with open(p["submissions"], "w") as f:
        for n in notes:
            j = note_json(n)
            if j["id"] in seen:
                continue
            seen.add(j["id"])
            counts[classify_status(j)] = counts.get(classify_status(j), 0) + 1
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    counts["unique_roots"] = len(seen)
    with open(p["status"], "w") as f:
        json.dump(counts, f, ensure_ascii=False, indent=1)
    log_prov(p, "submissions", {"invitation": cfg["sub_inv"], "smoke": smoke}, len(seen))
    print(f"  唯一根论坛 {len(seen)}；状态计数已写 {p['status']}")


def stage_reviews(client, venue, p, smoke):
    cfg = VENUES[venue]
    done = load_done(p["done_reviews"])
    forums = []
    with open(p["submissions"]) as f:
        for line in f:
            fid = json.loads(line)["id"]
            if fid not in done:
                forums.append(fid)
    if smoke:
        forums = forums[:1]
    print(f"[{now()}] reviews: 待取 {len(forums)} 个 forum（已完成 {len(done)}）")
    n_reviews = 0
    for i, fid in enumerate(forums):
        replies = with_retry(lambda: client.get_all_notes(forum=fid), f"forum {fid}")
        revs = [note_json(n) for n in replies
                if any(inv.endswith(cfg["review_suffix"])
                       for inv in (n.invitations or []))]
        with open(p["reviews"], "a") as f:
            for r in revs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(p["done_reviews"], "a") as f:
            f.write(fid + "\n")
        n_reviews += len(revs)
        if i % 200 == 0:
            print(f"  {i}/{len(forums)} forums, +{n_reviews} reviews", flush=True)
            log_prov(p, "reviews_progress", {"forum_index": i}, n_reviews)
        time.sleep(RATE_SLEEP)
    log_prov(p, "reviews", {"forums": len(forums), "smoke": smoke}, n_reviews)
    print(f"  完成：新增 {n_reviews} 条 review")


def stage_edits(client, venue, p, smoke):
    done = load_done(p["done_edits"])
    rids = []
    with open(p["reviews"]) as f:
        for line in f:
            rid = json.loads(line)["id"]
            if rid not in done:
                rids.append(rid)
    if smoke:
        rids = rids[:5]
    print(f"[{now()}] edits: 待取 {len(rids)} 条 review（已完成 {len(done)}）")
    n_edits = 0
    for i, rid in enumerate(rids):
        edits = with_retry(lambda: client.get_note_edits(note_id=rid), f"edits {rid}")
        with open(p["edits"], "a") as f:
            for e in edits:
                j = note_json(e)
                j["_review_id"] = rid
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        with open(p["done_edits"], "a") as f:
            f.write(rid + "\n")
        n_edits += len(edits)
        if i % 500 == 0:
            print(f"  {i}/{len(rids)} reviews, +{n_edits} edits", flush=True)
            log_prov(p, "edits_progress", {"review_index": i}, n_edits)
        time.sleep(RATE_SLEEP)
    log_prov(p, "edits", {"reviews": len(rids), "smoke": smoke}, n_edits)
    print(f"  完成：新增 {n_edits} 条 edit 记录")


def stage_reconcile(venue, p):
    """只比数量，不看内容。"""
    def count_lines(path):
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            return sum(1 for _ in f)
    out = {"venue": venue, "t": now(),
           "submissions_jsonl": count_lines(p["submissions"]),
           "reviews_jsonl": count_lines(p["reviews"]),
           "edit_records_jsonl": count_lines(p["edits"]),
           "forums_done": len(load_done(p["done_reviews"])),
           "reviews_edits_done": len(load_done(p["done_edits"])),
           "anchors": ANCHORS.get(venue, {})}
    if os.path.exists(p["status"]):
        out["status_counts"] = json.load(open(p["status"]))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(os.path.join(os.path.dirname(p["status"]), "reconcile.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=list(VENUES))
    ap.add_argument("--stage", required=True,
                    choices=["submissions", "reviews", "edits", "reconcile"])
    ap.add_argument("--smoke", action="store_true", help="烟测：≤20 submissions / 1 forum / 5 reviews")
    a = ap.parse_args()
    p = paths(a.venue)
    if a.stage == "reconcile":
        stage_reconcile(a.venue, p)
        return
    client = get_client()
    {"submissions": stage_submissions, "reviews": stage_reviews,
     "edits": stage_edits}[a.stage](client, a.venue, p, a.smoke)


if __name__ == "__main__":
    main()
