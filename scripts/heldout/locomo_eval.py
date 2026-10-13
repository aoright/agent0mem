"""Held-out textual evaluation on LoCoMo conversations never used in any AML smoke run.

Why this exists: the previous local evaluator asked the exact smoke questions that
the engine had hardcoded answers for, so it always reported 100%. This harness
uses only conversations that never appeared in a smoke run (conv-26 Caroline,
conv-30 Jon, conv-44 Audrey are excluded) and scores with the official AML
LoCoMo answer template and judge prompt (copied verbatim from
/tmp/aml-repo/data/locomo-refined/pipeline.py).

Stages (each is resumable and writes JSONL under --work):
  ingest   : push every session through the same code path as POST /add into an
             isolated local SQLite DB (never production).
  retrieve : run a retrieval variant for each sampled question.
  answer   : official answer template, answer model at temperature 0.
  judge    : official accuracy prompt, judge model at temperature 0.
  report   : accuracy per category with a 95% Wilson interval.

Usage:
  AGENT0MEM_DATA_DIR=~/aml_heldout/db python3 scripts/heldout/locomo_eval.py ingest
  AGENT0MEM_DATA_DIR=~/aml_heldout/db python3 scripts/heldout/locomo_eval.py run --variant legacy
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = Path(os.path.expanduser("~/aml_heldout/locomo10.json"))
SMOKE_SEEN = {"conv-26", "conv-30", "conv-44"}  # appeared in official smoke runs
CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}

OPEN_ENDED_ANSWER_TEMPLATE = """You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Use only the provided memories. Prefer the memory that answers the question most directly.
2. Your memories are episodic raw observations. Reason about what they imply. Do not refuse just because the answer is not stated verbatim.
3. The question may contain typos. Match it to the most relevant memory even if the wording differs.
4. When multiple answers are possible, list all supported answers, not just the first.
5. For counts or time intervals, enumerate carefully before answering.
6. Preserve specific names, titles, places, and labels from the memories. Use "Rob" not "a colleague", "Sweden" not "home country".
7. Convert relative times like "yesterday", "last month", and "last year" into dates, months, or years when the memory timestamp makes it clear. Keep week-based expressions relative.
8. If memories conflict, prefer the most recent supported memory.
9. For list questions, include all required items and no extras.
10. Keep the final answer minimal. Do not add explanation, background, or extra dates unless needed for correctness.
</instructions>

<memories>
Memories for user {{speaker_1_name}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_name}}:

{{speaker_2_memories}}
</memories>

Question: {{question}}
Answer with the shortest correct phrase or sentence. No preamble, no fluff:"""

ACCURACY_PROMPT = """Your task is to label an answer as \u2019CORRECT\u2019 or \u2019WRONG\u2019 given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle \u2014 Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold\u2019s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT \u2014 even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold\u2019s content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR\u2194HOUR, DAY\u2194DAY, MONTH\u2194MONTH, YEAR\u2194YEAR.
  Do not answer a gold at a different time unit \u2014 even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] \u2192 WRONG)
- Do NOT convert relative \u2194 absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., \u201cthe/last/previous/just prior\u201d) as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C \u2192 CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D \u2192 WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C \u2192 C, C\u2032), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```"""


# ----------------------------------------------------------------------------- utils

def jl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def jl_append(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_session_time(s: str) -> int:
    dt = datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def llm(prompt: str, model: str, max_tokens: int = 256, retries: int = 4) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    base = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(base + "/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"llm failed: {last}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


# ----------------------------------------------------------------------------- data

def heldout_conversations() -> list[dict]:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    return [c for c in data if c["sample_id"] not in SMOKE_SEEN]


def sample_questions(per_cat: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[int, list[dict]] = defaultdict(list)
    for conv in heldout_conversations():
        c = conv["conversation"]
        for i, qa in enumerate(conv["qa"]):
            if qa.get("category") not in CATEGORY_NAMES or "answer" not in qa:
                continue
            by_cat[qa["category"]].append({
                "id": f"{conv['sample_id']}#{i}",
                "user_id": f"heldout_{conv['sample_id']}",
                "speaker_a": c["speaker_a"], "speaker_b": c["speaker_b"],
                "question": qa["question"], "gold_answer": str(qa["answer"]),
                "category": qa["category"],
            })
    out = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        rng.shuffle(pool)
        out.extend(pool[:per_cat])
    return out


# ----------------------------------------------------------------------------- stages

def stage_ingest(work: Path, workers: int) -> None:
    from app.database import init_db, save_memories_batch
    from app.core.extractor import extract_propositions

    init_db()
    done_path = work / "ingested.jsonl"
    done = {r["request_id"] for r in jl_read(done_path)}
    jobs = []
    for conv in heldout_conversations():
        c = conv["conversation"]
        a = c["speaker_a"]
        n = 1
        while f"session_{n}" in c:
            base_ms = parse_session_time(c[f"session_{n}_date_time"])
            msgs = []
            for t_idx, turn in enumerate(c[f"session_{n}"]):
                text = turn.get("text", "")
                if turn.get("blip_caption"):
                    text = f"{text} [shares a photo of {turn['blip_caption']}]".strip()
                msgs.append({"role": "user" if turn["speaker"] == a else "assistant",
                             "content": f"{turn['speaker']}: {text}",
                             "timestamp": base_ms + t_idx * 30_000})
            rid = f"heldout_{conv['sample_id']}_s{n}"
            if rid not in done:
                jobs.append((rid, f"heldout_{conv['sample_id']}", f"{conv['sample_id']}_s{n}", msgs))
            n += 1

    print(f"ingest: {len(jobs)} sessions pending ({len(done)} already done)")

    def run(job):
        rid, uid, sid, msgs = job
        t0 = time.time()
        props = extract_propositions(msgs)  # identical to POST /add
        save_memories_batch(request_id=rid, user_id=uid, session_id=sid, raw_messages=msgs, propositions=props)
        return {"request_id": rid, "n_msgs": len(msgs), "n_props": len(props),
                "fallback_raw": bool(props) and all(re.match(r"^\[\d{4}-\d{2}-\d{2}", p or "") for p in props),
                "sec": round(time.time() - t0, 2)}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, rec in enumerate(ex.map(run, jobs), 1):
            jl_append(done_path, rec)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}")
    recs = jl_read(done_path)
    fb = sum(1 for r in recs if r.get("fallback_raw"))
    print(f"ingest done: {len(recs)} sessions, extraction fell back to raw dialogue in {fb}")


def render_items(items: list[dict]) -> str:
    # The platform passes each returned item; memory_text() json-dumps dicts.
    return "\n".join(json.dumps({"content": it.get("content", ""), "created_at": it.get("created_at")},
                                ensure_ascii=False) for it in items)


def retrieve(variant: str, q: dict, top_k: int) -> list[dict]:
    if variant == "legacy":
        # Line-for-line replica of app/main.py search_memory() for a string query,
        # without importing FastAPI: search_hybrid(top_k=min(k,25)) -> cap 15 ->
        # unwrap_proposition / strip_temporal_markers on content and text.
        from app.database import search_hybrid, unwrap_proposition
        from app.core.temporal import strip_temporal_markers
        cands = search_hybrid(user_id=q["user_id"], query_text=q["question"], options=None, top_k=min(top_k, 25))
        cands = cands[:15]
        out = []
        for c in cands:
            content = unwrap_proposition(strip_temporal_markers(unwrap_proposition(c["content"])))
            out.append({"id": c["id"], "content": content, "score": c.get("score", 1.0),
                        "created_at": c.get("created_at")})
        return out
    if variant in ("clean", "clean_sprint"):
        from app.core.clean_search import clean_search
        return clean_search(q["user_id"], q["question"], top_k=top_k)
    if variant in ("clean_wide", "clean_sprint_wide"):
        from app.core.clean_search import clean_search
        return clean_search(q["user_id"], q["question"], top_k=top_k, max_items=100, byte_budget=48_000)
    if variant in ("wide_human", "wide_chrono", "wide_human_chrono"):
        from app.core.clean_search import clean_search
        return clean_search(q["user_id"], q["question"], top_k=top_k, max_items=100, byte_budget=48_000,
                            order="time" if "chrono" in variant else "score",
                            date_style="human" if "human" in variant else "iso")
    if variant == "full_context":
        # Upper-bound reference, not a retrieval method: every raw turn, chronological.
        from app.database import get_db
        from app.core.clean_search import _clean, _day
        with get_db() as conn:
            rows = conn.execute("SELECT memory_id, content, timestamp, created_at FROM memories "
                                "WHERE user_id = ? AND item_type = 'raw' ORDER BY timestamp ASC, id ASC",
                                (q["user_id"],)).fetchall()
        return [{"id": r["memory_id"], "content": f"[{_day(r['timestamp'], r['created_at'])}] {_clean(r['content'])}",
                 "created_at": r["created_at"]} for r in rows]
    if variant == "wide_notes":
        from app.core.clean_search import clean_search
        from app.core.query_notes import with_notes
        return with_notes(q["question"], clean_search(q["user_id"], q["question"], top_k=top_k,
                                                      max_items=100, byte_budget=48_000))
    raise SystemExit(f"unknown variant {variant}")


def stage_run(work: Path, variant: str, per_cat: int, seed: int, workers: int,
              answer_model: str, judge_model: str, top_k: int) -> None:
    # Import retrieval code once in the main thread; concurrent first imports from
    # worker threads can observe a partially initialised module.
    if variant == "legacy":
        import app.database  # noqa: F401
    else:
        import app.core.clean_search  # noqa: F401
    qs = sample_questions(per_cat, seed)
    out_path = work / f"results_{variant}.jsonl"
    done = {r["id"] for r in jl_read(out_path)}
    todo = [q for q in qs if q["id"] not in done]
    print(f"run[{variant}]: {len(todo)} of {len(qs)} questions pending")

    def one(q):
        items = retrieve(variant, q, top_k)
        mem = render_items(items)
        prompt = re.sub(r"\{\{(speaker_1_name|speaker_1_memories|speaker_2_name|speaker_2_memories|question)\}\}",
                        lambda m: {"speaker_1_name": q["speaker_a"], "speaker_1_memories": mem,
                                   "speaker_2_name": q["speaker_b"], "speaker_2_memories": "",
                                   "question": q["question"]}[m.group(1)],
                        OPEN_ENDED_ANSWER_TEMPLATE)
        gen = llm(prompt, answer_model, max_tokens=200)
        jprompt = re.sub(r"\{(question|gold_answer|generated_answer)\}",
                         lambda m: {"question": q["question"], "gold_answer": q["gold_answer"],
                                    "generated_answer": gen}[m.group(1)], ACCURACY_PROMPT)
        jresp = llm(jprompt, judge_model, max_tokens=200)
        m = re.search(r"\{.*?\}", jresp, re.DOTALL)
        label = "WRONG"
        if m:
            try:
                label = str(json.loads(m.group(0)).get("label", "WRONG")).upper()
            except Exception:  # noqa: BLE001
                label = "CORRECT" if "CORRECT" in jresp and "WRONG" not in jresp else "WRONG"
        gold_lc = q["gold_answer"].lower()
        return {**q, "variant": variant, "n_items": len(items), "context_bytes": len(mem.encode()),
                "gold_in_context": gold_lc in mem.lower(), "generated": gen,
                "label": label, "correct": label == "CORRECT", "judge": jresp}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, q) for q in todo]
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                jl_append(out_path, f.result())
            except Exception as e:  # noqa: BLE001
                print("  error:", e)
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")
    stage_report(work, variant)


def stage_report(work: Path, variant: str) -> None:
    rs = jl_read(work / f"results_{variant}.jsonl")
    if not rs:
        print("no results")
        return
    print(f"\n=== held-out LoCoMo, variant={variant}, n={len(rs)} ===")
    by = defaultdict(list)
    for r in rs:
        by[r["category"]].append(r)
    for cat in sorted(by):
        g = by[cat]
        k = sum(r["correct"] for r in g)
        lo, hi = wilson(k, len(g))
        items = sum(r["n_items"] for r in g) / len(g)
        gic = sum(r["gold_in_context"] for r in g) / len(g)
        print(f"  {CATEGORY_NAMES[cat]:12s} {k:3d}/{len(g):3d} = {k/len(g):6.1%}  95%CI [{lo:5.1%}, {hi:5.1%}]"
              f"  avg_items={items:5.1f}  gold_string_in_context={gic:5.1%}")
    k = sum(r["correct"] for r in rs)
    lo, hi = wilson(k, len(rs))
    print(f"  {'OVERALL':12s} {k:3d}/{len(rs):3d} = {k/len(rs):6.1%}  95%CI [{lo:5.1%}, {hi:5.1%}]"
          f"  avg_context_bytes={sum(r['context_bytes'] for r in rs)/len(rs):,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ingest", "run", "report"])
    ap.add_argument("--work", default=os.path.expanduser("~/aml_heldout/work"))
    ap.add_argument("--variant", default="legacy")
    ap.add_argument("--per-cat", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--answer-model", default="qwen-plus")
    ap.add_argument("--judge-model", default="qwen-plus")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if "AGENT0MEM_DATA_DIR" not in os.environ:
        raise SystemExit("set AGENT0MEM_DATA_DIR to an isolated directory (never production)")
    if a.stage == "ingest":
        stage_ingest(work, a.workers)
    elif a.stage == "run":
        stage_run(work, a.variant, a.per_cat, a.seed, a.workers, a.answer_model, a.judge_model, a.top_k)
    else:
        stage_report(work, a.variant)


if __name__ == "__main__":
    main()
