"""Held-out LongMemEval-S evaluation for the agent0mem textual memory track.

Data: xiaowu0162/longmemeval-cleaned, longmemeval_s_cleaned.json (500 questions, each
with its own ~48-session haystack), downloaded to ~/aml_heldout/longmemeval/raw.

Official AML pipeline (/tmp/aml-repo/data/longmemeval-s/pipeline.py) uses exactly the
same answer template and judge prompt as LoCoMo, so they are imported from
locomo_eval.py and checked verbatim against the official file at runtime.

Exclusions: any question whose text or gold answer overlaps a logged smoke query or a
known smoke entity, plus any question that shares an evidence session with an excluded
question.

Stages (resumable, JSONL under --work):
  select : exclusion + stratified sample by question_type (fixed seed) -> selection.json
  ingest : every haystack session of every selected question -> its own user namespace,
           one POST /add-equivalent request per session
  run    : retrieval variant -> official answer template -> official judge
  report : per-category accuracy with 95% Wilson CI
  taxonomy : LLM-assisted classification of 15 sampled wrong answers

Adapter choices (not specified by the official pipeline, documented here):
  - speaker_1_name="user", all memories go in speaker_1_memories, speaker_2 empty.
  - The question is passed verbatim; question_date is NOT given to the answer model
    unless --question-date prefix is used (temporal questions are harder without it).
  - Answer max_tokens 512 (official sends no limit).

Usage:
  AGENT0MEM_DATA_DIR=~/aml_heldout/longmemeval/db python3 scripts/heldout/longmemeval_eval.py select
  AGENT0MEM_DATA_DIR=~/aml_heldout/longmemeval/db python3 scripts/heldout/longmemeval_eval.py ingest
  AGENT0MEM_DATA_DIR=~/aml_heldout/longmemeval/db python3 scripts/heldout/longmemeval_eval.py run --variant clean_wide
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import heldout_common as hc
from heldout_common import le

DATASET = "longmemeval"
RAW = hc.HELDOUT_HOME / DATASET / "raw" / "longmemeval_s_cleaned.json"
PIPELINE = hc.AML_REPO / "longmemeval-s" / "pipeline.py"


def parse_date(s: str) -> int:
    dt = datetime.strptime(s.strip(), "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def load() -> list[dict]:
    return json.loads(RAW.read_text(encoding="utf-8"))


def stage_select(work: Path, total: int, seed: int) -> None:
    data = load()
    excl: dict[str, str] = {}
    for x in data:
        text = f"{x['question']}\n{x['answer']}"
        hit = hc.smoke_query_hit(x["question"])
        if hit:
            excl[x["question_id"]] = f"smoke_query: {hit[:60]}"
            continue
        ent = hc.entity_hit(text)
        if ent:
            excl[x["question_id"]] = f"entity: {ent}"
    bad_sessions = {sid for x in data if x["question_id"] in excl for sid in x["answer_session_ids"]}
    for x in data:
        if x["question_id"] not in excl and bad_sessions & set(x["answer_session_ids"]):
            excl[x["question_id"]] = "shares evidence session with an excluded question"
    pool: dict[str, list] = defaultdict(list)
    for x in data:
        if x["question_id"] not in excl:
            pool[x["question_type"]].append(x["question_id"])
    chosen = hc.stratified(pool, total, seed)
    sel = {"dataset": DATASET, "seed": seed, "total_items": len(data), "excluded": excl,
           "n_excluded": len(excl), "selected": chosen,
           "pool_sizes": {k: len(v) for k, v in pool.items()}}
    hc.write_json(work / "selection.json", sel)
    by = defaultdict(int)
    qt = {x["question_id"]: x["question_type"] for x in data}
    for q in chosen:
        by[qt[q]] += 1
    print(f"select: {len(data)} items, excluded {len(excl)}, selected {len(chosen)} {dict(by)}")
    for k, v in list(excl.items())[:20]:
        print(f"  excluded {k}: {v}")


def selected(work: Path) -> list[dict]:
    sel = hc.read_json(work / "selection.json")
    ids = set(sel["selected"])
    return [x for x in load() if x["question_id"] in ids]


def stage_ingest(work: Path, workers: int) -> None:
    jobs = []
    for x in selected(work):
        uid = f"lme_{x['question_id']}"
        for i, (sid, date, sess) in enumerate(zip(x["haystack_session_ids"], x["haystack_dates"],
                                                  x["haystack_sessions"])):
            base = parse_date(date)
            msgs = [{"role": t["role"], "content": t["content"], "timestamp": base + k * 30_000}
                    for k, t in enumerate(sess) if str(t.get("content", "")).strip()]
            if msgs:
                jobs.append((f"lme_{x['question_id']}_s{i}", uid, sid, msgs))
    hc.ingest(jobs, work / "ingested.jsonl", workers)


def stage_run(work: Path, variant: str, workers: int, answer_model: str, judge_model: str, top_k: int,
              question_date: str) -> None:
    hc.verify_constants(PIPELINE, {"OPEN_ENDED_ANSWER_TEMPLATE": le.OPEN_ENDED_ANSWER_TEMPLATE,
                                   "ACCURACY_PROMPT": le.ACCURACY_PROMPT})
    hc.warm_imports(variant)
    out_path = work / f"results_{variant}.jsonl"
    done = {r["id"] for r in hc.jl_read(out_path)}
    todo = []
    for x in selected(work):
        if x["question_id"] in done:
            continue
        q = x["question"]
        if question_date == "prefix":
            q = f"(Today is {x['question_date']}.) {q}"
        todo.append({"id": x["question_id"], "user_id": f"lme_{x['question_id']}", "question": q,
                     "gold_answer": str(x["answer"]), "category": x["question_type"],
                     "abstention": x["question_id"].endswith("_abs")})

    def one(q: dict) -> dict:
        items = le.retrieve(variant, q, top_k)
        mem = le.render_items(items)
        values = {"speaker_1_name": "user", "speaker_1_memories": mem, "speaker_2_name": "assistant",
                  "speaker_2_memories": "", "question": q["question"]}
        prompt = re.sub(r"\{\{(speaker_1_name|speaker_1_memories|speaker_2_name|speaker_2_memories|question)\}\}",
                        lambda m: values[m.group(1)], le.OPEN_ENDED_ANSWER_TEMPLATE)
        gen = le.llm(prompt, answer_model, max_tokens=512)
        jprompt = re.sub(r"\{(question|gold_answer|generated_answer)\}",
                         lambda m: {"question": q["question"], "gold_answer": q["gold_answer"],
                                    "generated_answer": gen}[m.group(1)], le.ACCURACY_PROMPT)
        jresp = le.llm(jprompt, judge_model, max_tokens=300)
        label, perr = "WRONG", False
        m = re.search(r"\{.*?\}", jresp, re.DOTALL)
        try:
            label = str(json.loads(m.group(0)).get("label", "WRONG")).upper() if m else "WRONG"
            perr = m is None
        except Exception:  # noqa: BLE001
            perr = True
        return {**q, "variant": variant, "n_items": len(items), "context_bytes": len(mem.encode()),
                "gold_in_context": q["gold_answer"].lower() in mem.lower(), "context": mem,
                "generated": gen, "label": label, "correct": label == "CORRECT", "judge": jresp,
                "judge_parse_error": perr}

    hc.run_parallel(todo, one, out_path, workers)
    stage_report(work, variant)


def stage_report(work: Path, variant: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.report(rs, f"held-out LongMemEval-S, variant={variant}", out_json=work / f"summary_{variant}.json")
    if rs:
        ab = [r for r in rs if r.get("abstention")]
        if ab:
            k = sum(r["correct"] for r in ab)
            lo, hi = le.wilson(k, len(ab))
            print(f"  (abstention subset {k}/{len(ab)} = {k/len(ab):.1%} [{lo:.1%}, {hi:.1%}])")
        gic = sum(r["gold_in_context"] for r in rs) / len(rs)
        print(f"  gold string in context (lexical, lower bound on recall): {gic:.1%}")


def stage_taxonomy(work: Path, variant: str, model: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.failure_taxonomy(rs, lambda r: not r["correct"], lambda r: r["gold_answer"],
                        work / f"taxonomy_{variant}.jsonl", model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "ingest", "run", "report", "taxonomy"])
    ap.add_argument("--work", default=str(hc.HELDOUT_HOME / DATASET / "work"))
    ap.add_argument("--variant", default="clean_wide")
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--answer-model", default="qwen-plus")
    ap.add_argument("--judge-model", default="qwen-plus")
    ap.add_argument("--question-date", choices=["none", "prefix"], default="none")
    a = ap.parse_args()
    hc.load_env()
    hc.require_isolated_db(DATASET)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "select":
        stage_select(work, a.total, a.seed)
    elif a.stage == "ingest":
        stage_ingest(work, a.workers)
    elif a.stage == "run":
        stage_run(work, a.variant, a.workers, a.answer_model, a.judge_model, a.top_k, a.question_date)
    elif a.stage == "report":
        stage_report(work, a.variant)
    else:
        stage_taxonomy(work, a.variant, a.judge_model)


if __name__ == "__main__":
    main()
