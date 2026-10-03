"""Held-out PersonaMem-v2 (text, MCQ mode) evaluation for the agent0mem textual memory track.

Data: HF bowen-upenn/PersonaMem-v2, benchmark/text/benchmark.csv (5000 questions over
200 personas, 7 pref_type values) plus data/chat_history_32k/<persona>.json, fetched
on demand into ~/aml_heldout/personamem/raw.

Official AML pipeline: /tmp/aml-repo/data/personamem/pipeline_v2.py. Copied verbatim and
checked at runtime: RECALL_SUFFIX, MCQ_PROMPT_TEMPLATE, option construction and
extract_final_letter. Scoring is the official exact mapped-answer match (no LLM judge).

Deviations from the official code (documented, unavoidable):
  - The official pipeline feeds the raw chat_history to the model. A memory system
    replaces it with retrieved memories, sent as a system message before the query.
  - The official option shuffle seeds random with Python's hash(), which is salted per
    process (PYTHONHASHSEED), so it is not reproducible. A sha256-based seed of the same
    string is used instead.
  - The 32k-token chat history variant is used (the 128k variant only adds unrelated
    math/coding padding) and its leading system message (the full persona profile) is
    not ingested, so preferences must come from the conversation itself.

Exclusions: personas with any question overlapping a logged smoke query
("pickup basketball", "weekend reading time") are dropped entirely; questions that
mention another smoke entity are dropped. Then --personas personas are drawn with a
fixed seed and ~60 questions stratified by pref_type.

Usage:
  AGENT0MEM_DATA_DIR=~/aml_heldout/personamem/db python3 scripts/heldout/personamem_eval.py select
  AGENT0MEM_DATA_DIR=~/aml_heldout/personamem/db python3 scripts/heldout/personamem_eval.py ingest
  AGENT0MEM_DATA_DIR=~/aml_heldout/personamem/db python3 scripts/heldout/personamem_eval.py run --variant clean_wide
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import heldout_common as hc
from heldout_common import le

DATASET = "personamem"
RAW_DIR = hc.HELDOUT_HOME / DATASET / "raw"
CSV_PATH = RAW_DIR / "benchmark.csv"
HF_BASE = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main/"
PIPELINE = hc.AML_REPO / "personamem" / "pipeline_v2.py"

RECALL_SUFFIX = (
    " Please recall my related preferences from our conversation history "
    "to give personalized responses."
)

MCQ_PROMPT_TEMPLATE = """Please choose the best answer from the following options:

{options}

Think step by step about which answer best fits the user's query and conversation context.
Provide your reasoning first, then give your final answer as 'Final Answer: [Letter]'"""

MEMORY_PREAMBLE = "Memories retrieved from previous conversations with this user:\n\n{memories}"


def user_query_text(item: dict[str, Any]) -> str:
    query = item.get("user_query", item.get("question", item.get("query")))
    if isinstance(query, str) and query.strip().startswith("{"):
        try:
            query = ast.literal_eval(query)
        except (ValueError, SyntaxError):
            pass
    if isinstance(query, dict):
        return str(query.get("content", query.get("text", "")))
    return str(query)


def official_mcq_options(item: dict[str, Any]) -> tuple[list[str], dict[str, str], str]:
    correct_answer = str(item["correct_answer"])
    incorrect_answers = item.get("incorrect_answers", [])
    if isinstance(incorrect_answers, str):
        try:
            incorrect_answers = json.loads(incorrect_answers) if incorrect_answers else []
        except json.JSONDecodeError:
            incorrect_answers = []
    incorrect_answers = list(incorrect_answers)
    if not incorrect_answers:
        raise TypeError("PersonaMem v2 MCQ mode requires `incorrect_answers`.")

    options = [correct_answer] + [str(answer) for answer in incorrect_answers]
    query_for_seed = user_query_text(item) + RECALL_SUFFIX
    # Official: seed = hash(f"{persona_id}_{query}") % 2**32 (salted per process); stable variant here.
    seed = int(hashlib.sha256(f"{item.get('persona_id', '')}_{query_for_seed}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(options)

    letters = [chr(65 + i) for i in range(len(options))]
    mapping = dict(zip(letters, options))
    correct_letter = next(letter for letter, option in mapping.items() if option == correct_answer)
    return options, mapping, correct_letter


def extract_final_letter(text: Any) -> str:
    raw = str(text)
    if not raw:
        return ""
    patterns = [
        r"\$\\boxed\{([A-Z])\}\$",
        r"\\boxed\{([A-Z])\}",
        r"Final Answer:\s*([A-Z])",
        r"final answer:\s*([A-Z])",
        r"Answer:\s*([A-Z])",
        r"answer:\s*([A-Z])",
        r"final answer is\s*\$?\\boxed\{([A-Z])\}\$?",
        r"final answer is\s*([A-Z])",
        r"the answer is\s*\$?\\boxed\{([A-Z])\}\$?",
        r"the answer is\s*([A-Z])",
        r"\b([A-Z])\.\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return ""


# ----------------------------------------------------------------------------- data

def load_csv() -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows):
        r["qid"] = f"p{r['persona_id']}_{i}"
    return rows


def stage_select(work: Path, total: int, seed: int, n_personas: int) -> None:
    rows = load_csv()
    excl_persona: dict[str, str] = {}
    for r in rows:
        hit = hc.smoke_query_hit(user_query_text(r))
        if hit:
            excl_persona[r["persona_id"]] = f"question overlaps smoke query: {hit[:70]}"
    excl_q: dict[str, str] = {}
    by_persona: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["persona_id"] in excl_persona:
            excl_q[r["qid"]] = "persona excluded"
            continue
        ent = hc.entity_hit(f"{user_query_text(r)}\n{r['correct_answer']}\n{r['preference']}")
        if ent:
            excl_q[r["qid"]] = f"entity: {ent}"
            continue
        by_persona[r["persona_id"]].append(r)
    rng = random.Random(seed)
    personas = sorted(by_persona, key=int)
    rng.shuffle(personas)
    chosen_personas = sorted(personas[:n_personas], key=int)
    pool: dict[str, list] = defaultdict(list)
    for p in chosen_personas:
        for r in by_persona[p]:
            pool[r["pref_type"]].append(r["qid"])
    chosen = hc.stratified(pool, total, seed)
    sel = {"dataset": DATASET, "seed": seed, "total_items": len(rows), "excluded_personas": excl_persona,
           "excluded": excl_q, "n_excluded": len(excl_q), "personas": chosen_personas, "selected": chosen,
           "pool_sizes": {k: len(v) for k, v in pool.items()}}
    hc.write_json(work / "selection.json", sel)
    print(f"select: {len(rows)} questions; excluded personas {excl_persona}; excluded questions {len(excl_q)} "
          f"({sum(1 for v in excl_q.values() if v.startswith('entity'))} by entity); personas {chosen_personas}; "
          f"selected {len(chosen)} pool={sel['pool_sizes']}")


def selected(work: Path) -> list[dict]:
    want = set(hc.read_json(work / "selection.json")["selected"])
    return [r for r in load_csv() if r["qid"] in want]


def history_path(link: str) -> Path:
    p = RAW_DIR / link
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(HF_BASE + link, timeout=300) as resp:
            p.write_bytes(resp.read())
    return p


def stage_ingest(work: Path, workers: int) -> None:
    links = {}
    for r in selected(work):
        links[r["persona_id"]] = r["chat_history_32k_link"]
    jobs = []
    base = 1_735_689_600_000  # 2025-01-01T00:00:00Z; histories carry no timestamps, keep order
    for pid, link in sorted(links.items(), key=lambda x: int(x[0])):
        d = json.loads(history_path(link).read_text(encoding="utf-8"))
        hist = d["chat_history"] if isinstance(d, dict) else d
        msgs = [{"role": m["role"], "content": m["content"], "timestamp": base + i * 60_000}
                for i, m in enumerate(hist) if m.get("role") != "system" and str(m.get("content", "")).strip()]
        for c, chunk in enumerate(hc.chunk_messages(msgs)):
            jobs.append((f"pm_{pid}_c{c}", f"pm_{pid}", f"pm_{pid}", chunk))
    hc.ingest(jobs, work / "ingested.jsonl", workers)


def stage_run(work: Path, variant: str, workers: int, answer_model: str, top_k: int) -> None:
    hc.verify_constants(PIPELINE, {"RECALL_SUFFIX": RECALL_SUFFIX, "MCQ_PROMPT_TEMPLATE": MCQ_PROMPT_TEMPLATE})
    hc.verify_source_snippet(PIPELINE, "def extract_final_letter", 'return ""\n\n', Path(__file__))
    hc.warm_imports(variant)
    out_path = work / f"results_{variant}.jsonl"
    done = {r["id"] for r in hc.jl_read(out_path)}
    todo = []
    for r in selected(work):
        if r["qid"] in done:
            continue
        query = user_query_text(r)
        todo.append({"id": r["qid"], "user_id": f"pm_{r['persona_id']}", "question": query,
                     "category": r["pref_type"], "preference": r["preference"], "raw": r})

    def one(q: dict) -> dict:
        r = q.pop("raw")
        items = le.retrieve(variant, {"user_id": q["user_id"], "question": q["question"]}, top_k)
        mem = le.render_items(items)
        options, mapping, correct_letter = official_mcq_options(r)
        option_text = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
        messages = [{"role": "system", "content": MEMORY_PREAMBLE.format(memories=mem)},
                    {"role": "user", "content": q["question"] + RECALL_SUFFIX},
                    {"role": "system", "content": MCQ_PROMPT_TEMPLATE.format(options=option_text)}]
        gen = hc.chat(messages, answer_model, max_tokens=1500)
        pred = extract_final_letter(gen)
        correct = mapping.get(pred) == mapping[correct_letter]
        pref_lc = q["preference"].lower()
        return {**q, "variant": variant, "n_items": len(items), "context_bytes": len(mem.encode()),
                "context": mem, "generated": gen, "predicted_letter": pred, "gold_letter": correct_letter,
                "gold_answer": mapping[correct_letter], "correct": correct,
                "judge": f"predicted {pred or '(none)'} vs gold {correct_letter}",
                "preference_in_context": pref_lc in mem.lower()}

    hc.run_parallel(todo, one, out_path, workers)
    stage_report(work, variant)


def stage_report(work: Path, variant: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.report(rs, f"held-out PersonaMem-v2 MCQ (32k), variant={variant}", out_json=work / f"summary_{variant}.json")
    if rs:
        nl = sum(1 for r in rs if not r["predicted_letter"])
        print(f"  unparseable final letter: {nl}; chance level = 25%")


def stage_taxonomy(work: Path, variant: str, model: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.failure_taxonomy(rs, lambda r: not r["correct"],
                        lambda r: f"User preference to remember: {r['preference']}\nCorrect option: {r['gold_answer']}",
                        work / f"taxonomy_{variant}.jsonl", model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "ingest", "run", "report", "taxonomy"])
    ap.add_argument("--work", default=str(hc.HELDOUT_HOME / DATASET / "work"))
    ap.add_argument("--variant", default="clean_wide")
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--personas", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--answer-model", default="qwen-plus")
    ap.add_argument("--judge-model", default="qwen-plus", help="only used by the taxonomy stage")
    a = ap.parse_args()
    hc.load_env()
    hc.require_isolated_db(DATASET)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "select":
        stage_select(work, a.total, a.seed, a.personas)
    elif a.stage == "ingest":
        stage_ingest(work, a.workers)
    elif a.stage == "run":
        stage_run(work, a.variant, a.workers, a.answer_model, a.top_k)
    elif a.stage == "report":
        stage_report(work, a.variant)
    else:
        stage_taxonomy(work, a.variant, a.judge_model)


if __name__ == "__main__":
    main()
