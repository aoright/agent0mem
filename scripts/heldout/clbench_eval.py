"""Held-out CL-Bench evaluation for the agent0mem textual memory track.

Data: HF tencent/CL-bench, CL-bench.jsonl (1899 tasks; each task = system prompt +
1..6 user/assistant turns, the last user turn carrying a long reference document and
the question; graded by task-specific rubrics), downloaded to ~/aml_heldout/clbench/raw.

Official AML pipeline: /tmp/aml-repo/data/clbench/pipeline.py. Copied verbatim and
checked at runtime: _CLBENCH_ANSWER_PROMPT_TEMPLATE, render_answer_prompt_clbench,
format_selected_memories, the rubric judge prompt (strict all-or-nothing, binary
"Overall Score") and its parsing / retry logic.

Model substitution: official judge default is Qwen/Qwen3-14B (SiliconFlow); here
qwen-plus at temperature 0 for answering and judging.

Exclusions: any task whose full text overlaps a logged smoke query (NTSB excerpt,
"document-grounded question answering agent" system prompt, "CONVERSATION TRANSCRIPT"
BRO transcript, OTC MIL-53, ...) or mentions a known smoke entity anywhere. Tasks
longer than --max-chars (default 100k characters) are also dropped to bound cost;
both counts are reported separately.

Ingestion adapter (the official pipeline does not specify it): every non-system
message, including the final user turn with its reference document, is sent through
POST /add in batches of <= 6 messages / 12k characters; the query passed to retrieval
and to the answer template is the final user message verbatim; the system prompt is
passed in {{system_prompt}}. Each task has its own user namespace.

Score: binary rubric_clbench_score with 95% Wilson CI per context_category.

Usage:
  AGENT0MEM_DATA_DIR=~/aml_heldout/clbench/db python3 scripts/heldout/clbench_eval.py select
  AGENT0MEM_DATA_DIR=~/aml_heldout/clbench/db python3 scripts/heldout/clbench_eval.py ingest
  AGENT0MEM_DATA_DIR=~/aml_heldout/clbench/db python3 scripts/heldout/clbench_eval.py run --variant clean_wide
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import heldout_common as hc
from heldout_common import le

DATASET = "clbench"
RAW = hc.HELDOUT_HOME / DATASET / "raw" / "CL-bench.jsonl"
PIPELINE = hc.AML_REPO / "clbench" / "pipeline.py"


_CLBENCH_ANSWER_PROMPT_TEMPLATE = """\\
{{system_prompt}}

<context>
The following memories from previous conversations may provide additional context:

{{memories}}
</context>

<task>
Answer the question below thoroughly and completely. The question includes a reference document \u2014 read it carefully and base your answer on its content.

Requirements:
- Cover every aspect asked in the question.
- Follow all formatting and style rules defined above (e.g. use or avoid bullet points as instructed, use bold headers if required, etc.).
- Do not truncate your response. A complete, detailed answer is expected.
</task>

{{question}}"""


def render_answer_prompt_clbench(*, system_prompt: str, memories: str, question: str) -> str:
    """Exact rendering behaviour from the supplied ``memory_search.py``."""
    return (
        _CLBENCH_ANSWER_PROMPT_TEMPLATE.replace("{{system_prompt}}", system_prompt or "")
        .replace("{{memories}}", memories or "(no memories)")
        .replace("{{question}}", question)
    ).strip()


def format_selected_memories(selected: list[dict[str, Any]]) -> str:
    """The timestamped prompt block used by the supplied retrieval implementation."""
    lines: list[str] = []
    for item in selected:
        timestamp = str(item.get("created_at") or "").strip()
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"- [{timestamp}] {text}" if timestamp else f"- {text}")
    return "\n".join(lines)


def _normalize_model_output(text: str) -> str:
    cleaned = str(text or "").strip()
    if "Final Answer:" in cleaned:
        cleaned = cleaned.split("Final Answer:", 1)[1].strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned


def _normalize_rubrics(rubrics: Sequence[str] | Sequence[dict[str, Any]] | None) -> list[str]:
    normalized: list[str] = []
    for item in rubrics or []:
        text = str(item.get("rubric_criteria", "") if isinstance(item, dict) else item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _build_rubrics_text(rubrics: Sequence[str]) -> str:
    if not rubrics:
        return "No specific rubrics provided."
    return "\n".join(f"{index}. {rubric}" for index, rubric in enumerate(rubrics, start=1))


def _coerce_score(payload: dict[str, Any]) -> int:
    try:
        return 1 if int(payload.get("Overall Score", "")) == 1 else 0
    except (TypeError, ValueError):
        return 0


def _coerce_status_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [part.strip() for part in text.split(",") if part.strip()]
    return []


def _compute_requirement_ratio(status_list: Sequence[str]) -> float:
    if not status_list:
        return 0.0
    return sum(str(item).strip().lower() in {"yes", "y", "true", "1"} for item in status_list) / len(status_list)


def _failed_payload(reason: str) -> dict[str, Any]:
    return {
        "rubric_clbench_score": 0.0,
        "rubric_clbench_rationale": reason,
        "rubric_clbench_requirement_status": [],
        "rubric_clbench_requirement_ratio": 0.0,
    }


def rubric_judge_prompt(*, rubrics_text: str, model_output: str) -> str:
    """The supplied ``rubric_clbench.py`` judge prompt, including its protocol."""
    return f'''Starting now, you are a rigorous instruction-following grading teacher. Your task is to accurately grade and score student answers based on the 【Rubrics】.

Grading Criteria
This is a strict, all-or-nothing grading system. The final score is binary.
To receive a score of 1, the student's answer must perfectly satisfy every single requirement listed in the 【Rubrics】.
If even one requirement is not fully met, the final score will be 0.
Grading Process
Please strictly follow the steps below for analysis—no steps may be skipped:
Step 1: Analyze the Standard Answer
List all explicit requirements in the 【Rubrics】 item by item (including format, content, quantity, order, etc.).
Identify implicit requirements in the 【Rubrics】 (e.g., language style, logical structure).
Define specific evaluation criteria for each requirement (e.g., "must include X," "must not exceed Y").
Step 2: Check Each Requirement Against the Student's Answer
For every requirement in the 【Rubrics】, verify one by one whether the student's answer fully satisfies it.
Step 3: Self-Reflection
Before giving the final score, you must conduct the following checks:
  Completeness Check: Whether all requirements in the standard answer have been reviewed with no omissions.
  Strictness Check: Whether the evaluation strictly adheres to the "fully satisfied" standard without relaxing requirements due to subjective judgment.
  Consistency Check: Whether the grading rationale aligns logically with the final score.
  Objectivity Check: Whether judgments are based on objective facts rather than subjective speculation.
Output Format Requirements
【Grading Rationale】: xxx
【List of Requirement Satisfaction Status】: [x₁, x₂, …, xᵢ, …, xₙ] (where n is the total number of requirements in the 【Rubrics】, and xᵢ indicates whether the student's answer meets the i-th requirement, with values "yes"/"no")
【Overall Score】: x points (x is an integer, either 0 or 1.)

Content to Be Graded
【Rubrics】:
{rubrics_text}
【Student Response】:
{model_output}

Please strictly output ONLY the following JSON format (do not output any other content):
{{
  "Grading Rationale": "Your detailed grading rationale",
  "List of Requirement Satisfaction Status": ["yes", "no", ...],
  "Overall Score": 0 or 1
}}
'''


def call_judge_api(model: str, rubrics_text: str, model_output: str, max_retries: int) -> str | None:
    for attempt in range(max_retries):
        try:
            text = hc.chat([{"role": "user", "content": rubric_judge_prompt(rubrics_text=rubrics_text,
                                                                            model_output=model_output)}],
                           model, max_tokens=None)
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return text.strip()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(3)
    return None


def evaluate_rubric_clbench(model: str, rubrics: Sequence[str] | None, predicted_answer: str,
                            max_retries: int = 3) -> dict[str, Any]:
    # Synchronous port of the official evaluate_rubric_clbench (same branches and payloads).
    normalized_answer = _normalize_model_output(predicted_answer)
    normalized_rubrics = _normalize_rubrics(rubrics)
    if not normalized_answer:
        return _failed_payload("No model output (counted as score 0)")
    if not normalized_rubrics:
        return _failed_payload("No rubrics provided")
    rubrics_text = _build_rubrics_text(normalized_rubrics)
    for parse_attempt in range(max_retries):
        grading_result = call_judge_api(model, rubrics_text, normalized_answer, max_retries)
        if not grading_result:
            if parse_attempt < max_retries - 1:
                time.sleep(2)
                continue
            return _failed_payload("API call failed (counted as score 0)")
        try:
            payload = json.loads(grading_result)
            if "Overall Score" not in payload:
                raise ValueError("Missing 'Overall Score' field")
            statuses = _coerce_status_list(payload.get("List of Requirement Satisfaction Status", []))
            return {
                "rubric_clbench_score": float(_coerce_score(payload)),
                "rubric_clbench_rationale": str(payload.get("Grading Rationale", "") or ""),
                "rubric_clbench_requirement_status": statuses,
                "rubric_clbench_requirement_ratio": _compute_requirement_ratio(statuses),
            }
        except (json.JSONDecodeError, ValueError):
            if parse_attempt < max_retries - 1:
                time.sleep(2)
                continue
            return _failed_payload(f"JSON parse failed ({max_retries} attempts): {grading_result[:500]}")
    return _failed_payload("Unknown error (counted as score 0)")


# ----------------------------------------------------------------------------- data

def load() -> list[dict]:
    # Split on "\n" only: str.splitlines() also breaks on U+2028/U+0085 inside JSON strings.
    with RAW.open(encoding="utf-8", newline="\n") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def task_id(t: dict) -> str:
    return str(t["metadata"]["task_id"])


def full_text(t: dict) -> str:
    return "\n".join(str(m.get("content", "")) for m in t["messages"])


def stage_select(work: Path, total: int, seed: int, max_chars: int) -> None:
    data = load()
    excl: dict[str, str] = {}
    too_long = 0
    pool: dict[str, list] = defaultdict(list)
    bad_ctx: set[str] = set()
    for t in data:
        tid = task_id(t)
        txt = full_text(t)
        hit = hc.smoke_query_hit(txt)
        if hit:
            excl[tid] = f"smoke_query: {hit[:60]}"
        else:
            ent = hc.entity_hit(txt + "\n" + json.dumps(t.get("rubrics", []), ensure_ascii=False))
            if ent:
                excl[tid] = f"entity: {ent}"
        if tid in excl:
            bad_ctx.add(str(t["metadata"].get("context_id")))
    for t in data:
        tid = task_id(t)
        if tid in excl:
            continue
        if str(t["metadata"].get("context_id")) in bad_ctx:
            excl[tid] = "context_shared: same context_id as an excluded task"
            continue
        if len(full_text(t)) > max_chars:
            too_long += 1
            continue
        if t["messages"][-1]["role"] != "user":
            excl[tid] = "last message is not a user turn"
            continue
        pool[t["metadata"]["context_category"]].append(tid)
    chosen = hc.stratified(pool, total, seed)
    reasons = defaultdict(int)
    for v in excl.values():
        reasons[v.split(":")[0] + (":" + v.split(":")[1].strip() if v.startswith("entity") else "")] += 1
    sel = {"dataset": DATASET, "seed": seed, "total_items": len(data), "excluded": excl, "n_excluded": len(excl),
           "n_excluded_too_long": too_long, "max_chars": max_chars, "selected": chosen,
           "pool_sizes": {k: len(v) for k, v in pool.items()}}
    hc.write_json(work / "selection.json", sel)
    print(f"select: {len(data)} tasks, excluded for smoke overlap {len(excl)} {dict(reasons)}, "
          f"dropped as > {max_chars} chars {too_long}, selected {len(chosen)}")


def selected(work: Path) -> list[dict]:
    want = set(hc.read_json(work / "selection.json")["selected"])
    return [t for t in load() if task_id(t) in want]


def stage_ingest(work: Path, workers: int) -> None:
    jobs = []
    base = 1_735_689_600_000  # 2025-01-01T00:00:00Z; tasks carry no timestamps
    for t in selected(work):
        tid = task_id(t)
        msgs = [{"role": m["role"], "content": m["content"], "timestamp": base + i * 60_000}
                for i, m in enumerate(t["messages"]) if m["role"] != "system" and str(m.get("content", "")).strip()]
        for c, chunk in enumerate(hc.chunk_messages(msgs)):
            jobs.append((f"clb_{tid}_c{c}", f"clb_{tid}", f"clb_{tid}", chunk))
    hc.ingest(jobs, work / "ingested.jsonl", workers)


def stage_run(work: Path, variant: str, workers: int, answer_model: str, judge_model: str, top_k: int) -> None:
    hc.verify_constants(PIPELINE, {"_CLBENCH_ANSWER_PROMPT_TEMPLATE": _CLBENCH_ANSWER_PROMPT_TEMPLATE})
    hc.verify_source_snippet(PIPELINE, "def rubric_judge_prompt", "\n'''", Path(__file__))
    hc.verify_source_snippet(PIPELINE, "def format_selected_memories", 'return "\\n".join(lines)', Path(__file__))
    hc.verify_source_snippet(PIPELINE, "def render_answer_prompt_clbench", ").strip()", Path(__file__))
    hc.warm_imports(variant)
    out_path = work / f"results_{variant}.jsonl"
    done = {r["id"] for r in hc.jl_read(out_path)}
    todo = []
    for t in selected(work):
        tid = task_id(t)
        if tid in done:
            continue
        sys_prompt = "\n\n".join(m["content"] for m in t["messages"] if m["role"] == "system")
        todo.append({"id": tid, "user_id": f"clb_{tid}", "question": t["messages"][-1]["content"],
                     "system_prompt": sys_prompt, "category": t["metadata"]["context_category"],
                     "sub_category": t["metadata"].get("sub_category"), "rubrics": t.get("rubrics", []),
                     "n_turns": len(t["messages"])})

    def one(q: dict) -> dict:
        items = le.retrieve(variant, q, top_k)
        sel = [{"created_at": it.get("created_at"), "text": it.get("text") or it.get("content")} for it in items]
        mem = format_selected_memories(sel)
        prompt = render_answer_prompt_clbench(system_prompt=q["system_prompt"], memories=mem, question=q["question"])
        gen = hc.chat([{"role": "user", "content": prompt}], answer_model, max_tokens=8192)
        res = evaluate_rubric_clbench(judge_model, q["rubrics"], gen)
        score = res["rubric_clbench_score"]
        out = {k: v for k, v in q.items() if k not in ("question", "system_prompt")}
        return {**out, "question": q["question"][-2000:], "variant": variant, "n_items": len(items),
                "context_bytes": len(mem.encode()), "context": mem[:30_000], "generated": gen,
                "score": score, "correct": score >= 1.0, "judge": res["rubric_clbench_rationale"],
                "requirement_ratio": res["rubric_clbench_requirement_ratio"],
                "requirement_status": res["rubric_clbench_requirement_status"]}

    hc.run_parallel(todo, one, out_path, workers)
    stage_report(work, variant)


def stage_report(work: Path, variant: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.report(rs, f"held-out CL-Bench, variant={variant} (binary rubric score)",
              out_json=work / f"summary_{variant}.json")
    if rs:
        m, lo, hi = hc.mean_ci([r["requirement_ratio"] for r in rs])
        print(f"  mean requirement satisfaction ratio (diagnostic): {m:.1%} [{lo:.1%}, {hi:.1%}]")


def stage_taxonomy(work: Path, variant: str, model: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.failure_taxonomy(rs, lambda r: not r["correct"],
                        lambda r: "Rubrics:\n" + _build_rubrics_text(_normalize_rubrics(r.get("rubrics"))),
                        work / f"taxonomy_{variant}.jsonl", model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "ingest", "run", "report", "taxonomy"])
    ap.add_argument("--work", default=str(hc.HELDOUT_HOME / DATASET / "work"))
    ap.add_argument("--variant", default="clean_wide")
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-chars", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--answer-model", default="qwen-plus")
    ap.add_argument("--judge-model", default="qwen-plus")
    a = ap.parse_args()
    hc.load_env()
    hc.require_isolated_db(DATASET)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "select":
        stage_select(work, a.total, a.seed, a.max_chars)
    elif a.stage == "ingest":
        stage_ingest(work, a.workers)
    elif a.stage == "run":
        stage_run(work, a.variant, a.workers, a.answer_model, a.judge_model, a.top_k)
    elif a.stage == "report":
        stage_report(work, a.variant)
    else:
        stage_taxonomy(work, a.variant, a.judge_model)


if __name__ == "__main__":
    main()
