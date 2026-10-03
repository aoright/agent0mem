"""Held-out BEAM (100K split) evaluation for the agent0mem textual memory track.

Data: HF Mohammadta/BEAM, data/100K-00000-of-00001.parquet (20 conversations, 400 probing
questions in 10 categories). The parquet is converted once to JSON with the venv in
~/aml_heldout/venv (pyarrow) by the `prepare` stage; all other stages use stdlib only.

Official AML pipeline: /tmp/aml-repo/data/beam/pipeline.py. Copied verbatim and checked
at runtime: ANSWER_GENERATION_FOR_RAG, UNIFIED_LLM_JUDGE_BASE_PROMPT, BATCH_OUTPUT_FORMAT,
EQUIVALENCE_SYSTEM_PROMPT, the batch-judge rendering, rubric parsing and the
event-ordering Kendall-tau/F1 metric.

Model substitution: official answer/judge model is Qwen/Qwen3-14B (SiliconFlow,
thinking disabled); here qwen-plus (DashScope) at temperature 0 for both.

Exclusions: a conversation is dropped entirely when any of its probing questions
overlaps a logged smoke query (e.g. the "city autocomplete" ordering question and
"Have I obtained an API key" both come from conversation 2). Individual questions
mentioning a smoke entity are also dropped.

Ingestion adapter: each chat session is split into POST /add batches of <= 6 messages /
12k chars; the per-turn index markers ("->-> 1,1") are stripped; timestamps come from
the latest time_anchor seen (sessions without one continue from the previous anchor).

Score: llm_judge_score = mean rubric score (0 / 0.5 / 1) per question, reported as a
mean with a 95% normal CI plus Wilson CI on full credit. event_ordering questions also
get the official tau_norm * F1 metric (reported separately).

Usage:
  ~/aml_heldout/venv/bin/python scripts/heldout/beam_eval.py prepare
  AGENT0MEM_DATA_DIR=~/aml_heldout/beam/db python3 scripts/heldout/beam_eval.py select
  AGENT0MEM_DATA_DIR=~/aml_heldout/beam/db python3 scripts/heldout/beam_eval.py ingest
  AGENT0MEM_DATA_DIR=~/aml_heldout/beam/db python3 scripts/heldout/beam_eval.py run --variant clean_wide
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import heldout_common as hc
from heldout_common import le

DATASET = "beam"
RAW_DIR = hc.HELDOUT_HOME / DATASET / "raw"
PARQUET = RAW_DIR / "100K.parquet"
RAW = RAW_DIR / "100K.json"
PIPELINE = hc.AML_REPO / "beam" / "pipeline.py"

# Verbatim src/prompts.py:answer_generation_for_rag at UPSTREAM_COMMIT.
ANSWER_GENERATION_FOR_RAG = """
You are an assistant that MUST answer questions using ONLY the information provided in the context below. 

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Do NOT use your internal knowledge

CONTEXT:
<context>

QUESTION:
<question>

ANSWER REQUIREMENTS:
- Be direct and concise
- Only output the answer to the question without any explanation 

RESPONSE:
"""

# Verbatim src/prompts.py:unified_llm_judge_base_prompt at UPSTREAM_COMMIT.
UNIFIED_LLM_JUDGE_BASE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., \u201c$68,000\u201d, \u201c68k\u201d, \u201c68,000 USD\u201d, or \u201csixty-eight thousand dollars\u201d). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., \u201citemized list\u201d, \u201cno citations\u201d, \u201cone sentence\u201d).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
   
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""


BATCH_OUTPUT_FORMAT = """## OUTPUT FORMAT:
Return one independent evaluation for every indexed rubric criterion in JSON:

{
  "scores": [
    {"index": 0, "score": 1.0, "reason": "detailed justification"}
  ]
}

Include every index exactly once. Each score must be 1.0, 0.5, or 0.0.
NOTE: ONLY output the json object, without any explanation before or after that
"""


EQUIVALENCE_SYSTEM_PROMPT = """You are a binary classifier.
If the TWO snippets describe the SAME event/fact, reply **YES**
Otherwise reply **NO**. No extra words.
DO NOT provide any exaplanation."""


def render_batch_judge_prompt(question: str, response: str, rubrics: list[str]) -> str:
    criteria = "\n".join(f"[{index}] {rubric}" for index, rubric in enumerate(rubrics))
    prompt = (UNIFIED_LLM_JUDGE_BASE_PROMPT
              .replace("<question>", question)
              .replace("<rubric_item>", criteria)
              .replace("<llm_response>", response))
    prompt = prompt[:prompt.index("## OUTPUT FORMAT:")] + BATCH_OUTPUT_FORMAT
    return ("Evaluate every indexed RUBRIC CRITERION independently. Apply the complete protocol "
            "below separately to each criterion; do not let one criterion affect another.\n\n" + prompt)


def parse_json_object(response: str) -> dict:
    candidate = response.strip()
    if candidate.startswith("```"):
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            raise ValueError("model response does not contain a JSON object") from None
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    return payload


def parse_rubric_scores(response: str, count: int) -> list[dict]:
    payload = parse_json_object(response)
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, list):
        raise ValueError("judge response must contain a scores list")
    scores = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            raise ValueError("every score must be an object")
        try:
            index = int(item["index"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("every score needs a numeric index and score") from error
        if index in scores or not 0 <= index < count:
            raise ValueError("score indices must be unique and in range")
        if score not in {0.0, 0.5, 1.0}:
            raise ValueError("scores must be exactly 0, 0.5, or 1")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("every score needs a non-empty reason")
        scores[index] = {"index": index, "score": score, "reason": reason.strip()}
    if set(scores) != set(range(count)):
        raise ValueError("judge must return every rubric index exactly once")
    return [scores[index] for index in range(count)]


def kendall_tau_b(first: list[int], second: list[int]) -> float:
    concordant = discordant = first_ties = second_ties = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            first_delta = first[left] - first[right]
            second_delta = second[left] - second[right]
            if first_delta == 0 and second_delta == 0:
                continue
            if first_delta == 0:
                first_ties += 1
            elif second_delta == 0:
                second_ties += 1
            elif first_delta * second_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + first_ties) *
        (concordant + discordant + second_ties)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def event_ordering_metrics(reference: list[str], system: list[str]) -> dict:
    true_positive = len(set(reference) & set(system))
    false_positive = len([item for item in system if item not in reference])
    false_negative = len([item for item in reference if item not in system])
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = list(dict.fromkeys(reference + system))
    tie_rank = len(union) + 1

    def ranks(sequence: list[str]) -> list[int]:
        positions = {item: index + 1 for index, item in enumerate(sequence)}
        return [positions.get(item, tie_rank) for item in union]

    tau_norm = (kendall_tau_b(ranks(reference), ranks(system)) + 1) / 2
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tau_norm": tau_norm,
        "final_score": tau_norm * f1,
    }


def align_with_llm(model: str, reference: list[str], system: list[str]) -> tuple[list[str], list[str]]:
    # Same greedy alignment as the official align_with_llm, synchronous.
    used = set()
    system_output = []
    for candidate in system:
        matched_index = None
        for index, expected in enumerate(reference):
            if index in used:
                continue
            resp = hc.chat([{"role": "system", "content": EQUIVALENCE_SYSTEM_PROMPT},
                            {"role": "user", "content": f"First snippet: {expected}\n\nSecond snippet: {candidate}"}],
                           model, max_tokens=8)
            if "yes" in resp.casefold():
                matched_index = index
                break
        if matched_index is None:
            system_output.append(candidate)
        else:
            system_output.append(reference[matched_index])
            used.add(matched_index)
    return reference, system_output


# ----------------------------------------------------------------------------- data

def stage_prepare() -> None:
    import pyarrow.parquet as pq  # only available in ~/aml_heldout/venv

    rows = pq.read_table(PARQUET).to_pylist()
    RAW.write_text(json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"prepare: wrote {len(rows)} conversations to {RAW}")


def load() -> list[dict]:
    return json.loads(RAW.read_text(encoding="utf-8"))


def probing(conv: dict) -> dict[str, list[dict]]:
    pq = conv["probing_questions"]
    if isinstance(pq, dict):
        return pq
    try:
        return json.loads(pq)
    except json.JSONDecodeError:
        return ast.literal_eval(pq)


def gold_text(q: dict) -> str:
    for k in ("answer", "ideal_response", "ideal_summary", "expected_compliance"):
        if q.get(k):
            return str(q[k])
    return ""


def stage_select(work: Path, total: int, seed: int) -> None:
    data = load()
    excl_conv: dict[str, str] = {}
    excl_q: dict[str, str] = {}
    n_items = 0
    for conv in data:
        cid = str(conv["conversation_id"])
        for cat, qs in probing(conv).items():
            for i, q in enumerate(qs):
                n_items += 1
                hit = hc.smoke_query_hit(q["question"])
                if hit:
                    excl_conv[cid] = f"question overlaps smoke query: {hit[:70]}"
    pool: dict[str, list] = defaultdict(list)
    for conv in data:
        cid = str(conv["conversation_id"])
        for cat, qs in probing(conv).items():
            for i, q in enumerate(qs):
                qid = f"{cid}:{cat}:{i}"
                if cid in excl_conv:
                    excl_q[qid] = "conversation excluded"
                    continue
                ent = hc.entity_hit(f"{q['question']}\n{gold_text(q)}\n{json.dumps(q.get('rubric', []))}")
                if ent:
                    excl_q[qid] = f"entity: {ent}"
                    continue
                pool[cat].append(qid)
    chosen = hc.stratified(pool, total, seed)
    sel = {"dataset": DATASET, "seed": seed, "total_items": n_items, "excluded_conversations": excl_conv,
           "excluded": excl_q, "n_excluded": len(excl_q), "selected": chosen,
           "pool_sizes": {k: len(v) for k, v in pool.items()}}
    hc.write_json(work / "selection.json", sel)
    print(f"select: {n_items} items, excluded conversations {excl_conv}, excluded items {len(excl_q)}, "
          f"selected {len(chosen)}")
    for k, v in excl_q.items():
        if v != "conversation excluded":
            print(f"  excluded {k}: {v}")


def selected_items(work: Path) -> list[dict]:
    sel = hc.read_json(work / "selection.json")
    want = set(sel["selected"])
    out = []
    for conv in load():
        cid = str(conv["conversation_id"])
        for cat, qs in probing(conv).items():
            for i, q in enumerate(qs):
                qid = f"{cid}:{cat}:{i}"
                if qid in want:
                    out.append({"id": qid, "conv": cid, "category": cat, "raw": q})
    return out


_MARK_RE = re.compile(r"\s*->->\s*[\d,\s]+$")


def parse_anchor(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def stage_ingest(work: Path, workers: int) -> None:
    convs = {str(c["conversation_id"]): c for c in load()}
    need = sorted({it["conv"] for it in selected_items(work)}, key=lambda x: int(x) if x.isdigit() else x)
    jobs = []
    for cid in need:
        conv = convs[cid]
        uid = f"beam_{cid}"
        anchor = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for s_idx, sess in enumerate(conv["chat"]):
            msgs = []
            for t_idx, m in enumerate(sess):
                a = parse_anchor(m.get("time_anchor"))
                if a and a >= anchor:
                    anchor = a
                elif a:
                    anchor = a
                text = _MARK_RE.sub("", str(m.get("content", ""))).strip()
                if not text:
                    continue
                msgs.append({"role": m.get("role", "user"), "content": text,
                             "timestamp": int((anchor + timedelta(seconds=30 * t_idx)).timestamp() * 1000)})
            for c_idx, chunk in enumerate(hc.chunk_messages(msgs)):
                jobs.append((f"beam_{cid}_s{s_idx}_c{c_idx}", uid, f"beam_{cid}_s{s_idx}", chunk))
            anchor = anchor + timedelta(days=1)
    hc.ingest(jobs, work / "ingested.jsonl", workers)


def stage_run(work: Path, variant: str, workers: int, answer_model: str, judge_model: str, top_k: int) -> None:
    hc.verify_constants(PIPELINE, {"ANSWER_GENERATION_FOR_RAG": ANSWER_GENERATION_FOR_RAG,
                                   "UNIFIED_LLM_JUDGE_BASE_PROMPT": UNIFIED_LLM_JUDGE_BASE_PROMPT,
                                   "BATCH_OUTPUT_FORMAT": BATCH_OUTPUT_FORMAT,
                                   "EQUIVALENCE_SYSTEM_PROMPT": EQUIVALENCE_SYSTEM_PROMPT})
    hc.verify_source_snippet(PIPELINE, "def render_batch_judge_prompt", "+ prompt)", Path(__file__))
    hc.verify_source_snippet(PIPELINE, "def event_ordering_metrics", '"final_score": tau_norm * f1,', Path(__file__))
    hc.warm_imports(variant)
    out_path = work / f"results_{variant}.jsonl"
    done = {r["id"] for r in hc.jl_read(out_path)}
    todo = []
    for it in selected_items(work):
        if it["id"] in done:
            continue
        q = it["raw"]
        todo.append({"id": it["id"], "user_id": f"beam_{it['conv']}", "question": q["question"],
                     "category": it["category"], "rubric": [str(r) for r in q.get("rubric", [])],
                     "gold_answer": gold_text(q)})

    def one(q: dict) -> dict:
        items = le.retrieve(variant, q, top_k)
        mem = le.render_items(items)
        prompt = re.sub(r"<(context|question)>", lambda m: {"context": mem, "question": q["question"]}[m.group(1)],
                        ANSWER_GENERATION_FOR_RAG)
        gen = hc.chat([{"role": "user", "content": prompt}], answer_model, max_tokens=512)
        rubrics = q["rubric"] or [f"LLM response should state: {q['gold_answer']}"]
        jresp, scores, err = "", None, None
        for _ in range(3):
            jresp = hc.chat([{"role": "user", "content": render_batch_judge_prompt(q["question"], gen, rubrics)}],
                            judge_model, max_tokens=1024, json_mode=True)
            try:
                scores = parse_rubric_scores(jresp, len(rubrics))
                break
            except ValueError as e:
                err = str(e)
        score = sum(s["score"] for s in scores) / len(scores) if scores else 0.0
        rec = {**q, "variant": variant, "n_items": len(items), "context_bytes": len(mem.encode()), "context": mem,
               "generated": gen, "rubric_scores": scores, "score": score, "correct": score >= 1.0,
               "judge": jresp, "judge_error": None if scores else err}
        if q["category"] == "event_ordering":
            ref, sysout = align_with_llm(judge_model, rubrics, gen.split("\n"))
            rec["event_ordering"] = event_ordering_metrics(ref, sysout)
        return rec

    hc.run_parallel(todo, one, out_path, workers)
    stage_report(work, variant)


def stage_report(work: Path, variant: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.report(rs, f"held-out BEAM-100K, variant={variant} (llm_judge_score)", graded=True,
              out_json=work / f"summary_{variant}.json")
    eo = [r["event_ordering"]["final_score"] for r in rs if r.get("event_ordering")]
    if eo:
        m, lo, hi = hc.mean_ci(eo)
        print(f"  event_ordering tau_norm*F1: mean={m:.1%} [{lo:.1%}, {hi:.1%}] n={len(eo)}")
    je = sum(1 for r in rs if r.get("judge_error"))
    if je:
        print(f"  judge parse failures (scored 0): {je}")


def stage_taxonomy(work: Path, variant: str, model: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.failure_taxonomy(rs, lambda r: r["score"] < 0.5,
                        lambda r: (r.get("gold_answer") or "") + "\nRubric: " + json.dumps(r.get("rubric")),
                        work / f"taxonomy_{variant}.jsonl", model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "select", "ingest", "run", "report", "taxonomy"])
    ap.add_argument("--work", default=str(hc.HELDOUT_HOME / DATASET / "work"))
    ap.add_argument("--variant", default="clean_wide")
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--answer-model", default="qwen-plus")
    ap.add_argument("--judge-model", default="qwen-plus")
    a = ap.parse_args()
    if a.stage == "prepare":
        stage_prepare()
        return
    hc.load_env()
    hc.require_isolated_db(DATASET)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "select":
        stage_select(work, a.total, a.seed)
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
