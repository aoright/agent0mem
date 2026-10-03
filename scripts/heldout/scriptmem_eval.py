"""Held-out ScriptMem evaluation harness (BLOCKED on data: see below).

Data: github.com/memorax-ai/ScriptMem data/raw/{angry,enemy,friends,man_earth}.json.
The public release contains the 457 questions and gold options but, per its
data/README.md, "for copyright reasons, this release does not include the original
script conversation text" (each conversation is a synthetic `format_example`). The
memory side of the benchmark therefore cannot be reproduced from public data.

This script is ready for when licensed script text is supplied locally:
  ~/aml_heldout/scriptmem/raw/conversations/<source>.json   (source in angry, enemy,
  friends, man_earth), each in the documented schema
  {"speakers": [...], "session_1_date_time": "...", "session_1": [{"type": "dialogue" |
  "narration", "speaker": ..., "dia_id": ..., "text": ...}, ...], ...}
Without those files `ingest` stops with an explanation; `select` works on the public
questions and records the smoke-overlap exclusions.

Official AML pipeline: /tmp/aml-repo/data/scriptmem/pipeline.py. Copied verbatim and
checked at runtime: CHOICE_ANSWER_TEMPLATE, gold_letters, normalize_prediction_text,
predicted_letters / predicted_ordered_letters / predicted_option_letters, score_item.
Scoring is deterministic (no LLM judge). Answer model: qwen-plus at temperature 0
(official ANSWER_MODEL is set in an api_config module that is not in the repo copy).

Exclusions: every question overlapping a logged smoke query (the chain-lock courtroom
demonstration, Foreman, elevator, Billing questions) or mentioning a smoke entity.

Usage:
  AGENT0MEM_DATA_DIR=~/aml_heldout/scriptmem/db python3 scripts/heldout/scriptmem_eval.py select
  AGENT0MEM_DATA_DIR=~/aml_heldout/scriptmem/db python3 scripts/heldout/scriptmem_eval.py ingest
  AGENT0MEM_DATA_DIR=~/aml_heldout/scriptmem/db python3 scripts/heldout/scriptmem_eval.py run --variant clean_wide
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import heldout_common as hc
from heldout_common import le

DATASET = "scriptmem"
RAW_DIR = hc.HELDOUT_HOME / DATASET / "raw"
CONV_DIR = RAW_DIR / "conversations"
PIPELINE = hc.AML_REPO / "scriptmem" / "pipeline.py"
DATASET_FILES = ("angry.json", "enemy.json", "friends.json", "man_earth.json")


CHOICE_ANSWER_TEMPLATE = """
You are asked to answer a multiple-choice question based on your memories of a conversation.

<instructions>
1. Use only the provided memories. Prefer the memories that answer the question most directly.
2. Your memories are episodic raw observations. Reason about what they imply. Do not refuse just because the answer is not stated verbatim.
3. The question may contain typos. Match it to the most relevant memories even if the wording differs.
4. The question may be single-choice, multi-select, or ordering. The required output format is different for each type. Obey the format stated in the question, not a default format you invent.
5. Do not add unsupported options. Do not omit supported options. Do not hedge.
6. Preserve option letters exactly. Do not rewrite the answer as option text unless the question explicitly asks for that.
7. If memories conflict, prefer the most recent supported memory.
8. Choose "Cannot infer" only when no memory contains any relevant evidence after scanning all memories. Partial or indirect evidence requires a supported answer, not a refusal.
9. When memories conflict in direction, prefer the one semantically closest to the question's core\u2014not the one with the highest keyword overlap.
10. For multi-select: check every option independently. Include if any memory supports it; exclude only if clearly contradicted or out of scope.
11. For "most plausible", "underlying", or "most strongly implies" questions: compare the top candidates directly before choosing; do not default to the option with the most shared vocabulary.
12. Keep reasoning internal. The visible output must be just the answer string required by the question.
</instructions>

<memories>
Memories for user {{speaker_1_name}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_name}}:

{{speaker_2_memories}}
</memories>

Question: {{question}}
Return only the answer, exactly in the format requested by the question:
""".strip()


def gold_letters(answer: Any) -> list[str]:
    parts = answer if isinstance(answer, list) else [answer]
    letters: list[str] = []
    for part in parts:
        match = re.match(r"\s*([A-F])\.", str(part))
        if match:
            letters.append(match.group(1))
    return letters


def normalize_prediction_text(text: str) -> str:
    cleaned = str(text or "").strip()
    box_matches = list(re.finditer(r"\\box(?:ed)?\{([^}]*)(?:\}|$)", cleaned))
    if box_matches:
        return box_matches[-1].group(1).strip()
    lower = cleaned.lower()
    if "final answer:" in lower:
        index = lower.index("final answer:")
        cleaned = cleaned[index + len("final answer:") :].strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned


def predicted_letters(prediction: str, qa_type: str = "") -> tuple[list[str], bool]:
    normalized = normalize_prediction_text(prediction)
    if not normalized:
        return [], False

    if qa_type in {"multi_select", "ordering"}:
        return predicted_ordered_letters(normalized)

    return predicted_option_letters(normalized)


def predicted_ordered_letters(prediction: str) -> tuple[list[str], bool]:
    paren_matches = list(re.finditer(r"[\(\[]([^\)\]]*)[\)\]]", prediction))
    content = paren_matches[-1].group(1) if paren_matches else prediction
    letters = [letter.upper() for letter in re.findall(r"[A-Fa-f]", content)]
    if letters:
        return letters, len(set(letters)) != len(letters)
    return [], False


def predicted_option_letters(prediction: str) -> tuple[list[str], bool]:
    if re.fullmatch(r"[A-Fa-f]{1,5}", prediction):
        return [letter.upper() for letter in prediction], False
    if re.search(r"\(\s*[A-Fa-f]\s*\)\(\s*[A-Fa-f]\s*\)", prediction) or re.search(
        r"\[\s*[A-Fa-f]\s*\]\[\s*[A-Fa-f]\s*\]",
        prediction,
    ):
        return [], True

    options: set[str] = set()
    token_re = re.compile(r"\([^)]*\)|\[[^\]]*\]")
    for match in token_re.finditer(prediction):
        inner = match.group(0)[1:-1].strip()
        if not inner:
            continue

        single_letter_match = re.fullmatch(r"([A-Fa-f])", inner)
        if single_letter_match:
            options.add(single_letter_match.group(1).upper())
            continue

        labeled_text_match = re.match(r"^([A-Fa-f])\s*[.:]\s*.+$", inner)
        if labeled_text_match:
            options.add(labeled_text_match.group(1).upper())
            continue

        letters_only = re.sub(r"[^A-Za-z]", "", inner)
        if (
            letters_only
            and len(letters_only) <= 5
            and inner[0].upper() in {"A", "B", "C", "D", "E", "F"}
            and re.fullmatch(r"[A-Za-z ]+", inner)
        ):
            options.add(inner[0].upper())
    return sorted(options), False


def score_item(qa_type: str, gold: list[str], pred: list[str], malformed: bool) -> float:
    if malformed:
        return 0.0
    if qa_type == "single_choice":
        return 1.0 if len(gold) == 1 and len(pred) == 1 and gold[0] == pred[0] else 0.0
    if qa_type == "multi_select":
        return 1.0 if bool(gold) and set(gold) == set(pred) and len(pred) == len(set(pred)) else 0.0
    if qa_type == "ordering":
        return 1.0 if bool(gold) and gold == pred else 0.0
    raise ValueError(f"unsupported qa_type: {qa_type}")


# ----------------------------------------------------------------------------- data

def load_gold_records() -> list[dict]:
    records = []
    for filename in DATASET_FILES:
        source = filename[:-5]
        data = json.loads((RAW_DIR / filename).read_text(encoding="utf-8"))
        for sample_index, sample in enumerate(data):
            sample_id = sample.get("sample_id") or f"{source}-{sample_index}"
            for qa_index, qa in enumerate(sample.get("qa", [])):
                records.append({"qa_id": f"{source}:{sample_id}#q{qa_index:04d}", "dataset": source,
                                "qa_type": qa["qa_type"], "question": qa["question"], "answer": qa["answer"]})
    return records


def stage_select(work: Path, total: int, seed: int) -> None:
    recs = load_gold_records()
    excl: dict[str, str] = {}
    pool: dict[str, list] = defaultdict(list)
    for r in recs:
        hit = hc.smoke_query_hit(r["question"])
        if hit:
            excl[r["qa_id"]] = f"smoke_query: {hit[:60]}"
            continue
        ent = hc.entity_hit(r["question"] + "\n" + json.dumps(r["answer"], ensure_ascii=False))
        if ent:
            excl[r["qa_id"]] = f"entity: {ent}"
            continue
        pool[r["qa_type"]].append(r["qa_id"])
    chosen = hc.stratified(pool, total, seed)
    sel = {"dataset": DATASET, "seed": seed, "total_items": len(recs), "excluded": excl, "n_excluded": len(excl),
           "selected": chosen, "pool_sizes": {k: len(v) for k, v in pool.items()}}
    hc.write_json(work / "selection.json", sel)
    print(f"select: {len(recs)} questions, excluded {len(excl)}, selected {len(chosen)} pool={sel['pool_sizes']}")
    for k, v in excl.items():
        print(f"  excluded {k}: {v}")


def parse_session_date(s: str) -> int | None:
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return int(datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except (ValueError, AttributeError):
            pass
    return None


def stage_ingest(work: Path, workers: int) -> None:
    missing = [f for f in DATASET_FILES if not (CONV_DIR / f).exists()]
    if missing:
        raise SystemExit(
            "ScriptMem conversation text is not publicly released (copyright). Place licensed script "
            f"conversations in {CONV_DIR}/ ({', '.join(missing)} missing) using the schema in this "
            "script's docstring, then rerun ingest.")
    jobs = []
    fallback = 1_735_689_600_000
    for filename in DATASET_FILES:
        source = filename[:-5]
        conv = json.loads((CONV_DIR / filename).read_text(encoding="utf-8"))
        speakers = conv.get("speakers") or []
        n = 1
        prev = fallback
        while f"session_{n}" in conv:
            base = parse_session_date(str(conv.get(f"session_{n}_date_time", ""))) or prev + 86_400_000
            prev = base
            msgs = []
            for t_idx, turn in enumerate(conv[f"session_{n}"]):
                text = str(turn.get("text", "")).strip()
                if not text:
                    continue
                if turn.get("type") == "narration" or not turn.get("speaker"):
                    content, role = f"[Narration] {text}", "user"
                else:
                    content = f"{turn['speaker']}: {text}"
                    role = "user" if speakers and turn["speaker"] == speakers[0] else "assistant"
                msgs.append({"role": role, "content": content, "timestamp": base + t_idx * 30_000})
            for c, chunk in enumerate(hc.chunk_messages(msgs)):
                jobs.append((f"sm_{source}_s{n}_c{c}", f"sm_{source}", f"sm_{source}_s{n}", chunk))
            n += 1
    hc.ingest(jobs, work / "ingested.jsonl", workers)


def stage_run(work: Path, variant: str, workers: int, answer_model: str, top_k: int) -> None:
    hc.verify_constants(PIPELINE, {"CHOICE_ANSWER_TEMPLATE": CHOICE_ANSWER_TEMPLATE})
    hc.verify_source_snippet(PIPELINE, "def gold_letters", 'raise ValueError(f"unsupported qa_type: {qa_type}")',
                             Path(__file__))
    hc.warm_imports(variant)
    want = set(hc.read_json(work / "selection.json")["selected"])
    out_path = work / f"results_{variant}.jsonl"
    done = {r["id"] for r in hc.jl_read(out_path)}
    todo = [{"id": r["qa_id"], "user_id": f"sm_{r['dataset']}", "question": r["question"], "category": r["qa_type"],
             "source": r["dataset"], "gold": gold_letters(r["answer"])}
            for r in load_gold_records() if r["qa_id"] in want and r["qa_id"] not in done]

    def one(q: dict) -> dict:
        items = le.retrieve(variant, q, top_k)
        mem = le.render_items(items)
        prompt = (CHOICE_ANSWER_TEMPLATE.replace("{{speaker_1_name}}", "speaker 1")
                  .replace("{{speaker_1_memories}}", mem).replace("{{speaker_2_name}}", "speaker 2")
                  .replace("{{speaker_2_memories}}", "").replace("{{question}}", q["question"]))
        gen = hc.chat([{"role": "user", "content": prompt}], answer_model, max_tokens=256)
        pred, malformed = predicted_letters(gen, q["category"])
        s = score_item(q["category"], q["gold"], pred, malformed)
        return {**q, "variant": variant, "n_items": len(items), "context_bytes": len(mem.encode()), "context": mem,
                "generated": gen, "predicted": pred, "malformed": malformed, "score": s, "correct": s >= 1.0,
                "judge": f"pred={pred} gold={q['gold']} malformed={malformed}"}

    hc.run_parallel(todo, one, out_path, workers)
    stage_report(work, variant)


def stage_report(work: Path, variant: str) -> None:
    hc.report(hc.jl_read(work / f"results_{variant}.jsonl"), f"held-out ScriptMem, variant={variant}",
              out_json=work / f"summary_{variant}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "ingest", "run", "report"])
    ap.add_argument("--work", default=str(hc.HELDOUT_HOME / DATASET / "work"))
    ap.add_argument("--variant", default="clean_wide")
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--answer-model", default="qwen-plus")
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
        stage_run(work, a.variant, a.workers, a.answer_model, a.top_k)
    else:
        stage_report(work, a.variant)


if __name__ == "__main__":
    main()
