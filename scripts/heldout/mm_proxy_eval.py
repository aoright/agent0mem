"""Multimodal-track PROXY evaluation on held-out LoCoMo conversations.

The multimodal track has no smoke attempts left and its official dataset is not
available, so this proxy measures the two behaviours that dominated the official
multimodal smoke (9 of 13 items were evidence governance / abstention):

  abstain : LoCoMo category 5 (adversarial). The question attributes an event to the
            wrong speaker; the stored "adversarial_answer" is the trap. Correct means
            the answer says the information is not available for that person.
  image   : LoCoMo categories 1-4 whose evidence turns include a shared photo
            (blip_caption). Correct per the official AML LoCoMo judge prompt.

Both sets are needed: an engine that always abstains would score 100% on abstain and
0% on image, so the report shows each set and their mean (balanced accuracy).

Caveats (do not read results as a multimodal score prediction):
- English only; the official set also has Chinese conversations.
- Photos are represented by their BLIP captions, not pixels, so the vision pipeline
  is not exercised.
- The official multimodal answer/judge prompts are unknown; a neutral prompt is used.

Reuses the DB produced by locomo_eval.py ingest (smoke-seen conversations excluded).

Usage:
  AGENT0MEM_DATA_DIR=~/aml_heldout/db2 python3 scripts/heldout/mm_proxy_eval.py run --variant legacy
  AGENT0MEM_DATA_DIR=~/aml_heldout/db2 python3 scripts/heldout/mm_proxy_eval.py run --variant clean_wide
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo_eval as le  # noqa: E402

ANSWER_PROMPT = """You are answering a question about a user using only the memories retrieved from their past conversations.

Rules:
- Use only the memories below. Do not use outside knowledge.
- Be careful about WHO did or said something. If the memories show that a different person did it, the information is not available for the person asked about.
- If the memories do not contain the information needed, answer exactly: Not mentioned.
- Otherwise answer with the shortest correct phrase or sentence.

<memories>
{memories}
</memories>

Question: {question}
Answer:"""

ABSTAIN_JUDGE = """You are grading whether an assistant correctly declined to answer.

The question below asks about a fact that is NOT recorded for the person named in the question
(in the source conversation the fact belongs to someone else, or does not exist).
Trap answer (what a careless assistant would say): {trap}

Generated answer: {generated}

Label CORRECT if the generated answer says the information is not mentioned, unknown, not available,
or explicitly attributes it to a different person. Label WRONG if it asserts an answer (including
the trap answer or any other fact) as being true for the person named in the question.

Return only JSON: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""

SET_NAMES = ("abstain", "image")


def build_sets(per_set: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = defaultdict(list)
    for conv in le.heldout_conversations():
        c = conv["conversation"]
        img_ids = set()
        n = 1
        while f"session_{n}" in c:
            for t in c[f"session_{n}"]:
                if t.get("blip_caption"):
                    img_ids.add(t.get("dia_id"))
            n += 1
        for i, qa in enumerate(conv["qa"]):
            base = {"id": f"{conv['sample_id']}#{i}", "user_id": f"heldout_{conv['sample_id']}",
                    "question": qa["question"], "category": qa["category"]}
            if qa["category"] == 5:
                pools["abstain"].append({**base, "set": "abstain",
                                         "gold_answer": "Not mentioned",
                                         "trap": str(qa.get("adversarial_answer", ""))})
            elif qa["category"] in (1, 2, 3, 4) and "answer" in qa:
                if any(e in img_ids for e in qa.get("evidence", [])):
                    pools["image"].append({**base, "set": "image", "gold_answer": str(qa["answer"])})
    out = []
    for name in SET_NAMES:
        p = pools[name]
        rng.shuffle(p)
        out.extend(p[:per_set])
    return out


def parse_label(resp: str) -> str:
    m = re.search(r"\{.*?\}", resp, re.DOTALL)
    if m:
        try:
            return str(json.loads(m.group(0)).get("label", "WRONG")).upper()
        except Exception:  # noqa: BLE001
            pass
    return "CORRECT" if "CORRECT" in resp and "WRONG" not in resp else "WRONG"


def stage_run(work: Path, variant: str, per_set: int, seed: int, workers: int,
              answer_model: str, judge_model: str, top_k: int) -> None:
    if variant == "legacy":
        import app.database  # noqa: F401
    else:
        import app.core.clean_search  # noqa: F401
    qs = build_sets(per_set, seed)
    out_path = work / f"mm_results_{variant}.jsonl"
    done = {r["id"] for r in le.jl_read(out_path)}
    todo = [q for q in qs if q["id"] not in done]
    print(f"mm_proxy[{variant}]: {len(todo)} of {len(qs)} pending")

    def one(q):
        items = le.retrieve(variant, q, top_k)
        mem = le.render_items(items)
        gen = le.llm(ANSWER_PROMPT.format(memories=mem, question=q["question"]), answer_model, max_tokens=200)
        if q["set"] == "abstain":
            jresp = le.llm(ABSTAIN_JUDGE.format(trap=q["trap"], generated=gen), judge_model, max_tokens=60)
        else:
            jprompt = re.sub(r"\{(question|gold_answer|generated_answer)\}",
                             lambda m: {"question": q["question"], "gold_answer": q["gold_answer"],
                                        "generated_answer": gen}[m.group(1)], le.ACCURACY_PROMPT)
            jresp = le.llm(jprompt, judge_model, max_tokens=200)
        label = parse_label(jresp)
        return {**q, "variant": variant, "n_items": len(items), "context_bytes": len(mem.encode()),
                "generated": gen, "label": label, "correct": label == "CORRECT", "judge": jresp}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, q) for q in todo]
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                le.jl_append(out_path, f.result())
            except Exception as e:  # noqa: BLE001
                print("  error:", e)
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")
    stage_report(work, variant)


def stage_report(work: Path, variant: str) -> None:
    rs = le.jl_read(work / f"mm_results_{variant}.jsonl")
    if not rs:
        print("no results")
        return
    print(f"\n=== multimodal PROXY (held-out LoCoMo), variant={variant}, n={len(rs)} ===")
    accs = []
    for name in SET_NAMES:
        g = [r for r in rs if r["set"] == name]
        if not g:
            continue
        k = sum(r["correct"] for r in g)
        lo, hi = le.wilson(k, len(g))
        accs.append(k / len(g))
        print(f"  {name:8s} {k:3d}/{len(g):3d} = {k/len(g):6.1%}  95%CI [{lo:5.1%}, {hi:5.1%}]"
              f"  avg_items={sum(r['n_items'] for r in g)/len(g):5.1f}")
    if len(accs) == len(SET_NAMES):
        print(f"  balanced accuracy = {sum(accs)/len(accs):6.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--work", default=os.path.expanduser("~/aml_heldout/work2"))
    ap.add_argument("--variant", default="legacy")
    ap.add_argument("--per-set", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--answer-model", default="qwen-plus")
    ap.add_argument("--judge-model", default="qwen-plus")
    a = ap.parse_args()
    if "AGENT0MEM_DATA_DIR" not in os.environ:
        raise SystemExit("set AGENT0MEM_DATA_DIR to an isolated directory (never production)")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "run":
        stage_run(work, a.variant, a.per_set, a.seed, a.workers, a.answer_model, a.judge_model, a.top_k)
    else:
        stage_report(work, a.variant)


if __name__ == "__main__":
    main()
