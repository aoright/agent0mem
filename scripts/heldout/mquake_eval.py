"""Held-out MQuAKE (counterfactual multi-hop knowledge editing) evaluation.

Data: princeton-nlp/MQuAKE, datasets/MQuAKE-CF-3k-v2.json (3000 cases; 2-4 hop questions
whose answer changes after 1-4 counterfactual edits), downloaded to
~/aml_heldout/mquake/raw.

There is NO official AML pipeline for MQuAKE in /tmp/aml-repo/data. This harness uses
the generic AML open-ended answer template and judge (identical in the LoCoMo and
LongMemEval-S pipelines, imported from locomo_eval.py). Treat the numbers as a proxy.

Memory construction (adapter choice): the requested edits of each case are stated by
the user as facts ("Fernando Santos is a citizen of United Kingdom.") in one POST /add
request per case. All cases share ONE user namespace, so retrieval must find the right
edits among those of --pool cases (the 60 evaluated cases plus distractor cases).
Unedited hops are not stored: the answer model has to combine retrieved edits with its
own knowledge, which is the MQuAKE setting.

Exclusions: any case whose questions, edits or hops mention a smoke entity (Tarsem
Singh, Portal, Mukesh, Latvia, Dune Messiah, ...) or overlap a logged smoke query is
removed from both the evaluated set and the distractor pool.

Question: questions[0] of the case; gold: new_answer. Category: number of hops.

Usage:
  AGENT0MEM_DATA_DIR=~/aml_heldout/mquake/db python3 scripts/heldout/mquake_eval.py select
  AGENT0MEM_DATA_DIR=~/aml_heldout/mquake/db python3 scripts/heldout/mquake_eval.py ingest
  AGENT0MEM_DATA_DIR=~/aml_heldout/mquake/db python3 scripts/heldout/mquake_eval.py run --variant clean_wide
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import heldout_common as hc
from heldout_common import le

DATASET = "mquake"
RAW = hc.HELDOUT_HOME / DATASET / "raw" / "MQuAKE-CF-3k-v2.json"
USER_ID = "mquake_pool"


def load() -> list[dict]:
    return json.loads(RAW.read_text(encoding="utf-8"))


def case_text(c: dict) -> str:
    parts = list(c["questions"]) + [c["answer"], c["new_answer"]]
    for r in c["requested_rewrite"]:
        parts += [r["prompt"].format(r["subject"]), r["target_new"]["str"], r["target_true"]["str"]]
    for h in c["single_hops"] + c["new_single_hops"]:
        parts += [h["question"], h["answer"]]
    return "\n".join(parts)


def edit_statements(c: dict) -> list[str]:
    return [f"{r['prompt'].format(r['subject'])} {r['target_new']['str']}." for r in c["requested_rewrite"]]


def stage_select(work: Path, total: int, seed: int, pool_size: int) -> None:
    data = load()
    excl: dict[str, str] = {}
    pool: dict[str, list] = defaultdict(list)
    ok = []
    for c in data:
        cid = str(c["case_id"])
        txt = case_text(c)
        hit = next((hc.smoke_query_hit(q) for q in c["questions"] if hc.smoke_query_hit(q)), None)
        if hit:
            excl[cid] = f"smoke_query: {hit[:60]}"
            continue
        ent = hc.entity_hit(txt)
        if ent:
            excl[cid] = f"entity: {ent}"
            continue
        ok.append(cid)
        pool[f"{len(c['single_hops'])}-hop"].append(cid)
    chosen = hc.stratified(pool, total, seed)
    rng = random.Random(seed + 1)
    rest = [x for x in ok if x not in set(chosen)]
    rng.shuffle(rest)
    distractors = rest[:max(0, pool_size - len(chosen))]
    sel = {"dataset": DATASET, "seed": seed, "total_items": len(data), "excluded": excl, "n_excluded": len(excl),
           "selected": chosen, "distractors": distractors, "pool_sizes": {k: len(v) for k, v in pool.items()}}
    hc.write_json(work / "selection.json", sel)
    reasons = defaultdict(int)
    for v in excl.values():
        reasons[v.split(":")[0] + ":" + v.split(":")[1].strip()[:20]] += 1
    print(f"select: {len(data)} cases, excluded {len(excl)} {dict(reasons)}, selected {len(chosen)}, "
          f"distractor cases {len(distractors)}")


def stage_ingest(work: Path, workers: int) -> None:
    sel = hc.read_json(work / "selection.json")
    want = sel["selected"] + sel["distractors"]
    by_id = {str(c["case_id"]): c for c in load()}
    order = sorted(want, key=int)
    base = 1_735_689_600_000  # 2025-01-01T00:00:00Z
    jobs = []
    for i, cid in enumerate(order):
        msgs = [{"role": "user", "content": s, "timestamp": base + i * 3_600_000 + k * 1000}
                for k, s in enumerate(edit_statements(by_id[cid]))]
        jobs.append((f"mq_{cid}", USER_ID, f"mq_{cid}", msgs))
    hc.ingest(jobs, work / "ingested.jsonl", workers)


def stage_run(work: Path, variant: str, workers: int, answer_model: str, judge_model: str, top_k: int) -> None:
    hc.warm_imports(variant)
    sel = hc.read_json(work / "selection.json")
    by_id = {str(c["case_id"]): c for c in load()}
    out_path = work / f"results_{variant}.jsonl"
    done = {r["id"] for r in hc.jl_read(out_path)}
    todo = []
    for cid in sel["selected"]:
        if cid in done:
            continue
        c = by_id[cid]
        todo.append({"id": cid, "user_id": USER_ID, "question": c["questions"][0], "gold_answer": c["new_answer"],
                     "aliases": c.get("new_answer_alias", []), "category": f"{len(c['single_hops'])}-hop",
                     "edits": edit_statements(c)})

    def one(q: dict) -> dict:
        items = le.retrieve(variant, q, top_k)
        mem = le.render_items(items)
        values = {"speaker_1_name": "user", "speaker_1_memories": mem, "speaker_2_name": "assistant",
                  "speaker_2_memories": "", "question": q["question"]}
        prompt = re.sub(r"\{\{(speaker_1_name|speaker_1_memories|speaker_2_name|speaker_2_memories|question)\}\}",
                        lambda m: values[m.group(1)], le.OPEN_ENDED_ANSWER_TEMPLATE)
        gen = le.llm(prompt, answer_model, max_tokens=256)
        jprompt = re.sub(r"\{(question|gold_answer|generated_answer)\}",
                         lambda m: {"question": q["question"], "gold_answer": q["gold_answer"],
                                    "generated_answer": gen}[m.group(1)], le.ACCURACY_PROMPT)
        jresp = le.llm(jprompt, judge_model, max_tokens=300)
        m = re.search(r"\{.*?\}", jresp, re.DOTALL)
        try:
            label = str(json.loads(m.group(0)).get("label", "WRONG")).upper() if m else "WRONG"
        except Exception:  # noqa: BLE001
            label = "WRONG"
        low = mem.lower()
        edits_found = sum(1 for e in q["edits"] if e.lower().rstrip(".") in low)
        return {**q, "variant": variant, "n_items": len(items), "context_bytes": len(mem.encode()), "context": mem,
                "edits_in_context": edits_found, "all_edits_in_context": edits_found == len(q["edits"]),
                "generated": gen, "label": label, "correct": label == "CORRECT", "judge": jresp}

    hc.run_parallel(todo, one, out_path, workers)
    stage_report(work, variant)


def stage_report(work: Path, variant: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.report(rs, f"held-out MQuAKE-CF-3k-v2 (proxy, generic AML judge), variant={variant}",
              out_json=work / f"summary_{variant}.json")
    if rs:
        a = sum(r["all_edits_in_context"] for r in rs) / len(rs)
        print(f"  all edit statements present verbatim in context: {a:.1%} (lexical lower bound)")


def stage_taxonomy(work: Path, variant: str, model: str) -> None:
    rs = hc.jl_read(work / f"results_{variant}.jsonl")
    hc.failure_taxonomy(rs, lambda r: not r["correct"],
                        lambda r: f"{r['gold_answer']} (edited facts that must be used: {' '.join(r['edits'])})",
                        work / f"taxonomy_{variant}.jsonl", model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "ingest", "run", "report", "taxonomy"])
    ap.add_argument("--work", default=str(hc.HELDOUT_HOME / DATASET / "work"))
    ap.add_argument("--variant", default="clean_wide")
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--pool", type=int, default=300, help="cases whose edits are stored (evaluated + distractors)")
    ap.add_argument("--seed", type=int, default=7)
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
        stage_select(work, a.total, a.seed, a.pool)
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
