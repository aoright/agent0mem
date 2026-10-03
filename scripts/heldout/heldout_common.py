"""Shared helpers for the multi-dataset held-out textual evaluation harness.

Every <dataset>_eval.py in this directory imports this module and locomo_eval.py.
Nothing here touches production: the DB location comes from AGENT0MEM_DATA_DIR and
require_isolated_db() refuses production-looking or in-repo paths.

Provided:
  - smoke-overlap detection against /tmp/textual_smoke_searches.txt plus the list of
    entities known to have appeared in official smoke runs
  - stratified sampling with a fixed seed
  - ingestion through app.core.extractor.extract_propositions +
    app.database.save_memories_batch (the same code path as POST /add)
  - a chat() helper that supports system messages and JSON mode
  - per-category report with 95% Wilson (binary) or normal (graded) intervals
  - an LLM-assisted failure taxonomy over a fixed-seed sample of wrong answers
"""
from __future__ import annotations

import ast
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
from pathlib import Path
from typing import Callable, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import locomo_eval as le  # noqa: E402

AML_REPO = Path("/tmp/aml-repo/data")
SMOKE_FILE = Path("/tmp/textual_smoke_searches.txt")
HELDOUT_HOME = Path(os.path.expanduser("~/aml_heldout"))

# Entities / topics that appeared in official textual smoke runs (beyond the logged
# queries). Matched case-insensitively on word boundaries against question + gold
# text (and source text where the dataset script says so). Deliberately broad:
# false positives only shrink the held-out pool.
SEEN_ENTITIES = [
    "Tarsem Singh", "Mukesh", "Latvia", "Dune Messiah", "Portal",
    "foreman", "Billing", "elevator",
    "API key", "autocomplete",
    "NTSB", "MIL-53", "OTC uptake",
    "INR", "\u6668\u50f5",  # Chinese medical patient (INR / morning stiffness)
    "Sarah Garcia", "Joaquim Molins",
    "coffee maker", "pickup basketball", "weekend reading", "Starbucks Ethos",
    "\u8ba2\u5a5a\u7eaa\u5ff5",
]


# ----------------------------------------------------------------------------- io

def jl_read(path: Path) -> list[dict]:
    """JSONL reader that splits on newline only (locomo_eval.jl_read uses splitlines(),
    which also breaks on U+2028/U+0085 that can occur unescaped in model output)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="\n") as fh:
        return [json.loads(x) for x in fh if x.strip()]


# ----------------------------------------------------------------------------- smoke overlap

def smoke_queries(path: Path = SMOKE_FILE) -> list[str]:
    out: list[str] = []
    if not path.exists():
        raise SystemExit(f"smoke query log not found: {path}")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"query=(.*), returned \d+ items\s*$", line)
        if not m:
            continue
        raw = m.group(1).strip()
        try:
            q = ast.literal_eval(raw)
        except Exception:  # noqa: BLE001  (logger may truncate the repr)
            q = raw.strip("'\"")
        q = str(q).strip()
        if q and q not in out:
            out.append(q)
    return out


def _toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", (s or "").lower())


_SMOKE_CACHE: list[tuple[str, str, set]] | None = None
_ENTITY_RES = [re.compile(r"(?<![A-Za-z0-9])" + re.escape(e) + r"(?![A-Za-z0-9])", re.IGNORECASE)
               for e in SEEN_ENTITIES]


def _smoke_index() -> list[tuple[str, str, set]]:
    global _SMOKE_CACHE
    if _SMOKE_CACHE is None:
        _SMOKE_CACHE = []
        for q in smoke_queries():
            t = _toks(q)
            _SMOKE_CACHE.append((q, " ".join(t), set(t)))
    return _SMOKE_CACHE


def smoke_query_hit(text: str) -> str | None:
    """Return the smoke query this text overlaps with, if any.

    A hit is either (a) the first 80 normalised characters of a smoke query occur
    inside the text, or (b) for short texts, token Jaccard >= 0.6 with a smoke query.
    """
    toks = _toks(text)
    norm = " ".join(toks)
    tset = set(toks)
    short = len(norm) < 600
    for q, nq, qset in _smoke_index():
        key = nq[:80]
        if len(key) >= 20 and key in norm:
            return q
        if short and qset and tset:
            j = len(qset & tset) / len(qset | tset)
            if j >= 0.6:
                return q
    return None


def entity_hit(text: str) -> str | None:
    for e, rx in zip(SEEN_ENTITIES, _ENTITY_RES):
        if rx.search(text or ""):
            return e
    return None


# ----------------------------------------------------------------------------- sampling

def stratified(pool: dict[str, list], total: int, seed: int) -> list:
    """Roughly equal draw per category; leftover slots go to categories with spare items."""
    rng = random.Random(seed)
    cats = sorted(pool)
    pools = {c: list(pool[c]) for c in cats}
    for c in cats:
        rng.shuffle(pools[c])
    base = total // max(1, len(cats))
    take = {c: min(base, len(pools[c])) for c in cats}
    left = total - sum(take.values())
    while left > 0:
        progressed = False
        for c in cats:
            if left and take[c] < len(pools[c]):
                take[c] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    out = []
    for c in cats:
        out.extend(pools[c][:take[c]])
    return out


# ----------------------------------------------------------------------------- safety

def require_isolated_db(dataset: str) -> Path:
    d = os.environ.get("AGENT0MEM_DATA_DIR")
    if not d:
        raise SystemExit(f"set AGENT0MEM_DATA_DIR=~/aml_heldout/{dataset}/db (never production)")
    p = Path(os.path.expanduser(d)).resolve()
    if str(p).startswith("/var/lib/agent0mem") or str(p).startswith("/opt/"):
        raise SystemExit(f"refusing production-looking data dir: {p}")
    if str(p).startswith(str(ROOT.resolve())):
        raise SystemExit(f"refusing data dir inside the repository: {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_env() -> None:
    """Load DASHSCOPE_* from the repo .env if not already set (no other keys)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(DASHSCOPE_[A-Z_]+)\s*=\s*(.*?)\s*$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip("'\"")


def verify_constants(pipeline: Path, consts: dict[str, str]) -> None:
    """Assert local prompt copies equal the official pipeline module-level strings."""
    if not pipeline.exists():
        print(f"note: {pipeline} not present; skipping verbatim check")
        return
    tree = ast.parse(pipeline.read_text(encoding="utf-8"))
    found = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    found[t.id] = n.value.value
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.targets[0], ast.Name):
            # e.g. "".strip() wrappers
            try:
                found[n.targets[0].id] = eval(compile(ast.Expression(n.value), "<c>", "eval"), {})  # noqa: S307
            except Exception:  # noqa: BLE001
                pass
    for name, val in consts.items():
        if found.get(name) != val:
            raise SystemExit(f"prompt {name} differs from official {pipeline}")
    print(f"verbatim check OK against {pipeline.name}: {', '.join(consts)}")


def verify_source_snippet(pipeline: Path, start: str, end: str, own_file: Path) -> None:
    """Assert that the official source between start..end markers appears in own_file."""
    if not pipeline.exists():
        return
    src = pipeline.read_text(encoding="utf-8")
    i = src.index(start)
    j = src.index(end, i + len(start))
    snippet = src[i:j + len(end)]
    if snippet not in own_file.read_text(encoding="utf-8"):
        raise SystemExit(f"official snippet starting {start!r} not found verbatim in {own_file.name}")
    print(f"verbatim check OK against {pipeline.name}: snippet {start[:40]!r}")


# ----------------------------------------------------------------------------- LLM

def chat(messages: list[dict], model: str, max_tokens: int | None = 512, json_mode: bool = False,
         retries: int = 5, timeout: int = 300) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    base = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    payload: dict = {"model": model, "messages": messages, "temperature": 0}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(base + "/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return (json.loads(r.read().decode())["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"llm failed: {last}")


# ----------------------------------------------------------------------------- ingestion

def chunk_messages(msgs: list[dict], max_msgs: int = 6, max_chars: int = 12_000) -> list[list[dict]]:
    """Split a long session into POST /add sized batches without splitting a message."""
    out: list[list[dict]] = []
    cur: list[dict] = []
    size = 0
    for m in msgs:
        n = len(str(m.get("content", "")))
        if cur and (len(cur) >= max_msgs or size + n > max_chars):
            out.append(cur)
            cur, size = [], 0
        cur.append(m)
        size += n
    if cur:
        out.append(cur)
    return out


def ingest(jobs: list[tuple[str, str, str, list[dict]]], done_path: Path, workers: int) -> None:
    """jobs: (request_id, user_id, session_id, messages). Resumable via done_path."""
    from app.database import init_db, save_memories_batch
    from app.core.extractor import extract_propositions

    init_db()
    le.jl_append(done_path.parent / "ingest_meta.jsonl",
                 {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_fp": code_fingerprint(),
                  "n_jobs": len(jobs)})
    done = {r["request_id"] for r in jl_read(done_path)}
    todo = [j for j in jobs if j[0] not in done]
    print(f"ingest: {len(todo)} requests pending ({len(done)} already done, {len(jobs)} total)", flush=True)

    def run(job):
        rid, uid, sid, msgs = job
        t0 = time.time()
        props = extract_propositions([dict(m) for m in msgs])  # identical to POST /add
        save_memories_batch(request_id=rid, user_id=uid, session_id=sid, raw_messages=[dict(m) for m in msgs],
                            propositions=props)
        dialogue_like = sum(1 for p in props if re.match(r"^(\[\d{4}-\d{2}-\d{2}[^\]]*\] )?(user|assistant): ", p or ""))
        return {"request_id": rid, "user_id": uid, "n_msgs": len(msgs), "n_props": len(props),
                "fallback_raw": bool(props) and dialogue_like == len(props), "sec": round(time.time() - t0, 2)}

    t_start = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run, j) for j in todo]
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                le.jl_append(done_path, f.result())
            except Exception as e:  # noqa: BLE001
                print("  ingest error:", e, flush=True)
            if i % 50 == 0:
                el = time.time() - t_start
                print(f"  {i}/{len(todo)}  {el/60:.1f} min elapsed, eta {el/i*(len(todo)-i)/60:.1f} min", flush=True)
    recs = jl_read(done_path)
    fb = sum(1 for r in recs if r.get("fallback_raw"))
    print(f"ingest done: {len(recs)} requests; extraction fell back to raw dialogue in {fb}", flush=True)
    backfill_embeddings()


def backfill_embeddings(passes: int = 6, batch: int = 64) -> None:
    """Fill embeddings that save_memories_batch should have written but lost to HTTP 429.

    Under the harness's concurrency DashScope may throttle the embedding endpoint and
    save_memories_batch's short retry can give up, leaving rows without vectors. The
    gap is reported, then filled with the same get_embeddings() call, sequentially.
    Follows the storage policy of the loaded app code (raw-turn embedding flag and
    vector encoding), so it never adds vectors the engine itself would not store.
    """
    import app.database as db
    from app.core.embeddings import get_embeddings

    embed_raw = bool(getattr(db, "EMBED_RAW_TURNS", False))
    encode = getattr(db, "encode_vector", None) or (lambda v: json.dumps(v))
    where = "1=1" if embed_raw else (
        "(m.item_type = 'proposition' OR m.request_id NOT IN "
        "(SELECT DISTINCT request_id FROM memories WHERE item_type = 'proposition'))")
    with db.get_db() as conn:
        expected = conn.execute(f"SELECT COUNT(*) FROM memories m WHERE {where}").fetchone()[0]
        missing = [(r[0], r[1]) for r in conn.execute(
            "SELECT m.memory_id, m.content FROM memories m LEFT JOIN embeddings e ON e.memory_id = m.memory_id "
            f"WHERE e.memory_id IS NULL AND {where}").fetchall()]
    print(f"embedding coverage after ingest: {expected - len(missing)}/{expected} "
          f"({len(missing)} missing; embed_raw_turns={embed_raw})", flush=True)
    for p in range(passes):
        if not missing:
            break
        still = []
        for i in range(0, len(missing), batch):
            part = missing[i:i + batch]
            vecs = get_embeddings([c for _, c in part])
            with db.get_db() as conn:
                with conn:
                    for (mid, c), v in zip(part, vecs):
                        if v:
                            conn.execute("INSERT OR REPLACE INTO embeddings (memory_id, vector) VALUES (?, ?)",
                                         (mid, encode(v)))
                        else:
                            still.append((mid, c))
        missing = still
        if missing:
            time.sleep(5 * (p + 1))
    print(f"embedding backfill done; still missing {len(missing)}", flush=True)


CODE_FILES = ("app/database.py", "app/core/extractor.py", "app/core/embeddings.py", "app/core/clean_search.py",
              "app/core/temporal.py", "app/core/bm25.py")


def code_fingerprint() -> str:
    """Short hash of the app files that shape ingestion and retrieval (app/ is edited concurrently)."""
    import hashlib

    h = hashlib.sha1()
    for f in CODE_FILES:
        p = ROOT / f
        h.update(f.encode())
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:10]


# ----------------------------------------------------------------------------- run helpers

RUN_CODE_FP = ""


def warm_imports(variant: str) -> None:
    global RUN_CODE_FP
    RUN_CODE_FP = code_fingerprint()
    print(f"app code fingerprint at run start: {RUN_CODE_FP}", flush=True)
    if variant == "legacy":
        import app.database  # noqa: F401
        import app.core.temporal  # noqa: F401
    else:
        import app.core.clean_search  # noqa: F401


def run_parallel(todo: list[dict], fn: Callable[[dict], dict], out_path: Path, workers: int) -> None:
    print(f"run: {len(todo)} pending -> {out_path}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, q): q for q in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                le.jl_append(out_path, {**f.result(), "code_fp": RUN_CODE_FP})
            except Exception as e:  # noqa: BLE001
                print(f"  error on {futs[f].get('id')}: {e}", flush=True)
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)


def items_text(items: list[dict]) -> str:
    return le.render_items(items)


# ----------------------------------------------------------------------------- stats / report

def mean_ci(xs: list[float], z: float = 1.96) -> tuple[float, float, float]:
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = sum(xs) / n
    if n == 1:
        return (m, m, m)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = z * sd / math.sqrt(n)
    return (m, max(0.0, m - h), min(1.0, m + h))


def report(rs: list[dict], title: str, cat_key: str = "category", graded: bool = False,
           out_json: Path | None = None) -> dict:
    """Binary: r['correct'] with Wilson CI. Graded: r['score'] in [0,1] with normal CI,
    plus Wilson CI on full credit (score == 1)."""
    if not rs:
        print("no results")
        return {}
    print(f"\n=== {title}, n={len(rs)} ===")
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rs:
        by[str(r[cat_key])].append(r)
    summary: dict = {"title": title, "n": len(rs), "categories": {}}

    def line(name: str, g: list[dict]) -> dict:
        items = sum(r.get("n_items", 0) for r in g) / len(g)
        cb = sum(r.get("context_bytes", 0) for r in g) / len(g)
        if graded:
            m, lo, hi = mean_ci([float(r["score"]) for r in g])
            k = sum(1 for r in g if float(r["score"]) >= 1.0)
            wlo, whi = le.wilson(k, len(g))
            print(f"  {name:28s} n={len(g):3d}  mean={m:6.1%} 95%CI [{lo:5.1%}, {hi:5.1%}]"
                  f"  full={k/len(g):6.1%} [{wlo:5.1%}, {whi:5.1%}]  avg_items={items:5.1f}  avg_ctx_bytes={cb:7,.0f}")
            return {"n": len(g), "mean": m, "ci": [lo, hi], "full_credit": k / len(g), "full_ci": [wlo, whi],
                    "avg_items": items, "avg_context_bytes": cb}
        k = sum(1 for r in g if r["correct"])
        lo, hi = le.wilson(k, len(g))
        print(f"  {name:28s} {k:3d}/{len(g):3d} = {k/len(g):6.1%}  95%CI [{lo:5.1%}, {hi:5.1%}]"
              f"  avg_items={items:5.1f}  avg_ctx_bytes={cb:7,.0f}")
        return {"n": len(g), "k": k, "acc": k / len(g), "ci": [lo, hi], "avg_items": items, "avg_context_bytes": cb}

    for cat in sorted(by):
        summary["categories"][cat] = line(cat, by[cat])
    summary["overall"] = line("OVERALL", rs)
    if out_json:
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------- failure taxonomy

TAXONOMY_PROMPT = """You are auditing a memory-augmented QA system. Classify why the answer below was graded as failing.

Question:
{question}

Gold / reference (answer or rubric):
{gold}

Retrieved memory context given to the answer model (possibly truncated):
<context>
{context}
</context>

Generated answer:
{generated}

Grader output:
{judge}

Choose exactly one label:
- retrieval_miss: the information needed for the gold answer is NOT present in the retrieved context.
- gold_present_answer_wrong: the needed information IS present in the context, but the generated answer is wrong, incomplete, or refuses.
- judge_strict: the generated answer is actually acceptable (correct in substance) and the grader was too strict or mis-parsed.
- other: unanswerable / ambiguous gold / format issue not covered above.

Return only JSON: {{"label": "...", "reason": "one sentence"}}"""

TAXONOMY_LABELS = ("retrieval_miss", "gold_present_answer_wrong", "judge_strict", "other")


def failure_taxonomy(rs: list[dict], is_wrong: Callable[[dict], bool], gold_of: Callable[[dict], str],
                     out_path: Path, model: str, n: int = 15, seed: int = 3, ctx_chars: int = 24_000) -> dict:
    wrong = [r for r in rs if is_wrong(r)]
    rng = random.Random(seed)
    rng.shuffle(wrong)
    sample = wrong[:n]
    done = {r["id"]: r for r in jl_read(out_path)}
    for r in sample:
        if r["id"] in done:
            continue
        ctx = r.get("context", "")
        if len(ctx) > ctx_chars:
            ctx = ctx[:ctx_chars] + "\n...[truncated]"
        prompt = TAXONOMY_PROMPT.format(question=str(r.get("question", ""))[-4000:], gold=gold_of(r)[:3000],
                                        context=ctx, generated=str(r.get("generated", ""))[:4000],
                                        judge=str(r.get("judge", ""))[:2000])
        resp = chat([{"role": "user", "content": prompt}], model, max_tokens=200, json_mode=True)
        try:
            lab = json.loads(re.search(r"\{.*\}", resp, re.DOTALL).group(0))
        except Exception:  # noqa: BLE001
            lab = {"label": "other", "reason": resp[:200]}
        if lab.get("label") not in TAXONOMY_LABELS:
            lab["label"] = "other"
        rec = {"id": r["id"], "category": r.get("category"), "label": lab["label"], "reason": lab.get("reason", "")}
        le.jl_append(out_path, rec)
        done[r["id"]] = rec
    recs = [done[r["id"]] for r in sample if r["id"] in done]
    counts = {k: sum(1 for x in recs if x["label"] == k) for k in TAXONOMY_LABELS}
    print(f"\nfailure taxonomy ({len(recs)} sampled of {len(wrong)} failing): {counts}")
    for x in recs:
        print(f"  [{x['label']}] {x['category']}: {x['reason'][:160]}")
    return {"n_failing": len(wrong), "n_sampled": len(recs), "counts": counts}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def iter_unique(xs: Iterable) -> list:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
