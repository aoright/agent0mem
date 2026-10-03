"""Generic hybrid retrieval with no content- or benchmark-specific rules.

Design:
  - Candidates: every memory row of the user (distilled propositions and raw turns).
  - Lexical: BM25 over all rows.
  - Dense: cosine similarity for rows that have embeddings.
  - Fusion: reciprocal rank fusion (RRF), which needs no score calibration.
  - Context: raw-turn hits are expanded with their neighbouring turns in the same session.
  - Output: up to `max_items` items within a byte budget, each carrying an explicit
    date so the answer model can resolve relative and absolute time.

What it deliberately does NOT do: keyword-triggered query expansion, hand-written
answers, per-dataset boosts, abstention notices, or hard caps of 1-3 items. The
answer model decides what is relevant; the memory system's job is high recall
with low noise.
"""
from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:  # optional acceleration; results are identical up to float rounding
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

from app.core.bm25 import BM25Index
from app.core.embeddings import cosine_similarity, decode_vector, get_embeddings

RRF_K = 60
DEFAULT_MAX_ITEMS = 40
DEFAULT_BYTE_BUDGET = 24_000
NEIGHBOUR_WINDOW = 1

_MARKER_RE = re.compile(r"\s*\[(?:Conversation Date|Year|Month|Date|Event Time|Relative Time)[^\]]*\]")


def _fmt_day(day: str, style: str) -> str:
    """Render YYYY-MM-DD either as-is or as '20 May 2023 (Saturday)'."""
    if style != "human" or not day:
        return day
    try:
        d = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return day
    return f"{d.day} {d.strftime('%B %Y')} ({d.strftime('%A')})"


def _day(ts: Optional[int], created_at: Optional[str]) -> str:
    if ts:
        t = float(ts)
        if t > 1e11:
            t /= 1000.0
        try:
            return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            pass
    if created_at:
        return str(created_at)[:10]
    return ""


def _clean(text: str) -> str:
    # Remove the engine's own bracketed enrichment markers; the date is re-added once, explicitly.
    return _MARKER_RE.sub("", text or "").strip()


_CODE_QUERY_RE = re.compile(
    r"(?:\b[\w./-]+\.(?:py|pyx|pxd|js|ts|tsx|java|go|rs|cpp|cc|c|h|rb|php|sh)\b"
    r"|Traceback \(most recent call last\)|diff --git|```)"
)
_CODING_NS_CACHE: Dict[str, bool] = {}


def is_coding_namespace(user_id: str, query: str) -> bool:
    """True when the request belongs to a software-engineering memory namespace.

    Decided from data, not from topic keywords: the user's stored memories contain
    code patches / diffs, or the query itself carries code artifacts.
    """
    if _CODE_QUERY_RE.search(query or ""):
        return True
    if user_id in _CODING_NS_CACHE:
        return _CODING_NS_CACHE[user_id]
    from app.database import get_db

    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM memories WHERE user_id = ? AND (content LIKE '%diff --git%' "
            "OR content LIKE '%[Code Patch / Solution]%' OR content LIKE '%+++ b/%') LIMIT 1",
            (user_id,),
        ).fetchone()
    result = row is not None
    if result:
        if len(_CODING_NS_CACHE) > 5000:
            _CODING_NS_CACHE.clear()
        _CODING_NS_CACHE[user_id] = True
    return result


def _load_rows(user_id: str) -> List[dict]:
    from app.database import get_db

    with get_db() as conn:
        cur = conn.execute(
            "SELECT memory_id, session_id, item_type, role, content, timestamp, created_at "
            "FROM memories WHERE user_id = ? ORDER BY timestamp ASC, id ASC",
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def _load_vectors(ids: List[str]) -> Dict[str, List[float]]:
    from app.database import get_db

    out: Dict[str, List[float]] = {}
    with get_db() as conn:
        for i in range(0, len(ids), 500):
            batch = ids[i:i + 500]
            ph = ",".join("?" * len(batch))
            for r in conn.execute(f"SELECT memory_id, vector FROM embeddings WHERE memory_id IN ({ph})", batch):
                v = decode_vector(r["vector"])
                if v:
                    out[r["memory_id"]] = v
    return out


_INDEX_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_INDEX_CACHE_MAX = 64
_INDEX_LOCK = threading.Lock()


def _user_key(user_id: str) -> tuple:
    from app.database import get_db

    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*), MAX(id) FROM memories WHERE user_id = ?", (user_id,)).fetchone()
    return (row[0], row[1])


def _build_index(user_id: str) -> tuple:
    rows = _load_rows(user_id)
    by_id = {r["memory_id"]: r for r in rows}
    bm25 = BM25Index()
    bm25.fit([(r["memory_id"], _clean(r["content"])) for r in rows])
    vecs = _load_vectors(list(by_id)) if rows else {}
    ids = list(vecs)
    mat = None
    if ids and _np is not None:
        m = _np.asarray([vecs[i] for i in ids], dtype=_np.float32)
        norms = _np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = m / norms
    return rows, by_id, bm25, ids, mat, vecs


def _get_index(user_id: str) -> tuple:
    """Cached per-user index. The key changes on every insert, so a search right after
    a successful /add always sees the new memories."""
    key = _user_key(user_id)
    with _INDEX_LOCK:
        hit = _INDEX_CACHE.get(user_id)
        if hit and hit[0] == key:
            _INDEX_CACHE.move_to_end(user_id)
            return hit[1]
    idx = _build_index(user_id)
    with _INDEX_LOCK:
        _INDEX_CACHE[user_id] = (key, idx)
        _INDEX_CACHE.move_to_end(user_id)
        while len(_INDEX_CACHE) > _INDEX_CACHE_MAX:
            _INDEX_CACHE.popitem(last=False)
    return idx


def clean_search(user_id: str, query: str, top_k: int = 100,
                 max_items: int = DEFAULT_MAX_ITEMS, byte_budget: int = DEFAULT_BYTE_BUDGET,
                 order: str = "score", date_style: str = "iso") -> List[dict]:
    """order: 'score' (fused relevance) or 'time' (selected items re-sorted chronologically).
    date_style: 'iso' -> '[2023-05-20]', 'human' -> '[Said on 20 May 2023 (Saturday)]'."""
    rows, by_id, bm25, vec_ids, vec_mat, vecs = _get_index(user_id)
    if not rows:
        return []

    # Lexical ranking
    lex = [mid for mid, s in sorted(bm25.score(query), key=lambda x: x[1], reverse=True) if s > 0]

    # Dense ranking
    dense: List[str] = []
    qv = (get_embeddings([query]) or [None])[0] if vec_ids else None
    if qv:
        if vec_mat is not None:
            q = _np.asarray(qv, dtype=_np.float32)
            qn = float(_np.linalg.norm(q)) or 1.0
            sims = vec_mat @ (q / qn)
            order_idx = _np.argsort(-sims, kind="stable")
            dense = [vec_ids[i] for i in order_idx]
        else:
            sims_l = [(mid, cosine_similarity(qv, vecs[mid])) for mid in vec_ids]
            dense = [mid for mid, _ in sorted(sims_l, key=lambda x: x[1], reverse=True)]

    # Reciprocal rank fusion
    fused: Dict[str, float] = {}
    for ranking in (lex[:300], dense[:300]):
        for rank, mid in enumerate(ranking):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # Neighbour index for raw-turn context expansion: raw ids look like raw_<request>_<idx>
    def neighbours(mid: str) -> List[str]:
        m = re.match(r"^(raw_.+)_(\d+)$", mid)
        if not m:
            return []
        prefix, idx = m.group(1), int(m.group(2))
        return [f"{prefix}_{j}" for j in range(idx - NEIGHBOUR_WINDOW, idx + NEIGHBOUR_WINDOW + 1) if j != idx]

    limit = max(1, min(top_k, max_items))
    out: List[dict] = []
    used = set()
    seen_text = set()
    total = 0
    for mid, score in ranked:
        if len(out) >= limit:
            break
        if mid in used:
            continue
        r = by_id[mid]
        parts = []
        if r["item_type"] == "raw":
            for nid in [*neighbours(mid)[:NEIGHBOUR_WINDOW], mid, *neighbours(mid)[NEIGHBOUR_WINDOW:]]:
                n = by_id.get(nid)
                if n is not None and nid not in used:
                    parts.append(_clean(n["content"]))
                    used.add(nid)
        else:
            parts.append(_clean(r["content"]))
            used.add(mid)
        body = "\n".join(p for p in parts if p)
        key = re.sub(r"\W+", "", body.lower())[:300]
        if not body or key in seen_text:
            continue
        seen_text.add(key)
        day = _day(r["timestamp"], r["created_at"])
        if day and date_style == "human":
            content = f"[Said on {_fmt_day(day, 'human')}] {body}"
        else:
            content = f"[{day}] {body}" if day else body
        size = len(content.encode("utf-8"))
        if total + size > byte_budget and out:
            break
        total += size
        out.append({
            "id": mid,
            "content": content,
            "text": content,
            "score": round(score, 6),
            "created_at": r["created_at"],
            "timestamp": r["timestamp"],
        })
    if order == "time":
        out.sort(key=lambda it: (float(it.get("timestamp") or 0), it["id"]))
    return out
