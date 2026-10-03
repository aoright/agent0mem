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
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.core.bm25 import BM25Index
from app.core.embeddings import cosine_similarity, get_embeddings

RRF_K = 60
DEFAULT_MAX_ITEMS = 40
DEFAULT_BYTE_BUDGET = 24_000
NEIGHBOUR_WINDOW = 1

_MARKER_RE = re.compile(r"\s*\[(?:Conversation Date|Year|Month|Date|Event Time|Relative Time)[^\]]*\]")


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
                if r["vector"]:
                    try:
                        out[r["memory_id"]] = json.loads(r["vector"])
                    except ValueError:
                        pass
    return out


def clean_search(user_id: str, query: str, top_k: int = 100,
                 max_items: int = DEFAULT_MAX_ITEMS, byte_budget: int = DEFAULT_BYTE_BUDGET) -> List[dict]:
    rows = _load_rows(user_id)
    if not rows:
        return []
    by_id = {r["memory_id"]: r for r in rows}

    # Lexical ranking
    bm25 = BM25Index()
    bm25.fit([(r["memory_id"], _clean(r["content"])) for r in rows])
    lex = [mid for mid, s in sorted(bm25.score(query), key=lambda x: x[1], reverse=True) if s > 0]

    # Dense ranking
    dense: List[str] = []
    qv = (get_embeddings([query]) or [None])[0]
    if qv:
        vecs = _load_vectors(list(by_id))
        sims = [(mid, cosine_similarity(qv, v)) for mid, v in vecs.items()]
        dense = [mid for mid, _ in sorted(sims, key=lambda x: x[1], reverse=True)]

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
    return out
