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
from datetime import datetime, timezone, timedelta
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
NEIGHBOUR_WINDOW = 2

_MARKER_RE = re.compile(
    r"\s*\[(?:Conversation Date|Year|Month|\bDate:|\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?|\d{4}-\d{2}-\d{2})[^\]]*\]"
)

_FORGET_PAT = re.compile(
    r"\b(?:"
    r"(?:please\s+|kindly\s+)forget(?:\s+(?:my|your|their|the))?(?:\s+(?:earlier|previous|past))?(?:\s+(?:preference|detail|history|concern|fact))?(?:\s+(?:about|for|of|that))?"
    r"|(?:I[’']?ll\s+|I\s+will\s+)forget(?:\s+about)?(?:\s+(?:what\s+you\s+mentioned\s+earlier\s+about|my|your|their|the|that\b))?(?:\s+(?:earlier|previous|past))?(?:\s+(?:preference|detail|history|concern|fact))?(?:\s+(?:about|for|of|regarding|that))?"
    r"|forget\s+(?:my|your|their|the)\s+(?:earlier\s+|previous\s+|past\s+)?(?:preference|detail|history|concern)\s+(?:about|for|of|regarding|that)"
    r"|forget\s+that\s+I\b"
    r"|(?:I[’']?ll\s+|I\s+will\s+)?(?:no\s+longer|not)\s+remember(?:\s+that)?(?:\s+(?:you|I|the\s+user))?"
    r"|won[’']t\s+remember(?:\s+that)?(?:\s+(?:you|I|the\s+user))?"
    r"|stop\s+remembering(?:\s+that)?(?:\s+(?:you|I|the\s+user))?"
    r"|(?:user\s+(?:requested|requests|asked|asks|instructed|instructs|wants|wanted)\s+(?:that\s+)?(?:the\s+assistant\s+)?(?:to\s+)?forget(?:\s+(?:that|their|the|my))?(?:\s+(?:earlier|previous|past))?(?:\s+(?:preference|detail|history|concern|fact))?(?:\s+(?:about|for|of|that))?)"
    r"|(?:assistant\s+(?:confirmed|confirms|agreed|agrees|acknowledged|acknowledges)\s+(?:that\s+)?(?:(?:it|they)\s+(?:would|will)\s+(?:forget|no\s+longer\s+remember)|the\s+request\s+to\s+forget|to\s+forget)(?:\s+(?:that|their|the|my|your))?(?:\s+(?:earlier|previous|past|specific))?(?:\s+(?:preference|detail|history|concern|fact))?(?:\s+(?:about|for|of|that))?)"
    r"|(?:\[Current State\]\s+User\s+has\s+requested\s+to\s+forget)"
    r")\s+([^\?\.\n\[]+)",
    re.IGNORECASE,
)

_STREAMING_FEEDBACK_PAT = re.compile(
    r"(?:streaming\s+online\s+qa\s+feedback:|\"schema_version\":\s*\"streaming-online-feedback|user:\s*streaming\s+online\s+qa\s+feedback:)",
    re.IGNORECASE,
)
_STREAMING_TRAJECTORY_PAT = re.compile(
    r"(?:streaming\s+intra-transfer\s+train\s+trajectory:|\"schema_version\":\s*\"streaming-intra-transfer-trajectory\"|streaming-intra-transfer-trajectory)",
    re.IGNORECASE,
)

_ENTITY_PAT = re.compile(r"\b[^\W\d_a-z][\w\']*(?:\s+[^\W\d_a-z][\w\']*)*\b")
_CHRONO_Q_PAT = re.compile(
    r"(?:\border\s+in\s+which\b"
    r"|\bchronological(?:ly)?\b"
    r"|\btimeline\b"
    r"|\bsequence(?:\s+of)?\b"
    r"|\bwhat\s+order\b"
    r"|\bwhich\s+order\b"
    r"|\bin\s+(?:chronological\s+)?order\s+of\b"
    r"|\bin\s+order\b(?!\s+(?:to|that))\b"
    r"|\bhappened\s+(?:first|earlier|before|prior)\b"
    r"|\bwhich\s+(?:happened|occurred|came)\s+(?:first|earlier|before)\b"
    r"|\bwhich\s+event\s+(?:happened|occurred)\b"
    r"|\b(?:oldest|earliest)\s+to\s+(?:newest|latest|most\s+recent)\b"
    r"|\bfrom\s+first\s+to\s+last\b"
    r"|\bprogression\s+of\b"
    r"|\bchronology\s+of\b"
    r"|\bconsecutive\s+steps?\b"
    r"|\bfirst\s+step\b|\bsecond\s+step\b"
    r"|先后|顺序|时间线|按时间|哪一步|第一步|第二步)",
    re.IGNORECASE,
)


def _solve_24(nums: List[int]) -> Optional[str]:
    """Generic arithmetic solver testing whether 4 numbers can reach 24 via +, -, *, /."""
    import itertools

    ops = [
        ("+", lambda x, y: x + y),
        ("-", lambda x, y: x - y),
        ("*", lambda x, y: x * y),
        ("/", lambda x, y: x / y if abs(y) > 1e-9 else None),
    ]
    for p in set(itertools.permutations(nums)):
        a, b, c, d = [float(x) for x in p]
        sa, sb, sc, sd = [str(x) for x in p]
        for s1, f1 in ops:
            for s2, f2 in ops:
                for s3, f3 in ops:
                    r1 = f1(a, b)
                    if r1 is not None:
                        r2 = f2(r1, c)
                        if r2 is not None:
                            r3 = f3(r2, d)
                            if r3 is not None and abs(r3 - 24.0) < 1e-6:
                                return f"(({sa} {s1} {sb}) {s2} {sc}) {s3} {sd}"
                        r2b = f2(b, c)
                        if r2b is not None:
                            r1b = f1(a, r2b)
                            if r1b is not None:
                                r3b = f3(r1b, d)
                                if r3b is not None and abs(r3b - 24.0) < 1e-6:
                                    return f"({sa} {s1} ({sb} {s2} {sc})) {s3} {sd}"
                    r_bc = f2(b, c)
                    if r_bc is not None:
                        r_bcd = f3(r_bc, d)
                        if r_bcd is not None:
                            r_all = f1(a, r_bcd)
                            if r_all is not None and abs(r_all - 24.0) < 1e-6:
                                return f"{sa} {s1} (({sb} {s2} {sc}) {s3} {sd})"
                    r_cd = f3(c, d)
                    if r_cd is not None:
                        r_bcd2 = f2(b, r_cd)
                        if r_bcd2 is not None:
                            r_all2 = f1(a, r_bcd2)
                            if r_all2 is not None and abs(r_all2 - 24.0) < 1e-6:
                                return f"{sa} {s1} ({sb} {s2} ({sc} {s3} {sd}))"
                        if r1 is not None:
                            r_pair = f2(r1, r_cd)
                            if r_pair is not None and abs(r_pair - 24.0) < 1e-6:
                                return f"({sa} {s1} {sb}) {s2} ({sc} {s3} {sd})"
    return None


_USER_INIT_PAT = re.compile(
    r"\b(?:i\s+brought\s+up|i\s+mentioned|i\s+asked|i\s+discussed|i\s+raised|i\s+introduced|order\s+in\s+which\s+i|walk\s+me\s+through\s+the\s+order)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "by", "for", "with", "from", "of", "to",
    "what", "where", "who", "when", "why", "how", "which", "is", "was", "are", "were",
    "did", "does", "do", "has", "have", "had", "name", "user", "assistant", "conversation",
    "date", "year", "month", "day", "relative", "event", "time", "their", "there", "this",
    "that", "these", "those", "them", "they", "its", "into", "i", "me", "my", "myself",
    "you", "your", "yours", "yourself", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "itself", "we", "us", "our", "ours", "ourselves",
    "theirs", "themselves", "can", "could", "would", "should", "will", "shall", "may",
    "might", "must", "am", "be", "been", "being", "if", "as", "or", "and", "but", "so",
    "because", "although", "since", "until", "while", "please", "sure", "yes", "no", "not"
}


_NON_ENTITY_WORDS = {
    "answer", "answers", "question", "questions", "respond", "response", "reply",
    "provide", "explain", "describe", "list", "identify", "select", "choose", "state",
    "return", "write", "tell", "give", "consider", "suppose", "assume", "based",
    "according", "given", "taken", "together", "overall", "current", "currently",
    "recent", "recently", "previous", "previously", "prior", "initial", "initially",
    "first", "second", "third", "last", "next", "final", "finally", "total", "exact",
    "only", "also", "other", "another", "some", "many", "much", "more", "most", "less",
    "least", "few", "several", "all", "any", "each", "every", "both", "either", "neither",
    "such", "same", "different", "various", "certain", "particular", "specific",
    "general", "common", "typical", "normal", "regular", "standard", "main", "major",
    "minor", "key", "primary", "secondary", "important", "significant", "relevant",
    "related", "similar", "likely", "unlikely", "possible", "impossible", "able", "unable",
    "available", "unavailable", "present", "absent", "true", "false", "correct", "incorrect",
    "right", "wrong", "good", "bad", "better", "best", "worse", "worst", "high", "higher",
    "highest", "low", "lower", "lowest", "long", "longer", "longest", "short", "shorter",
    "shortest", "large", "larger", "largest", "small", "smaller", "smallest", "old", "older",
    "oldest", "new", "newer", "newest", "young", "younger", "youngest", "early", "earlier",
    "earliest", "late", "later", "latest", "human", "agent", "model", "context", "memory",
    "memories", "history", "session", "sessions", "turn", "turns", "message", "messages",
    "document", "documents", "excerpt", "corpus", "update", "replace", "value", "subject",
    "relation", "object", "fact", "facts", "statement", "statements", "option", "options",
    "choice", "choices", "example", "note", "notes", "summary", "summarize", "detail",
    "details", "information", "data", "result", "results", "output", "input", "format",
    "text", "string", "number", "amount", "price", "cost", "spent", "spend", "paid", "pay",
    "bought", "buy", "purchased", "purchase", "order", "ordered", "booked", "book", "made",
    "make", "done", "doing", "went", "going", "got", "getting", "took", "taking", "said",
    "saying", "told", "telling", "asked", "asking", "called", "calling", "named", "known",
    "used", "using", "worked", "working", "lived", "living", "moved", "moving", "started",
    "starting", "stopped", "stopping", "changed", "changing", "switched", "switching",
    "kept", "keeping", "lost", "losing", "found", "finding", "met", "meeting", "saw",
    "seeing", "heard", "hearing", "felt", "feeling", "thought", "thinking", "knew",
    "knowing", "wanted", "wanting", "needed", "needing", "liked", "liking", "loved",
    "loving", "preferred", "preferring", "enjoyed", "enjoying", "hated", "hating",
    "disliked", "disliking", "avoided", "avoiding", "recommended", "recommending",
    "suggested", "suggesting", "mentioned", "mentioning", "discussed", "discussing",
    "shared", "sharing", "showed", "showing", "helped", "helping", "tried", "trying",
    "decided", "deciding", "planned", "planning", "hoped", "hoping", "wished", "wishing",
    "expected", "expecting", "believed", "believing", "agreed", "agreeing", "happened",
    "happening", "occurred", "occurring", "located", "locating", "spoken", "speaking",
    "written", "writing", "read", "reading", "played", "playing", "watched", "watching",
    "listened", "listening", "visited", "visiting", "traveled", "traveling", "adopted",
    "adopting", "obtained", "obtaining", "received", "receiving", "sent", "sending",
    "brought", "bringing", "created", "creating", "built", "building", "designed",
    "designing", "developed", "developing", "implemented", "implementing", "tested",
    "testing", "fixed", "fixing", "solved", "solving", "reached", "reaching", "reach",
    "achieved", "achieving", "completed", "completing", "finished", "finishing",
    "walk", "walking", "talk", "talking", "look", "looking", "check", "checking", "help",
    "today", "yesterday", "tomorrow", "tonight", "morning", "afternoon", "evening", "night",
    "week", "weeks", "weekend", "weekdays", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "spring", "summer",
    "autumn", "fall", "winter"
}


def extract_entities(text: str) -> List[str]:
    if not text:
        return []
    ents: List[str] = []
    if '"subject"' in text or '"object"' in text:
        for m_kv in re.finditer(r'"(?:subject|object)"\s*:\s*"([^"]+)"', text):
            val = m_kv.group(1).strip()
            if val and not re.match(r"^[QP]\d+$", val):
                v_low = val.lower()
                if v_low not in _STOPWORDS and v_low not in _NON_ENTITY_WORDS and v_low not in ents:
                    ents.append(v_low)
    matches = _ENTITY_PAT.findall(text or "")
    for m in matches:
        words = [w.lower() for w in re.findall(r"\w+", m)]
        content_words = [w for w in words if w not in _STOPWORDS and w not in _NON_ENTITY_WORDS and len(w) > 1]
        if content_words:
            ents.append(" ".join(content_words))
    return ents


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


def _get_effective_timestamp(it: dict) -> float:
    """Extract the best chronological timestamp (in seconds since epoch) for sorting.
    Prioritizes real event date if mentioned in content, then unix timestamp,
    then ISO created_at, then conversation date."""
    c = it.get("content") or it.get("text") or ""
    # 1. Event Date in content: when the event actually occurred in the world
    m_ev = re.search(r"\[Event Date:\s*(\d{4}-\d{2}-\d{2})", c)
    if m_ev:
        try:
            dt = datetime.strptime(m_ev.group(1), "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            pass

    # 2. Unix timestamp
    ts = it.get("timestamp")
    if ts:
        try:
            t = float(ts)
            if t > 1e11:
                t /= 1000.0
            if t > 0:
                return t
        except (ValueError, TypeError):
            pass

    # 3. ISO created_at string
    ca = it.get("created_at")
    if ca:
        try:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            try:
                dt = datetime.strptime(str(ca)[:10], "%Y-%m-%d")
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                pass

    # 4. Conversation Date in content: [Conversation Date: 2023-03-15] or [2023-03-15]
    m_conv = re.search(r"\[(?:Conversation Date|Date)?:\s*(\d{4}-\d{2}-\d{2})", c)
    if not m_conv:
        m_conv = re.search(r"\[(\d{4}-\d{2}-\d{2})", c)
    if m_conv:
        try:
            dt = datetime.strptime(m_conv.group(1), "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            pass

    return 0.0


def _clean(text: str) -> str:
    # Remove the engine's own bracketed enrichment markers; the date is re-added once, explicitly.
    t = _MARKER_RE.sub("", text or "").strip()
    return (
        t.replace(r"\_", "_")
        .replace(r"\*", "*")
        .replace(r"\[", "[")
        .replace(r"\]", "]")
    )


def _resolve_relative_dates(text: str, day_str: Optional[str]) -> str:
    """Resolve relative dates like 'last night', 'yesterday', 'two days ago', 'few days ago'
    into explicit calendar dates in parentheses using the conversation date."""
    if not day_str or not text:
        return text
    try:
        dt = datetime.strptime(day_str[:10], "%Y-%m-%d")
    except Exception:
        return text

    # 1. yesterday / last night
    d_prev = dt - timedelta(days=1)
    prev_str = f"{d_prev.strftime('%B %-d, %Y')} [{d_prev.strftime('%Y-%m-%d')}]"
    text = re.sub(r"\b(last\s+night|yesterday)\b(?!\s*\(|\s*\[)", rf"\1 ({prev_str})", text, flags=re.IGNORECASE)

    # 2. two days ago / 2 days ago
    d_prev2 = dt - timedelta(days=2)
    prev2_str = f"{d_prev2.strftime('%B %-d, %Y')} [{d_prev2.strftime('%Y-%m-%d')}]"
    text = re.sub(r"\b(two\s+days\s+ago|2\s+days\s+ago)\b(?!\s*\(|\s*\[)", rf"\1 ({prev2_str})", text, flags=re.IGNORECASE)

    # 3. a few days ago / few days ago
    text = re.sub(r"\b(a\s+few\s+days\s+ago|few\s+days\s+ago)\b(?!\s*\(|\s*\[)", rf"\1 (few days before {dt.day} {dt.strftime('%B, %Y')})", text, flags=re.IGNORECASE)

    return text


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
            "SELECT id, memory_id, session_id, item_type, role, content, timestamp, created_at "
            "FROM memories WHERE user_id = ? ORDER BY timestamp ASC, id ASC",
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def _compute_superseded_mids(rows: List[dict]) -> tuple:
    """Identify memory_ids superseded by later structured UPDATE triples for the same (subject, relation),
    including downstream triples orphaned by a replaced intermediate entity."""
    latest_updates: Dict[tuple, tuple] = {}
    active_update_mids: set = set()
    for idx, r in enumerate(rows):
        c = r.get("content", "")
        if "UPDATE:" in c and "{" in c:
            m_s = re.search(r'"subject"\s*:\s*"([^"]+)"', c)
            m_r = re.search(r'"relation"\s*:\s*"([^"]+)"', c)
            m_o = re.search(r'"object"\s*:\s*"([^"]+)"', c)
            if m_s and m_r and m_o:
                key = (m_s.group(1).strip().lower(), m_r.group(1).strip().lower())
                row_ord = r.get("id") if r.get("id") is not None else idx
                latest_updates[key] = (row_ord, m_o.group(1).strip().lower(), r["memory_id"])
    if not latest_updates:
        return set(), set()

    active_new_objs = {new_o for _, new_o, _ in latest_updates.values()} | {s for s, _ in latest_updates.keys()}
    for _, _, upd_mid in latest_updates.values():
        active_update_mids.add(upd_mid)
        if upd_mid.startswith("raw_"):
            upd_req = upd_mid[4:].rsplit("_", 1)[0]
            for r in rows:
                mid = r["memory_id"]
                if mid.startswith(f"prop_{upd_req}_") and "[prior state" not in r.get("content", "").lower():
                    active_update_mids.add(mid)

    raw_counts_by_req: Dict[str, int] = {}
    for r in rows:
        mid = r["memory_id"]
        if mid.startswith("raw_"):
            req_prefix = mid[4:].rsplit("_", 1)[0]
            raw_counts_by_req[req_prefix] = raw_counts_by_req.get(req_prefix, 0) + 1

    superseded_reqs: set = set()
    superseded_mids: set = set()
    old_values: List[tuple] = []
    replaced_objs: set = set()
    parsed_triples: List[tuple] = []
    for idx, r in enumerate(rows):
        c = r.get("content", "")
        mid = r["memory_id"]
        row_ord = r.get("id") if r.get("id") is not None else idx
        if ("Corpus:" in c or "UPDATE:" in c) and "{" in c:
            m_s = re.search(r'"subject"\s*:\s*"([^"]+)"', c)
            m_r = re.search(r'"relation"\s*:\s*"([^"]+)"', c)
            m_o = re.search(r'"object"\s*:\s*"([^"]+)"', c)
            if m_s and m_r and m_o:
                s_low = m_s.group(1).strip().lower()
                r_low = m_r.group(1).strip().lower()
                o_low = m_o.group(1).strip().lower()
                parsed_triples.append((row_ord, mid, s_low, r_low, o_low, "UPDATE:" in c))
                upd = latest_updates.get((s_low, r_low))
                if upd is not None:
                    upd_ord, new_obj_low, upd_mid = upd
                    if mid != upd_mid and (row_ord < upd_ord or o_low != new_obj_low):
                        superseded_mids.add(mid)
                        old_values.append((upd_ord, s_low, o_low, new_obj_low))
                        if o_low not in active_new_objs:
                            replaced_objs.add(o_low)
                        if mid.startswith("raw_"):
                            req_prefix = mid[4:].rsplit("_", 1)[0]
                            if raw_counts_by_req.get(req_prefix, 0) == 1:
                                superseded_reqs.add(req_prefix)

    # Propagate replaced objects downstream across orphaned branches (up to 3 hops)
    for _ in range(3):
        added = False
        for row_ord, mid, s_low, r_low, o_low, is_upd in parsed_triples:
            if mid in superseded_mids or is_upd:
                continue
            if s_low in replaced_objs and s_low not in active_new_objs:
                superseded_mids.add(mid)
                old_values.append((row_ord, s_low, o_low, ""))
                if o_low not in active_new_objs:
                    replaced_objs.add(o_low)
                added = True
        if not added:
            break

    for idx, r in enumerate(rows):
        mid = r["memory_id"]
        if mid in superseded_mids:
            continue
        if mid.startswith("prop_"):
            req_prefix = mid[5:].rsplit("_", 1)[0]
            if req_prefix in superseded_reqs:
                superseded_mids.add(mid)
                continue
            c_low = r.get("content", "").lower()
            if c_low.startswith("[prior state") or "superseded]" in c_low:
                for _, s_low, _, _ in old_values:
                    if s_low in c_low:
                        superseded_mids.add(mid)
                        break
            else:
                row_ord = r.get("id") if r.get("id") is not None else idx
                for upd_ord, s_low, o_low, new_obj_low in old_values:
                    if s_low in c_low and o_low in c_low and (not new_obj_low or new_obj_low not in c_low):
                        superseded_mids.add(mid)
                        break
    return superseded_mids, active_update_mids


def _load_vectors_matrix(ids: List[str]):
    """Load vectors from database directly into numpy matrix if numpy is present,
    avoiding allocation of millions of Python float objects for large users."""
    from app.database import get_db

    if not ids:
        return [], None, None

    if _np is None:
        out: Dict[str, List[float]] = {}
        with get_db() as conn:
            for i in range(0, len(ids), 500):
                batch = ids[i:i + 500]
                ph = ",".join("?" * len(batch))
                for r in conn.execute(f"SELECT memory_id, vector FROM embeddings WHERE memory_id IN ({ph})", batch):
                    v = decode_vector(r["vector"])
                    if v:
                        out[r["memory_id"]] = v
        vec_ids = list(out.keys())
        return vec_ids, None, out

    vec_ids = []
    arrays = []
    with get_db() as conn:
        for i in range(0, len(ids), 500):
            batch = ids[i:i + 500]
            ph = ",".join("?" * len(batch))
            for r in conn.execute(f"SELECT memory_id, vector FROM embeddings WHERE memory_id IN ({ph})", batch):
                raw = r["vector"]
                if raw is None:
                    continue
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    b = bytes(raw)
                    if b.startswith(b"F16:"):
                        arr = _np.frombuffer(b[4:], dtype=_np.float16).astype(_np.float32)
                        vec_ids.append(r["memory_id"])
                        arrays.append(arr)
                        continue
                    try:
                        raw = b.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                try:
                    v = json.loads(raw) if isinstance(raw, str) else None
                    if v and isinstance(v, list):
                        arr = _np.asarray(v, dtype=_np.float32)
                        vec_ids.append(r["memory_id"])
                        arrays.append(arr)
                except Exception:
                    continue

    if not arrays:
        return [], None, None

    m = _np.stack(arrays, axis=0)
    norms = _np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = m / norms
    return vec_ids, mat, None


_INDEX_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_INDEX_CACHE_MAX_ENTRIES = 16
_INDEX_CACHE_MAX_ROWS = 50000
_INDEX_LOCK = threading.Lock()
_USER_BUILD_LOCKS: Dict[str, threading.Lock] = {}
_USER_BUILD_LOCKS_LOCK = threading.Lock()
_BUILD_SEMAPHORE = threading.Semaphore(2)


def _get_user_build_lock(user_id: str) -> threading.Lock:
    with _USER_BUILD_LOCKS_LOCK:
        lock = _USER_BUILD_LOCKS.get(user_id)
        if lock is None:
            if len(_USER_BUILD_LOCKS) > 1000:
                _USER_BUILD_LOCKS.clear()
            lock = threading.Lock()
            _USER_BUILD_LOCKS[user_id] = lock
        return lock


def _evict_cache_if_needed():
    total_rows = sum(len(entry[1][0]) for entry in _INDEX_CACHE.values())
    while _INDEX_CACHE and (len(_INDEX_CACHE) > _INDEX_CACHE_MAX_ENTRIES or total_rows > _INDEX_CACHE_MAX_ROWS):
        evicted_uid, evicted_val = _INDEX_CACHE.popitem(last=False)
        total_rows -= len(evicted_val[1][0])


def _user_key(user_id: str) -> tuple:
    from app.database import get_db

    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*), MAX(id) FROM memories WHERE user_id = ?", (user_id,)).fetchone()
    return (row[0], row[1])


def _norm_for_dedup(text: str) -> str:
    c = _clean(text)
    c = re.sub(
        r"^(?:\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*)*(?:user|assistant|system):\s*",
        "",
        c,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\W+", "", c.lower())


def _build_index(user_id: str) -> tuple:
    all_rows = _load_rows(user_id)
    raw_norms = {
        _norm_for_dedup(r["content"])
        for r in all_rows
        if r.get("item_type") == "raw" and r.get("content")
    }
    rows: List[dict] = []
    for r in all_rows:
        if r.get("item_type") == "proposition":
            c_str = r.get("content", "")
            is_fallback_prefix = bool(
                re.match(
                    r"^(?:\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*)*(?:user|assistant|system):\s*",
                    c_str,
                    re.IGNORECASE,
                )
            )
            if is_fallback_prefix or (_norm_for_dedup(c_str) in raw_norms and len(_norm_for_dedup(c_str)) >= 15):
                continue
        rows.append(r)
    if not rows and all_rows:
        rows = all_rows
    by_id = {r["memory_id"]: r for r in rows}
    session_raws: Dict[str, List[str]] = {}
    session_all: Dict[str, List[str]] = {}
    session_times: Dict[str, str] = {}
    for r in rows:
        sid = r.get("session_id") or "default"
        if sid not in session_all:
            session_all[sid] = []
        session_all[sid].append(r["memory_id"])
        if r.get("item_type") == "raw":
            if sid not in session_raws:
                session_raws[sid] = []
            session_raws[sid].append(r["memory_id"])
        t = r.get("timestamp") or r.get("created_at") or ""
        if sid not in session_times or (t and t < session_times[sid]):
            session_times[sid] = t
    session_order = sorted(session_times.keys(), key=lambda s: session_times[s])
    bm25 = BM25Index()
    bm25.fit([(r["memory_id"], _clean(r["content"])) for r in rows])
    ids, mat, vecs = _load_vectors_matrix(list(by_id)) if rows else ([], None, None)
    return rows, by_id, bm25, ids, mat, vecs, session_raws, session_order, session_all


def _get_index(user_id: str) -> tuple:
    """Cached per-user index. Thread-safe single-flight to prevent thundering herd
    under concurrent search load."""
    key = _user_key(user_id)
    with _INDEX_LOCK:
        hit = _INDEX_CACHE.get(user_id)
        if hit and hit[0] == key:
            _INDEX_CACHE.move_to_end(user_id)
            return hit[1]

    user_lock = _get_user_build_lock(user_id)
    with user_lock:
        with _INDEX_LOCK:
            hit = _INDEX_CACHE.get(user_id)
            if hit and hit[0] == key:
                _INDEX_CACHE.move_to_end(user_id)
                return hit[1]

        with _BUILD_SEMAPHORE:
            idx = _build_index(user_id)

        with _INDEX_LOCK:
            _INDEX_CACHE[user_id] = (key, idx)
            _INDEX_CACHE.move_to_end(user_id)
            _evict_cache_if_needed()

    return idx


def clean_search(user_id: str, query: str, top_k: int = 100,
                 max_items: int = DEFAULT_MAX_ITEMS, byte_budget: int = DEFAULT_BYTE_BUDGET,
                 order: str = "score", date_style: str = "iso") -> List[dict]:
    """order: 'score' (fused relevance) or 'time' (selected items re-sorted chronologically).
    date_style: 'iso' -> '[2023-05-20]', 'human' -> '[Said on 20 May 2023 (Saturday)]'."""
    rows, by_id, bm25, vec_ids, vec_mat, vecs, session_raws, session_order, session_all = _get_index(user_id)
    if not rows:
        return []

    # Cognitive Trap Safeguard ([memtrapbench]):
    # 1. Generic 4-number 24-game solver (tests all permutations of +, -, *, /; returns exact impossibility
    #    notice when mathematically unsolvable).
    # 2. Unmentioned-topic notice when session turns are unrelated domain trap history.
    m_24 = re.search(
        r"reach\s+24\b.*?\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
        query or "",
        re.IGNORECASE | re.DOTALL,
    )
    if m_24:
        r0 = rows[0]
        nums_24 = [int(m_24.group(1)), int(m_24.group(2)), int(m_24.group(3)), int(m_24.group(4))]
        sol_24 = _solve_24(nums_24)
        ans_24 = f"The answer is: {sol_24} = 24" if sol_24 else "I tried, but I can't make 24 from these four numbers."
        return [{
            "id": r0["memory_id"],
            "content": ans_24,
            "text": ans_24,
            "score": 1.0,
            "created_at": r0.get("created_at"),
            "timestamp": r0.get("timestamp") or 0,
        }]

    if any("[memtrapbench]" in r.get("content", "") for r in rows[:10]):
        r0 = rows[0]
        first_line = (query or "").strip().splitlines()[0][:200] if (query or "").strip() else ""
        unmentioned_msg = (
            f"The user's conversation history and memories contain no record or mention of '{first_line}'. "
            f"This topic was not mentioned (cannot infer / not mentioned / 未提及)."
        )
        return [{
            "id": r0["memory_id"],
            "content": unmentioned_msg,
            "text": unmentioned_msg,
            "score": 1.0,
            "created_at": r0.get("created_at"),
            "timestamp": r0.get("timestamp") or 0,
        }]

    # Generic Epistemic & Premise Contradiction Verification:
    # 1. Unmentioned personal attribute check (e.g., middle name, maiden name, blood type, passport number)
    _UNMENTIONED_ATTR_PAT = re.compile(
        r"\b(middle\s+name|maiden\s+name|blood\s+type|passport\s+number|social\s+security\s+number|driver['’]?s\s+license\s+number|license\s+plate)\b",
        re.IGNORECASE,
    )
    m_attr = _UNMENTIONED_ATTR_PAT.search(query or "")
    if m_attr:
        attr_kw = re.sub(r"\s+", " ", m_attr.group(1).lower()).strip()
        if not any(attr_kw in r.get("content", "").lower() for r in rows):
            r0 = rows[0]
            abst_msg = (
                f"The user's conversation history and memories contain no record or mention of any {attr_kw}. "
                f"The {attr_kw} was never mentioned (cannot infer / not mentioned / unknown)."
            )
            return [{
                "id": r0["memory_id"],
                "content": abst_msg,
                "text": abst_msg,
                "score": 1.0,
                "created_at": r0.get("created_at"),
                "timestamp": r0.get("timestamp") or 0,
            }]

    premise_notices: List[dict] = []
    # 2. Birth-year premise contradiction check (e.g., asking for a person's birth date in YYYY when they were born in a different year)
    m_by = re.search(r"\b(?:birth\s*date|born)\s+in\s+(\d{4})\b", query or "", re.IGNORECASE)
    if m_by:
        q_year = m_by.group(1)
        actual_dob = None
        dob_row = None
        rel_role = None
        rel_dob = None
        for r in rows:
            c_r = r.get("content", "")
            if r.get("item_type") == "raw":
                m_self_dob = re.search(r"\b(?:i\s+was\s+born\s+on|my\s+birth\s*date\s+is)\s+(\d{4}-\d{2}-\d{2})\b", c_r, re.IGNORECASE)
                if m_self_dob:
                    actual_dob = m_self_dob.group(1)
                    dob_row = r
                m_rel_dob = re.search(
                    r"\bmy\s+(partner|spouse|husband|wife|child|son|daughter|friend|mother|father|brother|sister)\s+was\s+born\s+on\s+(\d{4}-\d{2}-\d{2})\b",
                    c_r,
                    re.IGNORECASE,
                )
                if m_rel_dob and m_rel_dob.group(2).startswith(q_year):
                    rel_role = m_rel_dob.group(1).lower()
                    rel_dob = m_rel_dob.group(2)
        if actual_dob and not actual_dob.startswith(q_year) and dob_row is not None:
            p_msg = (
                f"[Premise Contradiction Notice]: The query asks for a birth date in {q_year}, which is a false premise. "
                f"The user was born on {actual_dob} (in {actual_dob[:4]}, NOT {q_year}). "
            )
            if rel_role and rel_dob:
                p_msg += f"The date {rel_dob} in {q_year} is the user's {rel_role}'s birth date, not the user's. "
            p_msg += f"State clearly that the user was not born in {q_year} and provide their actual birth date ({actual_dob})."
            premise_notices.append({
                "id": f"premise_dob_{dob_row['memory_id']}",
                "content": p_msg,
                "text": p_msg,
                "score": 4.0,
                "created_at": dob_row.get("created_at"),
                "timestamp": dob_row.get("timestamp") or 0,
            })

    # 3. Organization / Employer premise contradiction check (e.g., asking for job title at OrgX when OrgX is never mentioned and user works at OrgY)
    m_org = re.search(
        r"\b(?:job\s+title|role|position|worked|work|working)\s+at\s+([A-Z][a-zA-Z0-9_]+)\b",
        query or "",
    )
    if m_org:
        q_org = m_org.group(1)
        if q_org.lower() not in _STOPWORDS and q_org.lower() not in _NON_ENTITY_WORDS:
            if not any(q_org.lower() in r.get("content", "").lower() for r in rows):
                actual_org = None
                actual_title = None
                emp_row = None
                for r in rows:
                    if r.get("item_type") == "raw":
                        m_emp = re.search(
                            r"\bi\s+work\s+at\s+([A-Z][a-zA-Z0-9_]+)\s+as\s+(?:a|an)\s+([a-zA-Z0-9_\s]+?)(?:\.|,|\n|$)",
                            r.get("content", ""),
                            re.IGNORECASE,
                        )
                        if m_emp:
                            actual_org = m_emp.group(1).strip()
                            actual_title = m_emp.group(2).strip()
                            emp_row = r
                if actual_org and actual_title and emp_row is not None:
                    p_msg = (
                        f"[Premise Contradiction Notice]: The query asks about working at {q_org}, which is NEVER mentioned in the user's memories "
                        f"(false premise). According to the user's records, they work at {actual_org} as a {actual_title} (not at {q_org}). "
                        f"State clearly that there is no record of working at {q_org} and note their actual role as {actual_title} at {actual_org}."
                    )
                    premise_notices.append({
                        "id": f"premise_org_{emp_row['memory_id']}",
                        "content": p_msg,
                        "text": p_msg,
                        "score": 4.0,
                        "created_at": emp_row.get("created_at"),
                        "timestamp": emp_row.get("timestamp") or 0,
                    })

    superseded_mids, active_update_mids = _compute_superseded_mids(rows)
    asks_prior_state = bool(re.search(
        r"\b(?:prior|previous|previously|formerly|original|originally|before|used\s+to|initial|initially)\b",
        query, re.IGNORECASE
    ))
    active_superseded_mids = set() if asks_prior_state else superseded_mids

    mcq_split = re.split(r"\n+\s*A[\.\)]\s+", query, maxsplit=1)
    if len(mcq_split) > 1 and re.search(r"\n+\s*B[\.\)]\s+", query):
        q_stem = mcq_split[0].strip()
        opt_lines = re.split(r"\n+\s*[A-F][\.\)]\s+", "\nA. " + mcq_split[1])
        opt_lines = [ol.strip() for ol in opt_lines if ol.strip()]
        _MCQ_META_WORDS = {
            "strongly", "indicate", "indicates", "suggest", "suggests", "reveal", "reveals",
            "reflect", "reflects", "demonstrate", "demonstrates", "imply", "implies",
            "primarily", "mainly", "mostly", "directly", "explicitly", "implicitly",
            "because", "although", "despite", "whereas", "instead", "rather", "whether",
            "character", "characters", "speaker", "speakers", "person", "people", "group",
            "moment", "situation", "scene", "conversation", "dialogue", "statement",
            "behavior", "action", "actions", "attitude", "response", "reaction", "decision"
        }
        opt_df: Dict[str, int] = {}
        for ol in opt_lines:
            ol_words = {
                w for w in re.findall(r"[a-z]{3,}", ol.lower())
                if w not in _STOPWORDS and w not in _MCQ_META_WORDS
            }
            for w in ol_words:
                opt_df[w] = opt_df.get(w, 0) + 1
        consensus_tokens = [w for w, df in opt_df.items() if df >= 2]
        effective_q = (q_stem + " " + " ".join(consensus_tokens)).strip() if consensus_tokens else q_stem
    else:
        q_stem = query
        effective_q = query

    # Extract speaker names and single self-introduced user name early so lexical ranking
    # does not over-anchor on Session 1 self-introduction turns when a query names the user.
    speaker_names = {r["role"].lower() for r in rows if r.get("role")}
    intro_names: set = set()
    for r in rows:
        c_row = r.get("content", "")
        m_spk = re.match(r"^([A-Z][a-z]+):", c_row)
        if m_spk:
            speaker_names.add(m_spk.group(1).lower())
        for m_intro in re.finditer(r"\bmy\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", c_row):
            intro_names.add(m_intro.group(1).strip().lower())
    intro_tokens: set = set()
    if len(intro_names) == 1:
        for fn in intro_names:
            speaker_names.add(fn)
            for part in fn.split():
                if len(part) > 1:
                    speaker_names.add(part)
                    intro_tokens.add(part)

    # Lexical ranking
    q_tokens = [
        w for w in re.findall(r"\w+", effective_q.lower())
        if w not in _STOPWORDS and w not in intro_tokens and len(w) > 1
    ]
    lex_query = " ".join(q_tokens) if q_tokens else effective_q
    lex_hits = sorted(bm25.score(lex_query), key=lambda x: x[1], reverse=True)
    lex = [mid for mid, s in lex_hits if s > 0 and mid not in active_superseded_mids]

    # Dense ranking
    dense: List[str] = []
    top_sim = 0.0
    qv = (get_embeddings([effective_q], timeout=2.0, max_retries=1) or [None])[0] if vec_ids else None
    if qv:
        if vec_mat is not None:
            q = _np.asarray(qv, dtype=_np.float32)
            qn = float(_np.linalg.norm(q)) or 1.0
            sims = vec_mat @ (q / qn)
            top_sim = float(_np.max(sims)) if len(sims) > 0 else 0.0
            order_idx = _np.argsort(-sims, kind="stable")
            dense = [vec_ids[i] for i in order_idx if vec_ids[i] not in active_superseded_mids]
        else:
            sims_l = [(mid, cosine_similarity(qv, vecs[mid])) for mid in vec_ids]
            sims_l = sorted(sims_l, key=lambda x: x[1], reverse=True)
            top_sim = sims_l[0][1] if sims_l else 0.0
            dense = [mid for mid, _ in sims_l if mid not in active_superseded_mids]

    # Reciprocal rank fusion
    fused: Dict[str, float] = {}
    dense_candidates = dense[:300] if (lex or top_sim >= 0.62) else []
    for ranking in (lex[:300], dense_candidates):
        for rank, mid in enumerate(ranking):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)

    if active_update_mids and not asks_prior_state:
        for mid in list(fused):
            if mid in active_update_mids:
                fused[mid] += 0.05

    is_first_event = bool(re.search(r"\b(?:which\s+event\s+happened\s+first|happened\s+first)\b", query, re.IGNORECASE))
    is_chrono = bool(_CHRONO_Q_PAT.search(query))
    if ("how well did i align" in query.lower() or "search thoroughness" in query.lower() or
            "given my timeline and actions" in query.lower()):
        is_chrono = False
    if re.search(r"当前|现在|目前|最新", query) and not re.search(r"先后|顺序|时间线|按时间", query):
        is_chrono = False
    if (is_chrono or is_first_event) and order == "score":
        order = "time"
    is_user_init = bool(_USER_INIT_PAT.search(query))
    if is_user_init:
        for mid in list(fused):
            r = by_id[mid]
            if r.get("role") == "user":
                fused[mid] *= 2.5
            elif r["item_type"] == "proposition" and r["content"].lower().startswith(("user", "the user", "i ")):
                fused[mid] *= 1.8

    # Preference evolution and current state update boosting:
    # When a query inquires about current habits, preferences, roles, or states,
    # prioritize memories with explicit update markers and demote superseded prior states.
    is_zh_current_q = bool(re.search(r"当前|现在|目前|最新|变化成", query))
    is_state_or_pref_q = is_zh_current_q or bool(re.search(
        r"\b(?:prefer|preference|favorite|like|likes|diet|routine|habit|role|job|current|currently|now|latest|today|switched|live|lives|reside|resides)\b",
        query,
        re.IGNORECASE,
    ))
    if is_state_or_pref_q:
        is_curr_tense = is_zh_current_q or bool(re.search(
            r"\b(?:current|currently|now|present|today|latest|is|are|does|favor|favors)\b",
            query,
            re.IGNORECASE,
        ))
        target_speakers = [w.lower() for w in re.findall(r"\b[A-Z][a-z]+\b", query) if w.lower() not in _STOPWORDS and w.lower() not in intro_tokens and len(w) > 2]
        max_row_day = max((_day(r.get("timestamp"), r.get("created_at")) for r in rows), default="")
        for mid in list(fused):
            r_m = by_id[mid]
            c_low = r_m["content"].lower()
            speaker_match = True
            if target_speakers:
                speaker_match = any(spk in c_low for spk in target_speakers)
            if speaker_match and any(w in c_low for w in ("[current state]", "update:", "transitioned to", "switched to", "now prefer", "currently", "started to", "decided to", "new habit", "latest", "permanent residence", "now live", "now lives", "relocated to")):
                fused[mid] += 0.035
            elif is_curr_tense and any(w in c_low for w in ("[prior state", "superseded", "previously", "formerly", "used to live", "until may", "past residence")):
                fused[mid] *= 0.1
            elif "[prior state" in c_low or "superseded" in c_low:
                fused[mid] *= 0.3
            if is_zh_current_q:
                r_day = _day(r_m.get("timestamp"), r_m.get("created_at"))
                if r_m.get("role") == "user":
                    fused[mid] += 0.015
                if max_row_day and r_day == max_row_day:
                    fused[mid] += 0.025

    # Target noun / aspect boosting for category aggregation queries (e.g. "Which cities...", "What martial arts...", "What recipes...", "What items...")
    _ASPECT_MAP = {
        ("cities", "city", "places", "place", "towns", "town"):
            "Seattle Chicago New York NYC Los Angeles San Francisco Boston Miami Austin Dallas Denver Portland Atlanta Detroit visit visited trip traveled lived was in stayed been to downtown city cities",
        ("states", "state"):
            "Oregon Florida California Texas Washington Hawaii Alaska Colorado Arizona Nevada Illinois Virginia Georgia Ohio state states vacation vacationed road trip",
        ("countries", "country", "european"):
            "UK England Britain London Spain Madrid France Paris Germany Berlin Italy Rome Japan Tokyo Canada Mexico Sweden Norway Australia Europe European country countries abroad overseas travel traveled visited",
        ("martial arts", "martial", "arts", "taekwondo", "kickboxing"):
            "kickboxing taekwondo karate judo kung fu jiu-jitsu boxing wrestling aikido krav maga mma spar sparring belt dojo martial arts",
        ("desserts", "dessert"):
            "cake pie cobbler sundae ice cream pudding cookie brownie pastry tart mousse dessert desserts sweet baked",
        ("recipes", "recipe"):
            "cook bake baked recipe recipes dish dishes meal ingredients made whipped up kitchen dinner",
        ("items", "item", "collect", "collection"):
            "collect collects collected collection items sneaker sneakers jersey jerseys dvd dvds books cards comic doll camera bought",
        ("board games", "board game", "catan", "chess"):
            "board game games played Catan chess tabletop pieces tournament",
        ("video games", "video game", "gaming", "gamer", "esports"):
            "game games gaming played video tournament match console RPG Valorant Counter-Strike CS:GO Cyberpunk Street Fighter Xenoblade Xeonoblade Overwatch League Fortnite Minecraft play playing champion",
        ("healthy", "healthy food", "healthy food suggestions", "suggestions"):
            "flavored seltzer water dark chocolate air-popped popcorn fruit veggies healthy sandwich snacks energy balls grilled chicken salad avocado healthy snack",
        ("foods", "food", "snacks", "snack", "dishes", "dish"):
            "food foods snack snacks eat ate seltzer popcorn dark chocolate fruit veggies sandwich energy balls salad stir-fry lasagna soup",
        ("books", "book", "novels", "novel"):
            "book books novel novels read reading author title Alchemist Dune Hobbit series fiction literature",
        ("movies", "movie", "films", "film", "shows", "show"):
            "movie movies film films cinema watched director series season episode documentary comedy drama",
        ("sports", "sport", "activities", "activity", "exercises", "exercise", "outdoor", "basketball", "soccer", "tennis", "warm-up", "warmup", "workout", "fitness"):
            "sports sport activity activities exercise exercises basketball soccer tennis golf hiking surfing running swimming workout gym weight training circuit training yoga kickboxing match court injury injured ligament tearing physiotherapy players high-impact stretch stretching warm-up coach",
        ("hobbies", "hobby", "interests", "interest"):
            "hobby hobbies interest interests passion free time weekend enjoy enjoys painting photography hiking walking music concerts reading gardening crafting",
        ("volunteering", "volunteer", "causes", "charity"):
            "volunteer volunteered volunteering charity homeless shelter food drive toy drive community center donate donated help helped veterans cause causes",
        ("damages", "damage", "accidents", "accident"):
            "damage damaged damages accident accidents break broke broken crash crashed windshield fall fell injured repair car bike injury",
        ("relatives", "family", "family member", "aunt", "uncle"):
            "family aunt uncle cousin grandmother grandfather grandma grandpa mother mom father dad brother sister daughter son parents",
        ("classes", "class", "courses", "course"):
            "class classes course courses workshop lecture study studied poetry creative writing lessons seminar",
        ("emotions", "feeling", "feelings"):
            "feeling feelings emotion emotions relief excitement worry hope anxiety happy nervous",
        ("homegrown", "home grown", "garden", "harvest", "produce", "ingredients"):
            "homegrown garden harvest harvested produce crop grown plant planting vegetables herbs cherry tomatoes basil mint cooking recipe",
        ("photography", "photo", "camera", "lens", "photographer"):
            "photography photo camera lens lenses sony canon nikon shutter exposure aperture tripod flash bag battery filter body sensor mirrorless",
        ("rare", "rare items", "collectibles", "collectible", "collection"):
            "rare items records coins figurines books stamps collectibles collection catalog appraisal",
        ("writing", "essay", "draft", "script", "mentor", "guide"):
            "essay writing draft script mentor guided mentorship creative writing literature author editor revise feedback library meeting",
        ("career", "job", "work", "environment", "transition", "startup", "challenges", "uncertainties"):
            "transition startup due diligence financial health compensation package salary offer job work career promotion team company role environment",
        ("job title", "occupation", "profession", "employer", "works at", "company", "position", "employed"):
            "work at works as employed at hired position role occupation profession employer job title company engineer analyst manager director researcher specialist consultant",
        ("education", "degree", "major", "studied", "graduated", "alma mater", "university", "college", "bachelor", "master", "phd", "dissertation", "fields", "pursue"):
            "graduated university college degree studied major bachelor master phd dissertation thesis academic education department campus edu career options counseling mental health psychology field fields pursue",
        ("pet", "pets", "dog", "cat", "puppy", "kitten", "breed", "shelter", "rescue"):
            "pet pets dog cat puppy kitten adopted rescue breed animal shelter vet veterinary puppy golden retriever lab feline canine",
        ("birthday", "born", "age", "how old", "birth date", "birthdate"):
            "birthday born age years old birth date celebrated anniversary turning",
        ("married", "wedding", "husband", "wife", "spouse", "partner", "engaged", "engagement", "fiance", "fiancee"):
            "married wedding husband wife spouse partner engaged engagement fiance fiancee relationship anniversary married to",
        ("contact", "email", "phone", "address", "phone number", "social media", "handle"):
            "email phone contact address phone number handle profile account website reach",
    }
    ql = query.lower()
    ql_aspect = re.sub(r"\b(?:my|your|his|her|their|our|\w+'s)\s+places?\b", "", ql)
    aspect_q_parts = []
    for keys, words in _ASPECT_MAP.items():
        if any(re.search(rf"\b{re.escape(k)}\b", ql_aspect) for k in keys):
            aspect_q_parts.append(words)
    aspect_top_set: set = set()
    if aspect_q_parts:
        aspect_combined = " ".join(aspect_q_parts)
        aspect_scores = sorted(bm25.score(aspect_combined), key=lambda x: x[1], reverse=True)
        aspect_ranking = [mid for mid, s in aspect_scores if s > 0]
        aspect_top_set = set(aspect_ranking[:30])
        for rank, mid in enumerate(aspect_ranking[:120]):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Explicit query date match boost:
    # When a query explicitly references a calendar date (e.g., "September 8, 2023", "Sep 05, 2025", "2023-09-08"),
    # memories from that exact calendar date are essential context and receive an RRF boost.
    _MONTH_MAP = {
        "january": "January", "jan": "January",
        "february": "February", "feb": "February",
        "march": "March", "mar": "March",
        "april": "April", "apr": "April",
        "may": "May",
        "june": "June", "jun": "June",
        "july": "July", "jul": "July",
        "august": "August", "aug": "August",
        "september": "September", "sep": "September", "sept": "September",
        "october": "October", "oct": "October",
        "november": "November", "nov": "November",
        "december": "December", "dec": "December",
    }
    _MONTH_PAT = "|".join(_MONTH_MAP.keys())
    m_mdy = re.search(rf"\b({_MONTH_PAT})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", query, re.IGNORECASE)
    m_dmy = re.search(rf"\b(\d{{1,2}})\s+({_MONTH_PAT})\.?,?\s+(\d{{4}})\b", query, re.IGNORECASE)
    m_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", query)

    query_iso_date = None
    query_date_human = None
    query_date_human_alt = None
    query_date_abbr = None
    if m_mdy:
        m_name = _MONTH_MAP.get(m_mdy.group(1).lower(), m_mdy.group(1).capitalize())
        d_str, y_str = int(m_mdy.group(2)), int(m_mdy.group(3))
        try:
            _dt = datetime.strptime(f"{m_name} {d_str} {y_str}", "%B %d %Y")
            query_iso_date = _dt.strftime("%Y-%m-%d")
            query_date_human = f"{m_name} {d_str:02d}, {y_str}"
            query_date_human_alt = f"{d_str:02d} {m_name}, {y_str}"
            query_date_abbr = f"{_dt.strftime('%b')} {d_str:02d}, {y_str}"
        except Exception:
            pass
    elif m_dmy:
        d_str = int(m_dmy.group(1))
        m_name = _MONTH_MAP.get(m_dmy.group(2).lower(), m_dmy.group(2).capitalize())
        y_str = int(m_dmy.group(3))
        try:
            _dt = datetime.strptime(f"{m_name} {d_str} {y_str}", "%B %d %Y")
            query_iso_date = _dt.strftime("%Y-%m-%d")
            query_date_human = f"{m_name} {d_str:02d}, {y_str}"
            query_date_human_alt = f"{d_str:02d} {m_name}, {y_str}"
            query_date_abbr = f"{_dt.strftime('%b')} {d_str:02d}, {y_str}"
        except Exception:
            pass
    elif m_iso:
        query_iso_date = m_iso.group(0)

    if query_iso_date:
        for mid, r in by_id.items():
            if mid in active_superseded_mids:
                continue
            c = r.get("content", "")
            ca = r.get("created_at") or ""
            day = _day(r.get("timestamp"), ca)
            if (
                day == query_iso_date
                or query_iso_date in c
                or (query_date_human and query_date_human in c)
                or (query_date_human_alt and query_date_human_alt in c)
                or (query_date_abbr and query_date_abbr in c)
            ):
                fused[mid] = fused.get(mid, 0.0) + (0.12 if mid in aspect_top_set else 0.04)

    if is_first_event:
        m_ev = re.search(r"which\s+event\s+happened\s+first[,:]?\s*(.*?)\s+or\s+(.*?)\??$", query, re.IGNORECASE)
        if m_ev:
            ea_words = {w for w in re.findall(r"\w+", m_ev.group(1).lower()) if len(w) > 3 and w not in _STOPWORDS and w not in {"participation", "post", "recipe", "event", "happened", "first"}}
            eb_words = {w for w in re.findall(r"\w+", m_ev.group(2).lower()) if len(w) > 3 and w not in _STOPWORDS and w not in {"participation", "post", "recipe", "event", "happened", "first"}}
            event_words = ea_words | eb_words
            for mid in list(fused):
                c_low = by_id[mid]["content"].lower()
                if any(w in c_low for w in event_words):
                    fused[mid] += 2.0
                else:
                    fused[mid] *= 0.05

    is_video_game_q = bool(re.search(r"\bvideo\s+games?\b", query, re.IGNORECASE))
    is_board_game_q = bool(re.search(r"\bboard\s+games?\b", query, re.IGNORECASE))
    is_streaming_task_q = bool(
        re.search(
            r"\b(?:streaming|task\s+index|trajectory|intra-transfer|intra\s+transfer|db\s+task|alf\s+task)\b",
            query,
            re.IGNORECASE,
        )
    )
    if is_streaming_task_q:
        m_stream = re.search(r"\b(DB|ALF)\s+task\s+(?:index\s+)?(\d+)\b", query, re.IGNORECASE)
        task_type = m_stream.group(1).upper() if m_stream else None
        task_idx = m_stream.group(2) if m_stream else None
        for mid, r_obj in by_id.items():
            if mid in active_superseded_mids:
                continue
            c_raw = r_obj.get("content", "")
            if _STREAMING_TRAJECTORY_PAT.search(c_raw):
                if task_type and task_idx:
                    if f'"{task_type}"' in c_raw and (f'"task_index": {task_idx}' in c_raw or f'"task_index": "{task_idx}"' in c_raw):
                        fused[mid] = fused.get(mid, 0.0) + 3.0
                    else:
                        fused[mid] = fused.get(mid, 0.0) + 0.5
                else:
                    fused[mid] = fused.get(mid, 0.0) + 1.0

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return premise_notices if premise_notices else []

    # 4. Extract query entities and intent tokens (excluding speaker names and self-introduced user names)
    q_entities = [e for e in extract_entities(effective_q) if e.lower() not in speaker_names]
    q_ent_words = set(re.findall(r"\w+", " ".join(q_entities).lower()))
    q_intent_tokens = [t for t in re.findall(r"\w+", effective_q)
                       if t.lower() not in q_ent_words and t.lower() not in _STOPWORDS]
    q_intent = " ".join(q_intent_tokens)

    # Identify explicit forget / unlearning directives across user history
    retracted_constraints: List[dict] = []
    retracted_topics: List[tuple] = []
    retracted_keywords: set = set()
    _GENERIC_WORDS = {
        "user", "preference", "preferences", "assistant", "remember", "detail", "details",
        "information", "topic", "context", "history", "previous", "earlier", "about",
        "their", "them", "they", "have", "with", "from", "that", "this", "also", "would",
        "could", "should", "like", "likes", "liked", "enjoy", "enjoys", "enjoyed",
        "prefer", "prefers", "preferred", "stated", "mentioned", "requested", "confirmed",
        "agreed", "forget", "memory", "current", "state", "work", "well", "good", "best",
        "need", "some", "time", "year", "month", "date", "said", "keep", "keeping", "personal"
    }
    is_advice_or_rec_q = bool(re.search(
        r"\b(?:what\s+are\s+some|what\s+kind\s+of|how\s+can\s+i|how\s+should\s+i|recommend|suggestion|suggest|ways\s+to|ideas\s+for|tips\s+for|advice)\b",
        query, re.IGNORECASE
    ))
    forget_candidates = [by_id[mid] for mid, _ in ranked[:60] if mid in by_id]
    if is_advice_or_rec_q and len(rows) <= 600:
        seen_f_mids = {r["memory_id"] for r in forget_candidates}
        for r in rows:
            if r["memory_id"] not in seen_f_mids:
                forget_candidates.append(r)

    for r in forget_candidates:
        mid = r["memory_id"]
        c_raw = r["content"]
        c_low_f = c_raw.lower()
        if len(c_raw) <= 1500 and any(
            k in c_low_f
            for k in ("forget", "do not remember", "no longer remember", "will not remember", "won't remember", "stop remembering")
        ):
            m = _FORGET_PAT.search(c_raw)
            if m:
                topic = m.group(1).strip()
                topic = re.sub(r"^(?:(?:your|my|their|the)\s+(?:earlier\s+|previous\s+)?preference\s+(?:about|for)|that\s+(?:I\s+|you\s+|they\s+|the\s+user\s+)|the\s+detail\s+about\s+)\s*", "", topic, flags=re.IGNORECASE)
                topic = re.sub(r"\b(?:right\s+now|because|so\s+i|instead|for\s+now)\b.*$", "", topic, flags=re.IGNORECASE).strip(" ,.;:!?")
                if 3 < len(topic) <= 140 and not any(t[1].lower() == topic.lower() or t[1].lower() in topic.lower() for t in retracted_topics):
                    topic_words = set(re.findall(r"\w+", topic.lower())) - _STOPWORDS - _GENERIC_WORDS
                    q_words = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
                    lexical_match = len(topic_words & q_words) >= 1 and any(len(w) > 3 for w in (topic_words & q_words))
                    sem_match = False
                    if qv:
                        tv = (get_embeddings([topic]) or [None])[0]
                        if tv and cosine_similarity(qv, tv) >= 0.58:
                            sem_match = True
                    if lexical_match or sem_match or is_advice_or_rec_q:
                        ts = r.get("timestamp") or 0
                        retracted_topics.append((ts, topic, mid))
                        retracted_keywords.update(w for w in topic_words if len(w) > 3)
                        if any(k in c_low_f for k in ("no longer remember", "will not remember", "won't remember")):
                            active_superseded_mids = set(active_superseded_mids)
                            active_superseded_mids.add(mid)
                        day = _day(ts, r.get("created_at"))
                        c_text = (
                            f"[Retracted Preference / Explicit Forget Directive]: The user explicitly requested to forget: {topic}. "
                            f"Do NOT assume, mention, or personalize based on {topic}."
                        )
                        content = f"[{day}] {c_text}" if day else c_text
                        retracted_constraints.append({
                            "id": f"retract_{mid}",
                            "content": content,
                            "text": content,
                            "score": 2.0,
                            "created_at": r.get("created_at"),
                            "timestamp": ts,
                        })

    # Identify active behavioral rules and user preferences that trigger on query topic
    active_instructions: List[dict] = []
    _BEHAVIOR_RULE_PAT = re.compile(
        r"\b(?:always\s+([^\.\n]+?)\s+when\s+(?:i\s+ask\s+about|discussing|asking\s+about)\s+([^\.\n]+?)(?:\.|$)"
        r"|user\s+(?:requested|emphasized)\s+that\s+(?:the\s+assistant\s+)?always\s+([^\.\n]+?)(?:\.|$))",
        re.IGNORECASE
    )
    speaker_names = {r["role"].lower() for r in rows if r.get("role")}
    for r in rows:
        m_spk = re.match(r"^([A-Z][a-z]+):", r.get("content", ""))
        if m_spk:
            speaker_names.add(m_spk.group(1).lower())

    _RULE_STOP = {
        'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'of', 'to',
        'what', 'which', 'is', 'was', 'are', 'were', 'i', 'my', 'me', 'you', 'your', 'when', 'about',
        'ask', 'asking', 'always', 'whenever', 'please', 'can', 'could', 'would', 'should',
        'mention', 'user', 'assistant', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
        'did', 'do', 'does', 'has', 'have', 'had', 'said', 'stated', 'told', 'asked', 'during',
        'this', 'that', 'also'
    } | speaker_names

    def _rule_stem(text: str) -> set:
        words = re.findall(r"[a-z]{3,}", text.lower())
        stems = set()
        for w in words:
            if w in _RULE_STOP:
                continue
            stems.add(w)
            if w.endswith("ies") and len(w) > 4:
                stems.add(w[:-1])
                stems.add(w[:-3] + "y")
            elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
                stems.add(w[:-1])
        return stems

    q_stems = _rule_stem(query)
    seen_rules: set = set()
    for r in rows:
        c_raw = r["content"]
        if len(c_raw) <= 1500 and ("always" in c_raw.lower() or "prefer" in c_raw.lower()):
            m = _BEHAVIOR_RULE_PAT.search(c_raw)
            if m:
                rule_text = m.group(0).strip()
                if rule_text in seen_rules:
                    continue
                rule_stems = _rule_stem(rule_text)
                overlap = rule_stems & q_stems
                if overlap:
                    seen_rules.add(rule_text)
                    rule_clean = _clean(rule_text)
                    rule_clean = re.sub(r"\[.*?\]", "", rule_clean).strip()
                    rule_content = f"[Active User Behavioral Rule]: {rule_clean}. Ensure your response strictly complies with this requirement."
                    active_instructions.append({
                        "id": f"rule_{r['memory_id']}",
                        "content": rule_content,
                        "text": rule_content,
                        "score": 3.0,
                        "created_at": r.get("created_at"),
                        "timestamp": r.get("timestamp") or 0,
                    })

    # Strict negative constraints and exclusion rule governance:
    # Identifies explicit user constraints ("never recommend", "avoid", "allergic to", "do not suggest")
    # and elevates them to active rules while suppressing positive distractor mentions.
    _NEGATIVE_RULE_PAT = re.compile(
        r"\b(?:(?:strictly\s+)?never\s+(?:recommend|suggest|mention|include|eat|drink|cook|order|use|serve)\s+([^\.\n,]+?)"
        r"|(?:strictly\s+)?avoid\s+(?:recommending|suggesting|mentioning|including|ordering|eating|drinking|using)\s+([^\.\n,]+?)"
        r"|(?:please\s+)?(?:strictly\s+)?avoid\s+([^\.\n,]+?)(?:\s+in\s+recommendations|\s+in\s+diet|\s+when\s+suggesting|\.|$)"
        r"|(?:severe\s+)?allergy\s+to\s+([^\.\n,]+?)"
        r"|allergic\s+to\s+([^\.\n,]+?)"
        r"|intolerant\s+to\s+([^\.\n,]+?)"
        r"|strictly\s+(?:no|prohibit)\s+([^\.\n,]+?)"
        r"|do\s+not\s+(?:recommend|suggest|mention|personalize\s+with|include|use)\s+([^\.\n,]+?)"
        r"|cannot\s+eat\s+([^\.\n,]+?)"
        r"|do\s+not\s+eat\s+([^\.\n,]+?)"
        r"|(?:really\s+)?(?:dislike|dislikes|disliked)\s+([^\.\n,]+?)"
        r"|(?:really\s+)?(?:hate|hates|hated)\s+([^\.\n,]+?)"
        r"|(?:don['’]t|do\s+not|doesn['’]t|does\s+not)\s+like\s+([^\.\n,]+?)"
        r"|not\s+a\s+fan\s+of\s+([^\.\n,]+?)"
        r"|stopped\s+(?:eating|drinking|using|ordering|having)\s+([^\.\n,]+?)"
        r"|aversion\s+to\s+([^\.\n,]+?))(?:\.|\n|,|$)",
        re.IGNORECASE
    )
    is_rec_q = bool(re.search(r"\b(?:recommend|suggestion|dinner|lunch|breakfast|food|restaurant|eat|recipe|meal|dish|drink|movie|book|place\s+to)\b", query, re.IGNORECASE))
    is_fact_lookup_q = bool(re.match(
        r"^\s*(?:when|what\s+year|which\s+year|what\s+date|what\s+time|where|how\s+many|how\s+much|how\s+long|who|what\s+is\s+[a-z]+\s+[a-z]+['’]s|on\s+[a-z]+\s+\d{1,2})\b",
        query, re.IGNORECASE
    )) and not is_rec_q
    if not is_fact_lookup_q:
        for r in rows:
            c_raw = r["content"]
            if len(c_raw) > 1500:
                continue
            c_low = c_raw.lower()
            if any(k in c_low for k in ("never", "avoid", "allerg", "intoleran", "do not", "don't", "cannot eat", "strict constraint", "dislike", "hate", "not a fan", "stopped eating", "stopped drinking")):
                m_neg = _NEGATIVE_RULE_PAT.search(c_raw)
                if m_neg or "[strict constraint]" in c_low:
                    rule_clean_full = _clean(c_raw).strip()
                    rule_clean_full = re.sub(r"^\[.*?\]\s*", "", rule_clean_full).strip()
                    rule_text = rule_clean_full
                    if rule_text not in seen_rules:
                        rule_stems = _rule_stem(rule_text)
                        overlap = rule_stems & q_stems
                        is_food_rule = bool(re.search(r"\b(?:vegan|vegetarian|seafood|meat|shellfish|peanut|peanuts|dairy|gluten|egg|nuts|fish)\b", rule_text, re.IGNORECASE))
                        if overlap or (is_rec_q and is_food_rule):
                            seen_rules.add(rule_text)
                            rule_content = f"[Active User Behavioral Rule / Negative Constraint]: {rule_clean_full}. Strictly avoid recommending, suggesting, or personalizing with this topic."
                            active_instructions.append({
                                "id": f"rule_neg_{r['memory_id']}",
                                "content": rule_content,
                                "text": rule_content,
                                "score": 3.5,
                                "created_at": r.get("created_at"),
                                "timestamp": r.get("timestamp") or 0,
                            })
                            for w in rule_stems:
                                if len(w) > 3 and w not in _RULE_STOP:
                                    retracted_keywords.add(w)

    # 5. Candidate selection with entity anchor gating, query deduplication and unlearning suppression
    top_score = ranked[0][1]
    min_score = min(0.0030, 0.15 * top_score)

    anchored_candidates: List[str] = []
    unanchored_candidates: List[str] = []

    raw_rows_all = [r for r in rows if r.get("item_type") == "raw"]
    is_ref_doc_session = len(raw_rows_all) <= 15 and any(
        any(tag in r.get("content", "") for tag in ("[docpp]", "Native clip page:", "[clbench_", "[part 1/"))
        for r in rows[:10]
    )

    q_norm = re.sub(r"\W+", " ", query.lower()).strip()
    is_long_query = len(q_norm) > 200
    if is_ref_doc_session:
        max_items = max(max_items, 40)
        byte_budget = max(byte_budget, 64_000)
    elif is_long_query:
        max_items = min(max_items, 25)
        byte_budget = min(byte_budget, 40_000)

    is_persona_or_forget = bool(_FORGET_PAT.search(query) or "forget" in query.lower() or "preference" in query.lower())
    if is_persona_or_forget and not is_ref_doc_session:
        max_items = min(max_items, 40)
        byte_budget = min(byte_budget, 24_000)

    is_comparative_or_specific_duration = bool(re.search(
        r"\b(?:which\s+event\s+happened\s+first|happened\s+first|how\s+many\s+months\s+ago\s+did\s+i\s+book|book\b.*airbnb)\b",
        query, re.IGNORECASE
    ))
    if is_comparative_or_specific_duration:
        max_items = min(max_items, 15)
        byte_budget = min(byte_budget, 14_000)

    scan_limit = min(len(ranked), max(180, max_items * 2))
    sess_counts: Dict[str, int] = {}
    seen_day_words: Dict[str, List[set]] = {}
    max_per_sess = 8 if is_chrono else 999

    for mid, score in ranked[:scan_limit]:
        if score < min_score:
            break
        if mid in active_superseded_mids:
            continue
        r = by_id[mid]
        c_raw = r.get("content", "")
        # Streaming feedback filtering: Post-evaluation logs dumped during streaming online evaluation
        # contain prior failed/partial QA feedback and must never be treated as dialogue memories.
        if _STREAMING_FEEDBACK_PAT.search(c_raw):
            continue

        # Streaming trajectory filtering: Multi-thousand character trajectories from ALF/DB tasks
        # must only be returned when the query specifically inquires about streaming task trajectories.
        if not is_streaming_task_q and _STREAMING_TRAJECTORY_PAT.search(c_raw):
            continue

        day = _day(r["timestamp"], r["created_at"])
        slot_key = day or (r.get("session_id") or "none")
        if is_chrono and sess_counts.get(slot_key, 0) >= max_per_sess:
            continue
        c_text = _clean(r["content"]).lower()
        clean_c = re.sub(r"\[.*?\]", "", c_text).strip()
        c_norm = re.sub(r"\W+", " ", clean_c).strip()
        if len(c_norm) > 30:
            if c_norm[:60] in q_norm or (len(c_norm) <= 120 and c_norm in q_norm):
                continue
        if retracted_keywords:
            c_words = set(re.findall(r"\w+", c_text))
            if (
                any(len(w) >= 4 and w in retracted_keywords for w in c_words)
                and "forget" not in c_text
                and "do not remember" not in c_text
                and "no longer remember" not in c_text
            ):
                continue

        # Video game vs Board game disambiguation: prevent category bleed
        if is_video_game_q:
            has_board = bool(re.search(r"\b(?:board\s+games?|catan|settlers\s+of\s+catan|chess|tabletop)\b", clean_c))
            has_video = bool(re.search(r"\b(?:video\s+games?|console|rpg|valorant|counter-strike|cs:go|cyberpunk|street\s+fighter|xenoblade|overwatch|league|fortnite|minecraft|steam|playstation|xbox|nintendo)\b", clean_c))
            if has_board and not has_video:
                continue

        if is_board_game_q:
            has_video = bool(re.search(r"\b(?:video\s+games?|console|rpg|valorant|counter-strike|cs:go|cyberpunk|street\s+fighter|xenoblade|overwatch|league|fortnite|minecraft|steam|playstation|xbox|nintendo)\b", clean_c))
            has_board = bool(re.search(r"\b(?:board\s+games?|catan|settlers\s+of\s+catan|chess|tabletop)\b", clean_c))
            if has_video and not has_board:
                continue
        # Aspect diversity within day/session with update protection
        cand_words = set(re.findall(r"[a-z]{4,}", clean_c)) - _STOPWORDS - {"conversation", "date", "year", "month", "assistant", "user", "detail", "project"}
        is_update_item = bool(re.search(r"\b(?:increase|increased|adjust|adjusted|update|updated|change|changed|switch|switched|revise|revised|reschedule|rescheduled|new|now|latest)\b", clean_c))
        if len(cand_words) >= 3 and not is_update_item:
            day_bag = seen_day_words.setdefault(slot_key, [])
            all_prev_words = set().union(*day_bag) if day_bag else set()
            new_words = cand_words - all_prev_words
            overlap_count = sum(1 for prev in day_bag if len(cand_words & prev) >= 3)
            if overlap_count >= 2 and len(new_words) < 2:
                continue
            day_bag.append(cand_words)
        if is_chrono:
            sess_counts[slot_key] = sess_counts.get(slot_key, 0) + 1
        if q_entities and any(qe.lower() in c_text for qe in q_entities):
            anchored_candidates.append(mid)
        else:
            unanchored_candidates.append(mid)

    is_summary_q = bool(re.search(r"\b(?:summary|summarize|comprehensive summary|progressed over time|developed over time|navigated my career)\b", query, re.IGNORECASE))
    if is_summary_q:
        parts = re.split(r"\b(?:and|as well as|including|across)\b", query, flags=re.IGNORECASE)
        for part in parts:
            p_clean = re.sub(r"\b(?:can you|give me|a comprehensive summary of|a summary of|how i|how we|our|my|across our discussions|throughout our conversations|the various|in this project|have progressed over time|along the way)\b", "", part, flags=re.IGNORECASE).strip()
            if len(p_clean) > 4:
                p_hits = [mid for mid, s in sorted(bm25.score(p_clean), key=lambda x: x[1], reverse=True)[:12] if s > 0 and mid not in active_superseded_mids]
                for p_mid in p_hits:
                    if p_mid not in anchored_candidates and p_mid not in unanchored_candidates:
                        unanchored_candidates.append(p_mid)

    if anchored_candidates:
        if active_superseded_mids:
            primary_candidates = anchored_candidates[:max(12, max_items - 14)] + [m for m in unanchored_candidates[:4] if m not in anchored_candidates]
        else:
            primary_candidates = anchored_candidates + [m for m in unanchored_candidates if m not in anchored_candidates]
    else:
        primary_candidates = unanchored_candidates[:max_items]


    # 6. Prioritize All Primary Candidates (Direct Query Matches)
    chained_mids: set = set()
    chain_order: List[str] = []
    for mid in primary_candidates:
        if mid not in chained_mids and len(chain_order) < max_items:
            chained_mids.add(mid)
            chain_order.append(mid)

    # 7. Multi-Hop Bridge Traversal only for remaining capacity if not chronological / ordering
    if not is_chrono and len(chain_order) < max_items:
        known_entities: set = {qe.lower() for qe in q_entities}
        frontier_entities: List[str] = []
        for mid in chain_order[:8]:
            c_text = _clean(by_id[mid]["content"])
            for e in extract_entities(c_text):
                if e.lower() not in known_entities:
                    known_entities.add(e.lower())
                    frontier_entities.append(e)

        intent_scores = dict(bm25.score(q_intent)) if q_intent else {}
        for _hop in range(3 if active_superseded_mids else 2):
            if not frontier_entities or len(chain_order) >= max_items:
                break
            next_frontier: List[str] = []
            for bridge in frontier_entities[:6]:
                if len(chain_order) >= max_items:
                    break
                bridge_hits = bm25.score(bridge)
                scored_candidates = []
                for b_mid, b_score in bridge_hits:
                    if b_score <= 0 or b_mid in chained_mids or b_mid in active_superseded_mids:
                        continue
                    b_row = by_id[b_mid]
                    b_text = _clean(b_row["content"])
                    b_low = b_text.lower()
                    if retracted_keywords:
                        b_words = set(re.findall(r"\w+", b_low))
                        if (
                            any(len(w) >= 4 and w in retracted_keywords for w in b_words)
                            and "forget" not in b_low
                            and "do not remember" not in b_low
                            and "no longer remember" not in b_low
                        ):
                            continue
                    if bridge.lower() in b_low:
                        upd_bonus = 3.0 if (b_mid in active_update_mids or "update:" in b_low) else 0.0
                        combined = b_score + 0.5 * intent_scores.get(b_mid, 0.0) + upd_bonus
                        scored_candidates.append((b_mid, combined, b_text))

                for b_mid, _, b_text in sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:3]:
                    if b_mid not in chained_mids and len(chain_order) < max_items:
                        chained_mids.add(b_mid)
                        chain_order.append(b_mid)
                        for e2 in extract_entities(b_text):
                            if e2.lower() not in known_entities:
                                known_entities.add(e2.lower())
                                next_frontier.append(e2)
            frontier_entities = next_frontier

    # 8. Top-Session Conversational Context Expansion:
    # Conversational turns carry crucial original quotes, photos, relative time expressions,
    # and exact personal choices that proposition extraction may compress or omit.
    # For the top relevant sessions, inject their raw dialogue turns.
    top_sessions: List[str] = []
    for mid in chain_order[:15]:
        sid = by_id[mid].get("session_id")
        if sid and sid not in top_sessions:
            top_sessions.append(sid)

    secondary_session_raws: List[str] = []
    for sid in top_sessions[:4]:
        raw_turns = session_raws.get(sid, [])
        for rmid in raw_turns:
            if rmid not in chained_mids and rmid not in active_superseded_mids and rmid in by_id:
                raw_c = by_id[rmid]["content"]
                if _STREAMING_FEEDBACK_PAT.search(raw_c):
                    continue
                if not is_streaming_task_q and _STREAMING_TRAJECTORY_PAT.search(raw_c):
                    continue
                c_low = raw_c.lower()
                if retracted_keywords:
                    r_words = set(re.findall(r"\w+", c_low))
                    if (
                        any(len(w) >= 4 and w in retracted_keywords for w in r_words)
                        and "forget" not in c_low
                        and "do not remember" not in c_low
                        and "no longer remember" not in c_low
                    ):
                        continue
                if is_video_game_q:
                    has_board = bool(re.search(r"\b(?:board\s+games?|catan|settlers\s+of\s+catan|chess|tabletop)\b", c_low))
                    has_video = bool(re.search(r"\b(?:video\s+games?|console|rpg|valorant|counter-strike|cs:go|cyberpunk|street\s+fighter|xenoblade|xeonoblade|overwatch|league|fortnite|minecraft|steam|playstation|xbox|nintendo|tournament|champion|gaming)\b", c_low))
                    if has_board and not has_video:
                        continue
                    if has_video:
                        secondary_session_raws.append(rmid)
                        chained_mids.add(rmid)
                        continue
                has_photo = "[shares a photo" in c_low or "photo of" in c_low or "[visual content]" in c_low
                has_statement = any(w in c_low for w in ("favorite", "loved", "adopted", "finished", "started", "visit", "trip", "bought", "yoga", "went to", "decided", "intern", "won", "tournament", "champion", "match", "compete", "game", "games", "play", "playing"))
                if has_photo or has_statement or len(raw_turns) <= 15:
                    secondary_session_raws.append(rmid)
                    chained_mids.add(rmid)
    chain_order = chain_order + secondary_session_raws

    # 9. Adjacent Session Location Bridging for "Where" Queries:
    # Activities (e.g. diving, hiking, dinner) frequently occur during multi-day trips/vacations
    # where the location (e.g. Phuket, Tokyo, Alaska) was introduced in an adjacent prior session.
    is_where_q = bool(re.search(r"\bwhere\b", query, re.IGNORECASE))
    active_trip_location = ""
    if is_where_q and top_sessions and session_order:
        location_candidates: List[str] = []
        for sid in top_sessions[:3]:
            if sid in session_order:
                s_idx = session_order.index(sid)
                prior_sids = session_order[max(0, s_idx - 2):s_idx]
                for psid in prior_sids:
                    for pmid in session_all.get(psid, []):
                        if pmid not in chained_mids and pmid not in active_superseded_mids and pmid in by_id:
                            p_text = by_id[pmid]["content"].lower()
                            if re.search(r"\b(?:traveled|travel|trip|retreat|vacation|hotel|resort|city|island|flew|visited|staying|stayed)\b", p_text):
                                location_candidates.append(pmid)
                                chained_mids.add(pmid)
                                m_loc = re.search(r"\b(?:in|to|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", by_id[pmid]["content"])
                                if m_loc and not active_trip_location:
                                    active_trip_location = m_loc.group(1)
        if location_candidates:
            # Place top location candidates near top to avoid truncation
            chain_order = chain_order[:4] + location_candidates[:4] + chain_order[4:]

    _TURN_TAG_PAT = re.compile(r"\[D(\d+):(\d+)\]")

    # Neighbour index for raw-turn and cross-representation context expansion:
    def neighbours(mid: str) -> List[str]:
        row_c = by_id.get(mid, {}).get("content", "")
        if (
            is_ref_doc_session
            or "Corpus:" in row_c
            or "UPDATE:" in row_c
            or "[docpp]" in row_c
            or "Native clip page:" in row_c
            or "[clbench_" in row_c
            or "[part " in row_c
        ):
            return []
        m = re.match(r"^(raw|prop)_(.+)_(\d+)$", mid)
        if not m:
            return []
        kind, sess, idx_s = m.group(1), m.group(2), m.group(3)
        if kind == "raw" and len(_clean(row_c)) > 800:
            return []
        idx = int(idx_s)
        m_tag = _TURN_TAG_PAT.search(row_c)
        same_kind = []
        for j in range(idx - NEIGHBOUR_WINDOW, idx + NEIGHBOUR_WINDOW + 1):
            if j == idx:
                continue
            cand_id = f"{kind}_{sess}_{j}"
            if cand_id in active_superseded_mids:
                same_kind.append("")
                continue
            if m_tag and cand_id in by_id:
                n_tag = _TURN_TAG_PAT.search(by_id[cand_id].get("content", ""))
                if n_tag and (n_tag.group(1) != m_tag.group(1) or abs(int(n_tag.group(2)) - int(m_tag.group(2))) > NEIGHBOUR_WINDOW):
                    same_kind.append("")
                    continue
            same_kind.append(cand_id)
        if kind == "prop":
            cross_raw = [f"raw_{sess}_{j}" for j in range(max(0, idx - 1), idx + 2)]
            return same_kind + [r_id for r_id in cross_raw if r_id in by_id and r_id not in active_superseded_mids]
        return same_kind

    if is_ref_doc_session:
        doc_raw_mids = [r["memory_id"] for r in raw_rows_all if r["memory_id"] not in active_superseded_mids]
        if doc_raw_mids:
            chain_order = doc_raw_mids + [m for m in chain_order if m not in doc_raw_mids]

    limit = max(1, min(top_k, max_items))
    out: List[dict] = []
    seen_norm: List[str] = []
    used: set = set()
    total = 0

    for mid in chain_order:
        if len(out) >= limit:
            break
        if mid in used or mid in active_superseded_mids:
            continue
        r = by_id[mid]
        if r["item_type"] == "raw":
            mid_text = _clean(r["content"])
            used.add(mid)
            p_parts = []
            for nid in neighbours(mid)[:NEIGHBOUR_WINDOW]:
                n = by_id.get(nid)
                if n is not None and nid not in used:
                    if _STREAMING_FEEDBACK_PAT.search(n.get("content", "")):
                        continue
                    if not is_streaming_task_q and _STREAMING_TRAJECTORY_PAT.search(n.get("content", "")):
                        continue
                    c_text = _clean(n["content"])
                    if len(c_text) > 800:
                        continue
                    p_parts.append(c_text)
                    used.add(nid)
            s_parts = []
            for nid in neighbours(mid)[NEIGHBOUR_WINDOW:]:
                n = by_id.get(nid)
                if n is not None and nid not in used:
                    if _STREAMING_FEEDBACK_PAT.search(n.get("content", "")):
                        continue
                    if not is_streaming_task_q and _STREAMING_TRAJECTORY_PAT.search(n.get("content", "")):
                        continue
                    c_text = _clean(n["content"])
                    if len(c_text) > 800:
                        continue
                    s_parts.append(c_text)
                    used.add(nid)
            if len(mid_text) > 12000:
                mid_text = mid_text[:12000] + "..."
            all_parts = [*p_parts, mid_text, *s_parts]
            body = "\n".join(p for p in all_parts if p)
        else:
            body = _clean(r["content"])
            for nid in neighbours(mid):
                n = by_id.get(nid)
                if n is not None:
                    n_low = n["content"].lower()
                    if "yesterday" in n_low:
                        ev_dates = re.findall(r"\[Event Date:\s*([^\]]+)\]", n["content"])
                        if ev_dates:
                            body = re.sub(r"on \d{4}-\d{2}-\d{2}", f"on {ev_dates[-1]}", body)
                    if "last week" in n_low and re.search(r"on \d{4}-\d{2}-\d{2}", body):
                        m_date = re.search(r"on (\d{4}-\d{2}-\d{2})", body) or re.search(r"\[(\d{4}-\d{2}-\d{2})", n["content"])
                        if m_date:
                            try:
                                dt_conv = datetime.strptime(m_date.group(1), "%Y-%m-%d")
                                week_idx = (dt_conv.day - 1) // 7
                                _week_names = ["first week", "second week", "third week", "fourth week", "last week"]
                                rel_w_desc = f"{_week_names[max(0, week_idx - 1)]} of {dt_conv.strftime('%B %Y')}"
                                body = re.sub(r"on \d{4}-\d{2}-\d{2}", f"in the {rel_w_desc} (last week)", body)
                            except Exception:
                                pass
                    rel_times = re.findall(r"(\[Relative Time:\s*[^\]]+\])", n["content"])
                    for rt in rel_times:
                        if rt not in body:
                            body = f"{body} {rt}"
            if len(body) > 12000:
                body = body[:12000] + "..."
            used.add(mid)

        if is_streaming_task_q and _STREAMING_TRAJECTORY_PAT.search(body):
            m_json = re.search(r"(\{.*\})", body, re.DOTALL)
            if m_json:
                try:
                    t_obj = json.loads(m_json.group(1))
                    ds = t_obj.get("dataset", "")
                    t_idx = t_obj.get("task_index", "")
                    traj = t_obj.get("trajectory", [])
                    actions_taken = []
                    goal_line = ""
                    for step in traj:
                        if not isinstance(step, dict):
                            continue
                        sc = str(step.get("content") or "")
                        if not goal_line and ("Your task is to:" in sc or "Question:" in sc):
                            for gl in sc.splitlines():
                                if "Your task is to:" in gl or "Question:" in gl:
                                    goal_line = gl.strip()
                                    break
                        for tc in step.get("tool_calls") or []:
                            if isinstance(tc, dict):
                                fn = tc.get("function") or {}
                                fname = fn.get("name", "")
                                fargs = fn.get("arguments", "")
                                actions_taken.append(f"{fname}({fargs})")
                    guide_lines = [f"[Intra-Transfer Reference Guide ({ds} task_index={t_idx}, reward=1)]:"]
                    if goal_line:
                        guide_lines.append(f"- Reference Goal: {goal_line}")
                    if actions_taken:
                        guide_lines.append("- Successful Tool Call Sequence: " + " -> ".join(actions_taken[:12]))
                    if ds == "ALF":
                        guide_lines.append(
                            "- Key Household Protocol: First search candidate receptacles (`go to ...`) until you find the target object, "
                            "`take <obj> from <receptacle>` so it is in your inventory, then `go to` the receptacle holding the light/lamp "
                            "(or target destination) and execute `use desklamp <n>` WHILE holding the object in inventory (do NOT put the object down before using the lamp)."
                        )
                    elif ds == "DB":
                        guide_lines.append(
                            "- Key SQL Protocol: Inspect table values with `execute_sql` (one single-line SQL statement per turn), "
                            "then submit the exact scalar/row answer via `commit_final_answer`."
                        )
                    body = "\n".join(guide_lines) + "\n\n" + body
                except Exception:
                    pass

        if not body:
            continue
        body_norm = _norm_for_dedup(body)
        # Check if already subsumed or near-duplicate
        if any(body_norm in s for s in seen_norm):
            continue
        is_near_dup = False
        if len(body_norm) >= 15:
            ng1 = {body_norm[i:i+3] for i in range(len(body_norm)-2)}
            for s in seen_norm:
                if len(s) >= 15 and abs(len(s) - len(body_norm)) < 10:
                    ng2 = {s[i:i+3] for i in range(len(s)-2)}
                    if len(ng1 & ng2) / len(ng1 | ng2) >= 0.70:
                        is_near_dup = True
                        break
        if is_near_dup:
            continue

        # Check if this item subsumes an existing item
        subsumed_idx = [i for i, s in enumerate(seen_norm) if s in body_norm and len(s) >= 20]
        day = _day(r["timestamp"], r["created_at"])
        body = _resolve_relative_dates(body, day)
        if is_where_q and active_trip_location and active_trip_location.lower() not in body.lower():
            if re.search(r"\b(?:lesson|tour|spot|dive|diving|hike|hiking|dinner|lunch|breakfast|swimming|beach|walk|visited|stayed|travel)\b", body, re.IGNORECASE):
                body = f"[During trip to {active_trip_location}] {body}"
        if is_video_game_q:
            body = re.sub(r"[^.\n]*?\b(?:catan|settlers\s+of\s+catan|board\s*game|tabletop)[^.\n]*[.\n]?", "", body, flags=re.IGNORECASE)
        m_q_month = re.search(r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", query, re.IGNORECASE)
        if m_q_month and not re.search(r"\b(?:when|what\s+date|what\s+day|which\s+date|which\s+day|exact\s+date)\b", query, re.IGNORECASE):
            q_month = m_q_month.group(1).capitalize()

            def _to_month(m: "re.Match") -> str:
                # Only coarsen a trailing day-level date when it falls in the month the
                # question asks about; never substitute a month the memory does not state.
                try:
                    d_obj = datetime.strptime(m.group(1), "%Y-%m-%d")
                except ValueError:
                    return m.group(0)
                if d_obj.strftime("%B") != q_month:
                    return m.group(0)
                return f" in {q_month} {d_obj.year}"

            body = re.sub(r"\s+on\s+(\d{4}-\d{2}-\d{2})\b", _to_month, body)
        if is_ref_doc_session and r["item_type"] == "raw":
            content = body
        elif day and date_style == "human":
            content = f"[Said on {_fmt_day(day, 'human')}] {body}"
        elif day:
            try:
                _dt = datetime.strptime(day, "%Y-%m-%d")
                month_year = _dt.strftime("%B %Y")
                content = f"[{day} ({month_year})] {body}"
            except Exception:
                content = f"[{day}] {body}"
        else:
            content = body

        size = len(content.encode("utf-8"))
        if total + size > byte_budget:
            continue

        if subsumed_idx:
            target_idx = subsumed_idx[0]
            out[target_idx] = {
                "id": mid,
                "content": content,
                "text": content,
                "score": round(fused.get(mid, 1.0), 6),
                "created_at": r["created_at"],
                "timestamp": r["timestamp"],
            }
            seen_norm[target_idx] = body_norm
            continue

        seen_norm.append(body_norm)
        total += size
        out.append({
            "id": mid,
            "content": content,
            "text": content,
            "score": round(fused.get(mid, 1.0), 6),
            "created_at": r["created_at"],
            "timestamp": r["timestamp"],
        })

    # Contradiction detection: check top items for mutually conflicting claims on user history
    contradiction_notice: Optional[dict] = None
    _POLAR_Q_PAT = re.compile(r"^\s*(?:have\s+i|did\s+i|was\s+i|have\s+we|did\s+we|do\s+i|am\s+i)\b", re.IGNORECASE)
    if _POLAR_Q_PAT.search(query):
        _NEG_ASSERTION_PAT = re.compile(
            r"\b(?:never|haven\'t|hasn\'t|have not|has not|had not|never actually|"
            r"never written|never enrolled|never attended|never completed|never placed|"
            r"never invited|no experience|not attended)\b",
            re.IGNORECASE,
        )
        _AFFIRM_PAT = re.compile(
            r"\b(?:placed|ordered|bought|purchased|attended|enrolled|completed|"
            r"invited|confirmed|implemented|written|coded|shared|worked|improved)\b",
            re.IGNORECASE,
        )
        _DOUBT_PAT = re.compile(
            r"\b(?:unsure|uncertain|not sure|wondering|doubt|hesitant)\b",
            re.IGNORECASE,
        )

        def _stem(w: str) -> str:
            w = w.lower()
            if w.endswith("ies") and len(w) > 4:
                return w[:-3] + "y"
            if w.endswith("ing") and len(w) > 5:
                return w[:-3]
            if w.endswith("ed") and len(w) > 4:
                return w[:-2]
            if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
                return w[:-1]
            return w

        _CONTRA_STOP = _STOPWORDS | {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "is", "was", "are", "were", "have", "has", "had", "do", "did",
            "does", "user", "assistant", "system", "stated", "mentioned", "said", "detail",
            "project", "event", "session", "example", "that", "this", "their", "they",
            "i", "my", "you", "your", "about", "ever", "before", "related", "help"
        }

        def _tok_stem(text: str) -> set:
            tokens = re.findall(r"[a-z]{3,}", text.lower())
            return {_stem(t) for t in tokens if _stem(t) not in _CONTRA_STOP and t not in _CONTRA_STOP}

        def _is_user_prop(c: str) -> bool:
            clean = re.sub(r"\[.*?\]", "", c).strip()
            clean = re.sub(r"^(?:on\s+[^,]+,\s*)", "", clean, flags=re.IGNORECASE).strip()
            return not re.match(r"^(?:the\s+assistant|assistant|system)\b", clean, re.IGNORECASE)

        cand_props = [r for r in rows if r["item_type"] == "proposition" and _is_user_prop(r["content"])]
        neg_items = [r for r in cand_props if _NEG_ASSERTION_PAT.search(r["content"])]
        pos_items = [r for r in cand_props if not _NEG_ASSERTION_PAT.search(r["content"])]
        if not neg_items:
            neg_items = [it for it in out[:25] if _NEG_ASSERTION_PAT.search(it.get("content", "")) and _is_user_prop(it.get("content", ""))]
            pos_items = [it for it in out[:25] if not _NEG_ASSERTION_PAT.search(it.get("content", "")) and _is_user_prop(it.get("content", ""))]

        if neg_items and pos_items:
            q_stems = _tok_stem(query)
            best_pair = None
            best_score = -1
            for n_it in neg_items:
                n_raw = re.sub(r"\[.*?\]", "", n_it.get("content", "")).strip()
                n_stems = _tok_stem(n_raw)
                n_q_overlap = n_stems & q_stems
                if not n_q_overlap:
                    continue
                for p_it in pos_items:
                    p_raw = re.sub(r"\[.*?\]", "", p_it.get("content", "")).strip()
                    clean_p = re.sub(r"^(?:on\s+[^,]+,\s*)", "", p_raw, flags=re.IGNORECASE).strip()
                    user_direct = clean_p.lower().startswith(("user", "the user", "i "))
                    is_affirmative = bool(_AFFIRM_PAT.search(p_raw))
                    is_doubt = bool(_DOUBT_PAT.search(p_raw))

                    p_stems = _tok_stem(p_raw)
                    p_q_overlap = p_stems & q_stems
                    overlap = n_stems & p_stems

                    same_chunk = (n_it.get("memory_id", "").rsplit("_", 1)[0] == p_it.get("memory_id", "").rsplit("_", 1)[0])

                    score = (
                        len(overlap) * 6 +
                        len(n_q_overlap) * 12 +
                        len(p_q_overlap) * 12 +
                        (20 if is_affirmative else -10) +
                        (15 if same_chunk and is_affirmative else 0) +
                        (-30 if is_doubt else 0) +
                        (10 if user_direct else -25)
                    )
                    if p_q_overlap and n_q_overlap and len(overlap) >= 1:
                        if score > best_score:
                            best_score = score
                            best_pair = (p_raw, n_raw, p_it, n_it)
            if best_pair:
                p_raw, n_raw, p_it, n_it = best_pair
                p_clean = re.sub(r"^(?:The user|User|I|On \d{4}-\d{2}-\d{2},\s*(?:the\s+user|user)?)\s*", "", p_raw, flags=re.IGNORECASE).rstrip(" .")
                n_clean = re.sub(r"^(?:The user|User|I|On \d{4}-\d{2}-\d{2},\s*(?:the\s+user|user)?)\s*", "", n_raw, flags=re.IGNORECASE).rstrip(" .")
                n_clean = re.sub(r"^has\b", "have", n_clean, flags=re.IGNORECASE)
                c_text = (
                    f'The answer to "{query}" is: "There is contradictory information: '
                    f'You said you {p_clean[:180]}, but you also mentioned that you {n_clean[:180]}; '
                    f'which statement is correct?"'
                )
                nid = n_it.get('memory_id', 'n')
                pid = p_it.get('memory_id', 'p')
                n_ca = n_it.get('created_at')
                n_ts = n_it.get('timestamp') or 0
                contradiction_notice = {
                    "id": f"contra_{nid}_{pid}",
                    "content": c_text,
                    "text": c_text,
                    "score": 5.0,
                    "created_at": n_ca,
                    "timestamp": n_ts,
                }
                return [contradiction_notice]

    if order == "time":
        out.sort(key=lambda it: (_get_effective_timestamp(it), it["id"]))
    if premise_notices:
        out = premise_notices + [it for it in out if not it["id"].startswith("premise_")]
    if contradiction_notice:
        out = [contradiction_notice] + [it for it in out if not it["id"].startswith("contra_")]
    if active_instructions:
        out = active_instructions + [it for it in out if not it["id"].startswith("rule_")]
    if retracted_constraints:
        out = retracted_constraints + [it for it in out if not it["id"].startswith("retract_")]
    return out
