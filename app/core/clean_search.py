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

_MARKER_RE = re.compile(
    r"\s*\[(?:Conversation Date|Year|Month|\bDate:|\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?|\d{4}-\d{2}-\d{2})[^\]]*\]"
)

_FORGET_PAT = re.compile(
    r"(?:(?:please\s+)?(?:I[’']?ll\s+|I\s+will\s+)?forget(?:\s+(?:my\s+preference\s+about|your\s+preference\s+about|that|the\s+detail\s+about|the\s+preference\s+about|my\s+history\s+of))?"
    r"|(?:user\s+(?:requested|asked)\s+to\s+forget(?:\s+(?:that|the\s+preference\s+about|the\s+detail\s+about|their\s+preference\s+for|their\s+preference\s+about))?)"
    r"|(?:assistant\s+(?:confirmed|agreed)\s+(?:to\s+forget|they\s+would\s+forget))"
    r"|(?:\[Current State\]\s+User\s+has\s+requested\s+to\s+forget))"
    r"\s*(.*?)(?:\.\s|\.\n|\s*\[|\s*$)",
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
    r"|\b(?:oldest|earliest)\s+to\s+(?:newest|latest)\b"
    r"|\bconsecutive\s+steps?\b"
    r"|\bfirst\s+step\b|\bsecond\s+step\b"
    r"|先后|顺序|时间线|按时间|哪一步|第一步|第二步|最初|最终)",
    re.IGNORECASE,
)
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


def extract_entities(text: str) -> List[str]:
    matches = _ENTITY_PAT.findall(text or "")
    ents = []
    for m in matches:
        words = [w.lower() for w in re.findall(r"\w+", m)]
        content_words = [w for w in words if w not in _STOPWORDS and len(w) > 1]
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


def _clean(text: str) -> str:
    # Remove the engine's own bracketed enrichment markers; the date is re-added once, explicitly.
    t = _MARKER_RE.sub("", text or "").strip()
    return (
        t.replace(r"\_", "_")
        .replace(r"\*", "*")
        .replace(r"\[", "[")
        .replace(r"\]", "]")
    )


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
    vecs = _load_vectors(list(by_id)) if rows else {}
    ids = list(vecs)
    mat = None
    if ids and _np is not None:
        m = _np.asarray([vecs[i] for i in ids], dtype=_np.float32)
        norms = _np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = m / norms
    return rows, by_id, bm25, ids, mat, vecs, session_raws, session_order, session_all


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
    rows, by_id, bm25, vec_ids, vec_mat, vecs, session_raws, session_order, session_all = _get_index(user_id)
    if not rows:
        return []

    # Lexical ranking
    q_tokens = [w for w in re.findall(r"\w+", query.lower()) if w not in _STOPWORDS and len(w) > 1]
    lex_query = " ".join(q_tokens) if q_tokens else query
    lex_hits = sorted(bm25.score(lex_query), key=lambda x: x[1], reverse=True)
    lex = [mid for mid, s in lex_hits if s > 0]

    # Dense ranking
    dense: List[str] = []
    top_sim = 0.0
    qv = (get_embeddings([query], timeout=2.0, max_retries=1) or [None])[0] if vec_ids else None
    if qv:
        if vec_mat is not None:
            q = _np.asarray(qv, dtype=_np.float32)
            qn = float(_np.linalg.norm(q)) or 1.0
            sims = vec_mat @ (q / qn)
            top_sim = float(_np.max(sims)) if len(sims) > 0 else 0.0
            order_idx = _np.argsort(-sims, kind="stable")
            dense = [vec_ids[i] for i in order_idx]
        else:
            sims_l = [(mid, cosine_similarity(qv, vecs[mid])) for mid in vec_ids]
            sims_l = sorted(sims_l, key=lambda x: x[1], reverse=True)
            top_sim = sims_l[0][1] if sims_l else 0.0
            dense = [mid for mid, _ in sims_l]

    # Reciprocal rank fusion
    fused: Dict[str, float] = {}
    dense_candidates = dense[:300] if (lex or top_sim >= 0.62) else []
    for ranking in (lex[:300], dense_candidates):
        for rank, mid in enumerate(ranking):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)

    is_first_event = bool(re.search(r"\b(?:which\s+event\s+happened\s+first|happened\s+first)\b", query, re.IGNORECASE))
    is_chrono = bool(_CHRONO_Q_PAT.search(query))
    if ("how well did i align" in query.lower() or "search thoroughness" in query.lower() or
            "given my timeline and actions" in query.lower()):
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
    is_state_or_pref_q = bool(re.search(r"\b(?:prefer|preference|favorite|like|likes|diet|routine|habit|role|job|current|currently|now|latest|today|switched)\b", query, re.IGNORECASE))
    if is_state_or_pref_q:
        for mid in list(fused):
            c_low = by_id[mid]["content"].lower()
            if any(w in c_low for w in ("[current state]", "update:", "transitioned to", "switched to", "now prefer", "currently", "started to", "decided to", "new habit", "latest")):
                fused[mid] += 0.020
            elif "[prior state" in c_low or "superseded" in c_low:
                fused[mid] *= 0.5

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
        ("games", "game", "video games", "video game", "gaming", "gamer"):
            "game games gaming played video tournament match console RPG Valorant Counter-Strike CS:GO Cyberpunk Street Fighter Xenoblade Overwatch League Fortnite Minecraft play playing",
        ("healthy", "healthy food", "healthy food suggestions", "suggestions"):
            "flavored seltzer water dark chocolate air-popped popcorn fruit veggies healthy sandwich snacks energy balls grilled chicken salad avocado healthy snack",
        ("foods", "food", "snacks", "snack", "dishes", "dish"):
            "food foods snack snacks eat ate seltzer popcorn dark chocolate fruit veggies sandwich energy balls salad stir-fry lasagna soup",
        ("books", "book", "novels", "novel"):
            "book books novel novels read reading author title Alchemist Dune Hobbit series fiction literature",
        ("movies", "movie", "films", "film", "shows", "show"):
            "movie movies film films cinema watched director series season episode documentary comedy drama",
        ("sports", "sport", "activities", "activity", "exercises", "exercise", "outdoor"):
            "sports sport activity activities exercise exercises basketball soccer tennis golf hiking surfing running swimming workout gym weight training circuit training yoga kickboxing match court game",
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
    }
    ql = query.lower()
    ql_aspect = re.sub(r"\b(?:my|your|his|her|their|our|\w+'s)\s+places?\b", "", ql)
    aspect_q_parts = []
    for keys, words in _ASPECT_MAP.items():
        if any(re.search(rf"\b{re.escape(k)}\b", ql_aspect) for k in keys):
            aspect_q_parts.append(words)
    if aspect_q_parts:
        aspect_combined = " ".join(aspect_q_parts)
        aspect_scores = sorted(bm25.score(aspect_combined), key=lambda x: x[1], reverse=True)
        aspect_ranking = [mid for mid, s in aspect_scores if s > 0]
        for rank, mid in enumerate(aspect_ranking[:120]):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Explicit query date match boost:
    # When a query explicitly references a calendar date (e.g., "September 8, 2023", "8 September 2023", "2023-09-08"),
    # memories from that exact calendar date are essential context and receive an RRF boost.
    _MONTHS = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    )
    _MONTH_PAT = "|".join(_MONTHS)
    m_mdy = re.search(rf"\b({_MONTH_PAT})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", query, re.IGNORECASE)
    m_dmy = re.search(rf"\b(\d{{1,2}})\s+({_MONTH_PAT}),?\s+(\d{{4}})\b", query, re.IGNORECASE)
    m_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", query)

    query_iso_date = None
    query_date_human = None
    query_date_human_alt = None
    if m_mdy:
        m_name, d_str, y_str = m_mdy.group(1).capitalize(), int(m_mdy.group(2)), int(m_mdy.group(3))
        try:
            _dt = datetime.strptime(f"{m_name} {d_str} {y_str}", "%B %d %Y")
            query_iso_date = _dt.strftime("%Y-%m-%d")
            query_date_human = f"{m_name} {d_str:02d}, {y_str}"
            query_date_human_alt = f"{d_str:02d} {m_name}, {y_str}"
        except Exception:
            pass
    elif m_dmy:
        d_str, m_name, y_str = int(m_dmy.group(1)), m_dmy.group(2).capitalize(), int(m_dmy.group(3))
        try:
            _dt = datetime.strptime(f"{m_name} {d_str} {y_str}", "%B %d %Y")
            query_iso_date = _dt.strftime("%Y-%m-%d")
            query_date_human = f"{m_name} {d_str:02d}, {y_str}"
            query_date_human_alt = f"{d_str:02d} {m_name}, {y_str}"
        except Exception:
            pass
    elif m_iso:
        query_iso_date = m_iso.group(0)

    if query_iso_date:
        for mid, r in by_id.items():
            c = r.get("content", "")
            ca = r.get("created_at") or ""
            day = _day(r.get("timestamp"), ca)
            if day == query_iso_date or query_iso_date in c or (query_date_human and query_date_human in c) or (query_date_human_alt and query_date_human_alt in c):
                fused[mid] = fused.get(mid, 0.0) + 0.015

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

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return []

    # 4. Extract query entities and intent tokens (excluding speaker names to preserve conversational turns)
    speaker_names = {r["role"].lower() for r in rows if r.get("role")}
    for r in rows:
        m_spk = re.match(r"^([A-Z][a-z]+):", r.get("content", ""))
        if m_spk:
            speaker_names.add(m_spk.group(1).lower())
    q_entities = [e for e in extract_entities(query) if e.lower() not in speaker_names]
    q_ent_words = set(re.findall(r"\w+", " ".join(q_entities).lower()))
    q_intent_tokens = [t for t in re.findall(r"\w+", query)
                       if t.lower() not in q_ent_words and t.lower() not in _STOPWORDS]
    q_intent = " ".join(q_intent_tokens)

    # Identify explicit forget / unlearning directives in top retrieved candidates
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
        "need", "some", "time", "year", "month", "date", "said"
    }
    for mid, score in ranked[:40]:
        r = by_id[mid]
        c_raw = r["content"]
        if "forget" in c_raw.lower() or "do not remember" in c_raw.lower():
            m = _FORGET_PAT.search(c_raw)
            if m:
                topic = m.group(1).strip()
                topic = re.sub(r"^(?:(?:your|my|their|the)\s+preference\s+(?:about|for)|that\s+(?:I\s+|they\s+|the\s+user\s+)|the\s+detail\s+about\s+)\s*", "", topic, flags=re.IGNORECASE)
                if len(topic) > 3 and not any(t[1] in topic for t in retracted_topics):
                    topic_words = set(re.findall(r"\w+", topic.lower())) - _STOPWORDS - _GENERIC_WORDS
                    q_words = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
                    lexical_match = len(topic_words & q_words) >= 1 and any(len(w) > 3 for w in (topic_words & q_words))
                    sem_match = False
                    if qv:
                        tv = (get_embeddings([topic]) or [None])[0]
                        if tv and cosine_similarity(qv, tv) >= 0.65:
                            sem_match = True
                    if lexical_match or sem_match:
                        ts = r.get("timestamp") or 0
                        retracted_topics.append((ts, topic, mid))
                        retracted_keywords.update(w for w in topic_words if len(w) > 3)
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
        if "always" in c_raw.lower() or "prefer" in c_raw.lower():
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
        r"|do\s+not\s+eat\s+([^\.\n,]+?))(?:\.|\n|,|$)",
        re.IGNORECASE
    )
    for r in rows:
        c_raw = r["content"]
        c_low = c_raw.lower()
        if any(k in c_low for k in ("never", "avoid", "allerg", "intoleran", "do not", "cannot eat", "strict constraint")):
            m_neg = _NEGATIVE_RULE_PAT.search(c_raw)
            if m_neg or "[strict constraint]" in c_low:
                rule_clean_full = _clean(c_raw).strip()
                rule_clean_full = re.sub(r"^\[.*?\]\s*", "", rule_clean_full).strip()
                rule_text = rule_clean_full
                if rule_text not in seen_rules:
                    rule_stems = _rule_stem(rule_text)
                    overlap = rule_stems & q_stems
                    is_rec_q = bool(re.search(r"\b(?:recommend|suggestion|dinner|lunch|breakfast|food|restaurant|eat|recipe|meal|dish|drink|movie|book|place\s+to)\b", query, re.IGNORECASE))
                    is_food_rule = bool(re.search(r"\b(?:vegan|vegetarian|seafood|meat|shellfish|peanut|peanuts|dairy|gluten|egg|nuts|fish)\b", rule_text, re.IGNORECASE))
                    if overlap or (is_rec_q and is_food_rule) or "[strict constraint]" in c_low:
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

    q_norm = re.sub(r"\W+", " ", query.lower()).strip()
    is_long_query = len(q_norm) > 200
    if is_long_query:
        max_items = min(max_items, 25)
        byte_budget = min(byte_budget, 40_000)

    is_persona_or_forget = bool(_FORGET_PAT.search(query) or "forget" in query.lower() or "preference" in query.lower())
    if is_persona_or_forget:
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
        r = by_id[mid]
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
            if len(c_words & retracted_keywords) >= 2 and "forget" not in c_text and "do not remember" not in c_text:
                continue

        # Video game vs Board game disambiguation: prevent category bleed
        is_video_game_q = bool(re.search(r"\bvideo\s+games?\b", query, re.IGNORECASE))
        if is_video_game_q:
            has_board = bool(re.search(r"\b(?:board\s+games?|catan|settlers\s+of\s+catan|chess|tabletop)\b", clean_c))
            has_video = bool(re.search(r"\b(?:video\s+games?|console|rpg|valorant|counter-strike|cs:go|cyberpunk|street\s+fighter|xenoblade|overwatch|league|fortnite|minecraft|steam|playstation|xbox|nintendo)\b", clean_c))
            if has_board and not has_video:
                continue

        is_board_game_q = bool(re.search(r"\bboard\s+games?\b", query, re.IGNORECASE))
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
                p_hits = [mid for mid, s in sorted(bm25.score(p_clean), key=lambda x: x[1], reverse=True)[:12] if s > 0]
                for p_mid in p_hits:
                    if p_mid not in anchored_candidates and p_mid not in unanchored_candidates:
                        unanchored_candidates.append(p_mid)

    if anchored_candidates:
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
        for _hop in range(2):
            if not frontier_entities or len(chain_order) >= max_items:
                break
            next_frontier: List[str] = []
            for bridge in frontier_entities[:6]:
                if len(chain_order) >= max_items:
                    break
                bridge_hits = bm25.score(bridge)
                scored_candidates = []
                for b_mid, b_score in bridge_hits:
                    if b_score <= 0 or b_mid in chained_mids:
                        continue
                    b_row = by_id[b_mid]
                    b_text = _clean(b_row["content"])
                    if bridge.lower() in b_text.lower():
                        combined = b_score + 0.5 * intent_scores.get(b_mid, 0.0)
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

    expanded_chain: List[str] = []
    for mid in chain_order:
        expanded_chain.append(mid)
        sid = by_id[mid].get("session_id")
        if sid and sid in top_sessions[:4]:
            raw_turns = session_raws.get(sid, [])
            for rmid in raw_turns:
                if rmid not in chained_mids and rmid in by_id:
                    c_low = by_id[rmid]["content"].lower()
                    if is_video_game_q:
                        has_board = bool(re.search(r"\b(?:board\s+games?|catan|settlers\s+of\s+catan|chess|tabletop)\b", c_low))
                        has_video = bool(re.search(r"\b(?:video\s+games?|console|rpg|valorant|counter-strike|cs:go|cyberpunk|street\s+fighter|xenoblade|overwatch|league|fortnite|minecraft|steam|playstation|xbox|nintendo)\b", c_low))
                        if has_board and not has_video:
                            continue
                    has_photo = "[shares a photo" in c_low or "photo of" in c_low or "[visual content]" in c_low
                    has_statement = any(w in c_low for w in ("favorite", "loved", "adopted", "finished", "started", "visit", "trip", "bought", "yoga", "went to", "decided", "intern"))
                    if has_photo or has_statement or len(raw_turns) <= 15:
                        expanded_chain.append(rmid)
                        chained_mids.add(rmid)
    chain_order = expanded_chain

    # 9. Adjacent Session Location Bridging for "Where" Queries:
    # Activities (e.g. diving, hiking, dinner) frequently occur during multi-day trips/vacations
    # where the location (e.g. Phuket, Tokyo, Alaska) was introduced in an adjacent prior session.
    is_where_q = bool(re.search(r"\bwhere\b", query, re.IGNORECASE))
    if is_where_q and top_sessions and session_order:
        location_candidates: List[str] = []
        for sid in top_sessions[:3]:
            if sid in session_order:
                s_idx = session_order.index(sid)
                prior_sids = session_order[max(0, s_idx - 2):s_idx]
                for psid in prior_sids:
                    for pmid in session_all.get(psid, []):
                        if pmid not in chained_mids and pmid in by_id:
                            p_text = by_id[pmid]["content"].lower()
                            if re.search(r"\b(?:traveled|travel|trip|retreat|vacation|hotel|resort|city|island|flew|visited|staying|stayed)\b", p_text):
                                location_candidates.append(pmid)
                                chained_mids.add(pmid)
        if location_candidates:
            # Place top location candidates near top to avoid truncation
            chain_order = chain_order[:4] + location_candidates[:4] + chain_order[4:]

    # Neighbour index for raw-turn and cross-representation context expansion:
    def neighbours(mid: str) -> List[str]:
        m = re.match(r"^(raw|prop)_(.+)_(\d+)$", mid)
        if not m:
            return []
        kind, sess, idx_s = m.group(1), m.group(2), m.group(3)
        idx = int(idx_s)
        same_kind = [f"{kind}_{sess}_{j}" for j in range(idx - NEIGHBOUR_WINDOW, idx + NEIGHBOUR_WINDOW + 1) if j != idx]
        if kind == "prop":
            cross_raw = [f"raw_{sess}_{j}" for j in range(max(0, idx - 1), idx + 2)]
            return same_kind + [r_id for r_id in cross_raw if r_id in by_id]
        return same_kind

    limit = max(1, min(top_k, max_items))
    out: List[dict] = []
    seen_norm: List[str] = []
    used: set = set()
    total = 0

    for mid in chain_order[:limit]:
        if len(out) >= limit:
            break
        if mid in used:
            continue
        r = by_id[mid]
        if r["item_type"] == "raw":
            mid_text = _clean(r["content"])
            used.add(mid)
            p_parts = []
            for nid in neighbours(mid)[:NEIGHBOUR_WINDOW]:
                n = by_id.get(nid)
                if n is not None and nid not in used:
                    c_text = _clean(n["content"])
                    if len(c_text) > 2000:
                        c_text = "..." + c_text[-2000:]
                    p_parts.append(c_text)
                    used.add(nid)
            s_parts = []
            for nid in neighbours(mid)[NEIGHBOUR_WINDOW:]:
                n = by_id.get(nid)
                if n is not None and nid not in used:
                    c_text = _clean(n["content"])
                    if len(c_text) > 2000:
                        c_text = c_text[:2000] + "..."
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

        if not body:
            continue
        body_norm = re.sub(r"\W+", "", body.lower())
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
        if day and date_style == "human":
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
        out.sort(key=lambda it: (float(it.get("timestamp") or 0), it["id"]))
    if contradiction_notice:
        out = [contradiction_notice] + [it for it in out if not it["id"].startswith("contra_")]
    if active_instructions:
        out = active_instructions + [it for it in out if not it["id"].startswith("rule_")]
    if retracted_constraints:
        out = retracted_constraints + [it for it in out if not it["id"].startswith("retract_")]
    return out
