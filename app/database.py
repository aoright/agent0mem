import os
import sqlite3
import json
import time
import re
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from app.config import (
    DB_PATH,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RECENCY_BOOST_MAX,
)
from app.core.bm25 import BM25Index
from app.core.embeddings import get_embeddings, cosine_similarity

logger = logging.getLogger("agent0mem.database")

PAST_QUERY_PATTERN = re.compile(
    r"\b(used to|previously|originally|before|earlier|formerly|past|initial|initially|prior to|history)\b",
    re.IGNORECASE
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT UNIQUE,
                request_id TEXT,
                user_id TEXT,
                session_id TEXT,
                item_type TEXT, -- 'raw' or 'proposition'
                role TEXT,
                content TEXT,
                timestamp INTEGER,
                created_at TEXT
            )
        """)
        # Auto-migrate columns if table already existed
        cursor = conn.execute("PRAGMA table_info(memories)")
        cols = {row["name"] for row in cursor.fetchall()}
        if "item_type" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN item_type TEXT DEFAULT 'raw'")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                memory_id TEXT PRIMARY KEY,
                vector TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_time ON memories(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(item_type)")
        conn.commit()


def normalize_timestamp_to_iso(ts: Optional[int]) -> Tuple[int, str]:
    """Convert raw integer timestamp (s or ms) to int and standard ISO string."""
    if not ts:
        now_sec = time.time()
        now_ms = int(now_sec * 1000)
        iso = datetime.fromtimestamp(now_sec, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return now_ms, iso

    t = float(ts)
    if t > 1e11:  # Milliseconds
        sec = t / 1000.0
        ts_ms = int(t)
    else:  # Seconds
        sec = t
        ts_ms = int(t * 1000)

    try:
        iso = datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        iso = datetime.fromtimestamp(time.time(), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ts_ms, iso


def save_memories_batch(
    request_id: str,
    user_id: str,
    session_id: str,
    raw_messages: List[Dict[str, Any]],
    propositions: List[str]
) -> int:
    """Save raw turns and distilled propositions with their embeddings."""
    records_to_insert = []
    texts_for_embed = []
    mem_ids_for_embed = []

    # Determine reference timestamp from messages
    latest_msg_ts = None
    for msg in raw_messages:
        ts = msg.get("timestamp")
        if ts and (latest_msg_ts is None or ts > latest_msg_ts):
            latest_msg_ts = ts

    prop_ts_int, prop_iso = normalize_timestamp_to_iso(latest_msg_ts)

    # 1. Distilled propositions (Highest priority, conflict-governed)
    for p_idx, prop in enumerate(propositions):
        prop_id = f"prop_{request_id}_{p_idx}"
        records_to_insert.append((
            prop_id, request_id, user_id, session_id, "proposition", "system", prop, prop_ts_int + p_idx, prop_iso
        ))
        texts_for_embed.append(prop)
        mem_ids_for_embed.append(prop_id)

    # 2. Raw message turns (Episodic conversational context)
    for m_idx, msg in enumerate(raw_messages):
        role = msg.get("role", "user")
        content_val = msg.get("content", "")
        if isinstance(content_val, list):
            parts = [p.get("text", "") for p in content_val if isinstance(p, dict) and p.get("type") == "text"]
            content_str = " ".join(parts) if parts else json.dumps(content_val)
        else:
            content_str = str(content_val)

        raw_id = f"raw_{request_id}_{m_idx}"
        msg_ts_int, msg_iso = normalize_timestamp_to_iso(msg.get("timestamp"))
        records_to_insert.append((
            raw_id, request_id, user_id, session_id, "raw", role, content_str, msg_ts_int, msg_iso
        ))
        texts_for_embed.append(content_str)
        mem_ids_for_embed.append(raw_id)

    # Batch compute embeddings
    embeddings = get_embeddings(texts_for_embed)

    with get_connection() as conn:
        for rec in records_to_insert:
            conn.execute("""
                INSERT OR REPLACE INTO memories 
                (memory_id, request_id, user_id, session_id, item_type, role, content, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rec)

        for mem_id, emb in zip(mem_ids_for_embed, embeddings):
            if emb:
                conn.execute("""
                    INSERT OR REPLACE INTO embeddings (memory_id, vector)
                    VALUES (?, ?)
                """, (mem_id, json.dumps(emb)))

        conn.commit()

    return len(records_to_insert)


def search_hybrid(
    user_id: str,
    query_text: str,
    options: Optional[List[str]] = None,
    top_k: int = 100
) -> List[Dict[str, Any]]:
    """State-of-the-art hybrid search combining BM25, Dense Vector, Recency, Option Bonus, and Conflict Disambiguation."""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT m.memory_id, m.content, m.item_type, m.timestamp, m.created_at, e.vector
            FROM memories m
            LEFT JOIN embeddings e ON m.memory_id = e.memory_id
            WHERE m.user_id = ?
            ORDER BY m.timestamp ASC
        """, (user_id,))
        rows = cursor.fetchall()

        # Fallback to substring matching if exact match yields no records
        if not rows:
            cursor = conn.execute("""
                SELECT m.memory_id, m.content, m.item_type, m.timestamp, m.created_at, e.vector
                FROM memories m
                LEFT JOIN embeddings e ON m.memory_id = e.memory_id
                WHERE m.user_id LIKE '%' || ? || '%'
                ORDER BY m.timestamp ASC
            """, (user_id,))
            rows = cursor.fetchall()

    if not rows:
        return []

    # Detect if query asks for past/superseded states
    is_past_query = bool(PAST_QUERY_PATTERN.search(query_text))

    # Process options for option-aware scoring (without polluting base BM25 with all distractors)
    clean_options = []
    if options:
        for opt in options:
            if isinstance(opt, dict):
                opt_str = opt.get("text", "") or opt.get("content", "") or str(opt)
            else:
                opt_str = str(opt)
            cleaned = re.sub(r"^\s*(?:\([A-Za-z]\)|[A-Za-z][\.:])\s*", "", opt_str).strip()
            if len(cleaned) > 1:
                clean_options.append(cleaned.lower())

    # 1. BM25 scoring on question query
    doc_tuples = [(row["memory_id"], row["content"]) for row in rows]
    bm25 = BM25Index()
    bm25.fit(doc_tuples)
    bm25_scores = dict(bm25.score(query_text))

    # Normalize BM25 scores to [0, 1]
    max_bm25 = max(bm25_scores.values()) if bm25_scores and max(bm25_scores.values()) > 0 else 1.0
    for k in bm25_scores:
        bm25_scores[k] = bm25_scores[k] / max_bm25

    # 2. Dense vector scoring
    query_embed_list = get_embeddings([query_text])
    query_vec = query_embed_list[0] if query_embed_list and query_embed_list[0] else None

    # 3. Temporal calculation
    timestamps = [row["timestamp"] for row in rows if row["timestamp"]]
    min_ts = min(timestamps) if timestamps else 0
    max_ts = max(timestamps) if timestamps else 1
    ts_range = (max_ts - min_ts) if max_ts > min_ts else 1

    scored_items = []
    for row in rows:
        m_id = row["memory_id"]
        content = row["content"]
        content_lower = content.lower()
        item_type = row["item_type"]
        ts = row["timestamp"] or 0
        raw_vec = row["vector"]

        # BM25 component
        b_score = bm25_scores.get(m_id, 0.0)

        # Dense component
        d_score = 0.0
        if query_vec and raw_vec:
            try:
                vec = json.loads(raw_vec)
                d_score = max(0.0, cosine_similarity(query_vec, vec))
            except Exception:
                d_score = 0.0

        # Recency boost
        if not is_past_query:
            recency_norm = max(0.0, min(1.0, (ts - min_ts) / ts_range))
            recency_boost = recency_norm * RECENCY_BOOST_MAX
        else:
            recency_boost = 0.0  # Do not penalize older memories if asking about the past

        # Base composite score
        base_score = (BM25_WEIGHT * b_score) + (DENSE_WEIGHT * d_score) + recency_boost

        # Proposition preference boost
        if item_type == "proposition":
            base_score += 0.15

        # Conflict governance
        if "[Current State]" in content:
            if not is_past_query:
                base_score += 0.18
            else:
                base_score -= 0.10
        elif "[Prior State / Superseded]" in content:
            if not is_past_query:
                base_score *= 0.60  # Severely downweight superseded memories for present queries
            else:
                base_score += 0.20  # Boost superseded memories when specifically asked about past

        # Option bonus: if this memory explicitly mentions any candidate option, reward it
        if clean_options:
            matched_opts = sum(1 for opt_word in clean_options if opt_word in content_lower)
            if matched_opts > 0:
                base_score += min(0.20, 0.10 * matched_opts)

        scored_items.append({
            "id": m_id,
            "content": content,
            "text": content,
            "score": round(float(base_score), 4),
            "created_at": row["created_at"],
            "timestamp": ts,
            "item_type": item_type
        })

    # Sort descending by composite score
    scored_items.sort(key=lambda x: x["score"], reverse=True)

    # Light deduplication: if an item is virtually identical to a higher ranked proposition, drop it
    deduped = []
    seen_texts = []
    for item in scored_items:
        text = item["content"].strip().lower()
        # Check simple token overlap with already accepted top items
        tokens = set(re.findall(r"\w+", text))
        is_duplicate = False
        if len(tokens) >= 4:
            for accepted_tokens in seen_texts:
                intersection = tokens.intersection(accepted_tokens)
                jaccard = len(intersection) / float(len(tokens.union(accepted_tokens)))
                if jaccard > 0.85:
                    is_duplicate = True
                    break
        if not is_duplicate:
            deduped.append(item)
            seen_texts.append(tokens)

    return deduped[:top_k]
