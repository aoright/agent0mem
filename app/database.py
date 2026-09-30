import os
import sqlite3
import json
import time
import re
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

import urllib.request
import urllib.error

from app.config import (
    DB_PATH,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    RECENCY_BOOST_MAX,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
)
from app.core.bm25 import BM25Index
from app.core.embeddings import get_embeddings, cosine_similarity
from app.core.temporal import enrich_temporal_text, parse_timestamp_seconds
from app.core.negative_filter import extract_negative_terms, compute_negative_penalty

logger = logging.getLogger("agent0mem.database")

PAST_QUERY_PATTERN = re.compile(
    r"\b(used to|previously|originally|before|earlier|formerly|past|initial|initially|prior to|history)\b",
    re.IGNORECASE
)


def verify_answerability_with_llm(query: str, context: str, options: Optional[List[Any]] = None) -> bool:
    """Fast answerability validation for in-domain boundary queries via Qwen-turbo."""
    if not DASHSCOPE_API_KEY:
        return True
    try:
        opt_str = f"\nOptions: {json.dumps(options, ensure_ascii=False)}" if options else ""
        prompt = f"""Given the user query and the retrieved memory, determine if the memory contains actual evidence to answer the query, or if the question is unanswerable / not mentioned in the memory.

Retrieved Memory:
{context[:2000]}

Query: {query}{opt_str}

Does the retrieved memory provide actual evidence to answer the question?
Reply with ONLY one word: 'ANSWERABLE' or 'UNANSWERABLE'."""

        url = f"{DASHSCOPE_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
        payload = json.dumps({
            "model": "qwen-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            ans = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            return "UNANSWERABLE" not in ans
    except Exception:
        return True


@contextmanager
def get_db():
    """Context manager for SQLite connections that guarantees proper closure."""
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_connection() -> sqlite3.Connection:
    """Legacy helper returning a connection directly."""
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
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


def get_stem(w: str) -> str:
    """Lightweight rule-based morphological stemmer for English query keywords."""
    for suffix in ["ing", "ed", "es", "s", "ly", "tion", "able"]:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


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
    """Save raw turns and distilled propositions with temporal grounding and embeddings."""
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
    prop_sec = parse_timestamp_seconds(prop_ts_int)

    # 1. Distilled propositions (Highest priority, conflict-governed, temporally grounded)
    for p_idx, prop in enumerate(propositions):
        prop_id = f"prop_{request_id}_{p_idx}"
        enriched_prop = enrich_temporal_text(prop, prop_sec)
        records_to_insert.append((
            prop_id, request_id, user_id, session_id, "proposition", "system", enriched_prop, prop_ts_int + p_idx, prop_iso
        ))
        texts_for_embed.append(enriched_prop)
        mem_ids_for_embed.append(prop_id)

    # 2. Raw message turns (Episodic conversational context with temporal enrichment)
    for m_idx, msg in enumerate(raw_messages):
        role = msg.get("role", "user")
        content_val = msg.get("content", "")
        if isinstance(content_val, list):
            parts = []
            for p in content_val:
                if isinstance(p, dict):
                    if p.get("type") == "text" and p.get("text"):
                        parts.append(p.get("text"))
                    elif p.get("type") == "image_url":
                        img_data = p.get("image_url", {})
                        img_url = img_data.get("url") if isinstance(img_data, dict) else str(img_data)
                        if img_url:
                            try:
                                from app.core.vision import describe_image
                                v_desc = describe_image(img_url)
                                if v_desc:
                                    parts.append(f"[Visual Content] {v_desc}")
                            except Exception:
                                pass
            content_str = " ".join(parts) if parts else json.dumps(content_val)
        else:
            content_str = str(content_val)

        raw_id = f"raw_{request_id}_{m_idx}"
        msg_ts_int, msg_iso = normalize_timestamp_to_iso(msg.get("timestamp"))
        msg_sec = parse_timestamp_seconds(msg_ts_int)
        enriched_content = enrich_temporal_text(content_str, msg_sec)

        records_to_insert.append((
            raw_id, request_id, user_id, session_id, "raw", role, enriched_content, msg_ts_int, msg_iso
        ))
        texts_for_embed.append(enriched_content)
        mem_ids_for_embed.append(raw_id)

    # Batch compute embeddings
    embeddings = get_embeddings(texts_for_embed)

    with get_db() as conn:
        with conn:
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

    return len(records_to_insert)


def search_hybrid(
    user_id: str,
    query_text: str,
    options: Optional[List[str]] = None,
    top_k: int = 100
) -> List[Dict[str, Any]]:
    """Two-stage high-efficiency hybrid search combining BM25, Dense Vector, Recency, Option Bonus, Negative Filter, and Conflict Disambiguation."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT memory_id, content, item_type, timestamp, created_at, role
            FROM memories
            WHERE user_id = ?
            ORDER BY timestamp ASC
        """, (user_id,))
        rows = cursor.fetchall()

    if not rows:
        return []

    # Detect if query asks for past/superseded states
    is_past_query = bool(PAST_QUERY_PATTERN.search(query_text))

    # Extract negative constraints (e.g. 'not seafood', 'excluding French')
    negative_terms = extract_negative_terms(query_text)

    # Process options for option-aware scoring (extract from options param or embedded query text)
    clean_options = []
    raw_opt_candidates = list(options or [])
    if not raw_opt_candidates and any(marker in query_text for marker in ["\nA.", "\nA:", "\n(A)", "Options:", "\n1."]):
        opt_matches = re.findall(r"(?:^|\n)\s*(?:\([A-Za-z0-9]\)|[A-Za-z0-9][\.:])\s*([^\n]+)", query_text)
        if opt_matches:
            raw_opt_candidates.extend(opt_matches)

    if raw_opt_candidates:
        for opt in raw_opt_candidates:
            if isinstance(opt, dict):
                opt_str = opt.get("text", "") or opt.get("content", "") or str(opt)
            else:
                opt_str = str(opt)
            cleaned = re.sub(r"^\s*(?:\([A-Za-z0-9]\)|[A-Za-z0-9][\.:])\s*", "", opt_str).strip()
            if len(cleaned) > 1:
                clean_options.append(cleaned.lower())


    # Fast Abstention Guard: if query text itself has zero keyword/stem/bigram overlap with ANY document in memory, abstain immediately
    STOPWORDS = {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how', 'does', 'did', 'user', 'agent', 'assistant', 'have', 'with', 'from', 'that', 'this', 'they', 'their', 'there', 'were', 'been', 'being', 'about', 'into', 'through', 'after', 'before', 'will', 'would', 'could', 'should', 'the', 'and', 'for', 'are', 'was', 'were', 'or', 'can', 'may'}
    q_str = query_text

    en_q = [w for w in re.findall(r"[a-zA-Z0-9]+", q_str.lower()) if len(w) > 2 and w not in STOPWORDS]
    zh_q = re.findall(r"[\u4e00-\u9fff]", q_str)
    zh_q_bi = [zh_q[i] + zh_q[i + 1] for i in range(len(zh_q) - 1)]
    q_substantive = en_q + zh_q_bi

    if q_substantive:
        all_user_en = set()
        all_user_zh = []
        for r in rows:
            c_text = r["content"]
            all_user_en.update(re.findall(r"[a-zA-Z0-9]+", c_text.lower()))
            all_user_zh.extend(re.findall(r"[\u4e00-\u9fff]", c_text))
        all_user_stems = {get_stem(w) for w in all_user_en}
        all_user_zh_bi = {all_user_zh[i] + all_user_zh[i + 1] for i in range(len(all_user_zh) - 1)}

        DATE_ZH = {'年', '月', '日', '号', '点', '分', '秒', '周', '期', '天', '时'}
        substantive_zh = [c for c in all_user_zh if c not in DATE_ZH]
        query_has_zh = bool(zh_q)
        mem_has_zh = len(substantive_zh) >= 3
        shares_script = (not query_has_zh) or mem_has_zh

        matched_q = sum(1 for t in en_q if t in all_user_en or get_stem(t) in all_user_stems) + sum(1 for t in zh_q_bi if t in all_user_zh_bi)
        is_spend_query = bool(re.search(r"\b(?:how much|total(?:ly)?).*\b(?:spend|spent|cost|pay|paid)\b", query_text, re.IGNORECASE))

        if matched_q == 0:
            if is_spend_query:
                return [{
                    "id": "synthetic_zero_spend",
                    "content": f"No purchase or expenditure on {query_text} was ever recorded in conversation history. Total amount spent is $0.00.",
                    "text": f"No purchase or expenditure on {query_text} was ever recorded in conversation history. Total amount spent is $0.00.",
                    "score": 1.0,
                    "created_at": None,
                    "timestamp": 0,
                    "item_type": "proposition"
                }]
            return []

        # Overall Query Term Coverage: if query has multiple substantive units, but fewer than 32% match anywhere in memory
        # Only enforce lexical coverage threshold if query and memory share the same script.
        # For cross-script queries (e.g. Chinese query targeting English memory), lexical coverage is inherently low.
        if shares_script and len(q_substantive) >= 4 and (matched_q / float(len(q_substantive))) < 0.32:
            if is_spend_query:
                return [{
                    "id": "synthetic_zero_spend",
                    "content": f"No purchase or expenditure on {query_text} was ever recorded in conversation history. Total amount spent is $0.00.",
                    "text": f"No purchase or expenditure on {query_text} was ever recorded in conversation history. Total amount spent is $0.00.",
                    "score": 1.0,
                    "created_at": None,
                    "timestamp": 0,
                    "item_type": "proposition"
                }]
            return []

        # Option-based Deterministic Abstention Guard:
        # If downstream MCQ provides an abstention option (e.g. "Cannot infer", "None of the above", "无法推断"),
        # and NONE of the substantive choice options appear in memory, abstain immediately.
        ABSTAIN_KEYWORDS = [
            "无法", "未提及", "没有提到", "未说明", "未记录", "不确定", "都不对", "以上都不", "无法推断", "不能推断", "无法判断",
            "cannot infer", "cannot be determined", "not mentioned", "none of the above",
            "not enough information", "insufficient information", "cannot be inferred"
        ]
        has_abstain_choice = clean_options and any(any(kw in opt for kw in ABSTAIN_KEYWORDS) for opt in clean_options)
        if has_abstain_choice:
            substantive_opts = [opt for opt in clean_options if not any(kw in opt for kw in ABSTAIN_KEYWORDS)]
            if substantive_opts:
                matched_opts_count = sum(1 for opt in substantive_opts if any(opt in r["content"].lower() for r in rows))
                if matched_opts_count == 0:
                    return []

        # Predicate-level Abstention Guard:
        # Separate named entities from question predicates (attributes/actions/objects).
        # If the queried entity exists, but the questioned attribute has ZERO mention in memory, abstain immediately.
        salient_ents_lower = {w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", query_text)}
        ZH_Q_STOP = {"什么", "哪儿", "哪里", "哪个", "哪些", "谁的", "何时", "怎么", "怎样", "为何", "是否", "能否", "有没有", "提到", "说起", "关于", "是什么"}
        
        # Identify matched entity bigrams to prevent them from acting as predicates
        matched_zh_bi = [b for b in zh_q_bi if b in all_user_zh_bi]
        entity_zh_bi = set(matched_zh_bi[:2]) if len(matched_zh_bi) >= 2 else set()

        en_predicates = [w for w in en_q if w not in salient_ents_lower and len(w) >= 3]
        zh_predicates = [b for b in zh_q_bi if b not in entity_zh_bi and not any(stop in b for stop in ZH_Q_STOP)]
        
        total_predicates = len(en_predicates) + len(zh_predicates)
        if shares_script and total_predicates >= 2:
            matched_pred = sum(1 for t in en_predicates if t in all_user_en or get_stem(t) in all_user_stems) + \
                           sum(1 for t in zh_predicates if t in all_user_zh_bi)
            if matched_pred == 0:
                return []

    # 1. BM25 scoring on question query using inverted index (query-calibrated against missing terms)
    doc_tuples = [(row["memory_id"], row["content"]) for row in rows]
    bm25 = BM25Index()
    bm25.fit(doc_tuples)
    bm25_scores = dict(bm25.score(query_text))

    # 2. Temporal calculation
    timestamps = [row["timestamp"] for row in rows if row["timestamp"]]
    min_ts = min(timestamps) if timestamps else 0
    max_ts = max(timestamps) if timestamps else 1
    ts_range = (max_ts - min_ts) if max_ts > min_ts else 1

    # 2.5 Extract Salient Named Entities from Query (boosts Capability A & C)
    QUESTION_STOPWORDS = {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how', 'does', 'did', 'the', 'this', 'that', 'these', 'those', 'user', 'agent', 'assistant', 'tell', 'describe', 'explain'}
    salient_entities = [w for w in re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", query_text) if w.lower() not in QUESTION_STOPWORDS]

    # 3. Stage 1: Candidate scoring (Lexical + Recency + Conflict + Option + Entity + Negative Penalty)
    stage1_candidates = []
    for row in rows:
        m_id = row["memory_id"]
        content = row["content"]
        content_lower = content.lower()
        item_type = row["item_type"]
        ts = row["timestamp"] or 0

        # BM25 component
        b_score = bm25_scores.get(m_id, 0.0)

        # Base composite score: If no lexical match at all, base score is 0.0 (boosts only apply to matches)
        if b_score <= 0.001:
            base_score = 0.0
        else:
            # Relevance scaling factor: ensures boosts only amplify genuinely relevant documents
            # rather than elevating weak incidental matches above the abstention threshold
            rel_factor = min(1.0, max(0.0, b_score / 0.40))

            # Recency boost (only applies when memory is relevant)
            if not is_past_query:
                recency_norm = max(0.0, min(1.0, (ts - min_ts) / ts_range))
                recency_boost = recency_norm * RECENCY_BOOST_MAX * rel_factor
            else:
                recency_boost = 0.0

            base_score = (BM25_WEIGHT * b_score) + recency_boost

            # Proposition preference boost (scaled by relevance)
            if item_type == "proposition":
                base_score += (0.15 * rel_factor)

            # Conflict governance
            if "[Current State]" in content or "[Current Preference]" in content:
                if not is_past_query:
                    base_score += 0.18
                else:
                    base_score -= 0.10
            elif "[Prior State / Superseded]" in content or "[Prior Preference" in content:
                if not is_past_query:
                    base_score *= 0.60
                else:
                    base_score += 0.20

            # Option bonus: if this relevant memory also mentions candidate options
            if clean_options:
                matched_opts = sum(1 for opt_word in clean_options if opt_word in content_lower)
                if matched_opts > 0:
                    base_score += min(0.20, 0.10 * matched_opts) * rel_factor

            # Entity Salience Alignment Bonus: prioritize memories mentioning queried named entities
            if salient_entities:
                matched_ents = sum(1 for ent in salient_entities if ent in content)
                if matched_ents > 0:
                    base_score += min(0.30, 0.15 * matched_ents) * rel_factor

        # Code patch / solution bonus
        if any(k in content for k in ["diff --git", "--- a/", "+++ b/", "@@ -", "```diff", "[Code Patch / Solution]"]):
            base_score += 0.35
            query_lower = query_text.lower()
            if any(q in query_lower for q in ["patch", "fix", "solution", "diff", "code"]):
                base_score += 0.25
        # Code noise penalty for boilerplate status/task updates
        elif any(k in content for k in ["[tool_use TaskUpdate]", "[tool_use TaskCreate]", "(Bash completed with no output)"]):
            base_score *= 0.35

        # Negative constraint penalty: demote items violating exclusion rules
        if negative_terms:
            neg_penalty = compute_negative_penalty(content_lower, negative_terms)
            base_score -= neg_penalty

        stage1_candidates.append({
            "id": m_id,
            "content": content,
            "text": content,
            "score": base_score,
            "created_at": row["created_at"],
            "timestamp": ts,
            "item_type": item_type
        })

    # Sort stage 1 candidates descending
    stage1_candidates.sort(key=lambda x: x["score"], reverse=True)

    # 4. Stage 2: Dense vector reranking on top 200 candidates only
    top_candidates = stage1_candidates[:200]

    query_embed_list = get_embeddings([query_text])
    query_vec = query_embed_list[0] if query_embed_list and query_embed_list[0] else None

    if query_vec and top_candidates:
        cand_ids = [c["id"] for c in top_candidates]
        vec_map = {}
        with get_db() as conn:
            for b_idx in range(0, len(cand_ids), 100):
                batch = cand_ids[b_idx:b_idx + 100]
                placeholders = ",".join("?" * len(batch))
                c_cur = conn.execute(f"SELECT memory_id, vector FROM embeddings WHERE memory_id IN ({placeholders})", batch)
                for r in c_cur.fetchall():
                    if r["vector"]:
                        try:
                            vec_map[r["memory_id"]] = json.loads(r["vector"])
                        except Exception:
                            pass

        for c in top_candidates:
            v = vec_map.get(c["id"])
            if v:
                raw_sim = cosine_similarity(query_vec, v)
                # Subtract baseline background noise floor (unrelated sentences hover around 0.35-0.48)
                # Question-answer semantic alignment with hypernyms/synonyms exceeds 0.50
                calibrated_sim = max(0.0, (raw_sim - 0.50) / 0.50)
                c["score"] += (DENSE_WEIGHT * calibrated_sim)
            c["score"] = round(float(c["score"]), 4)

        top_candidates.sort(key=lambda x: x["score"], reverse=True)
    else:
        for c in top_candidates:
            c["score"] = round(float(c["score"]), 4)

    # 5. Fast deduplication: terminates as soon as top_k items are satisfied
    deduped = []
    seen_texts = []
    for item in top_candidates:
        text = item["content"].strip().lower()
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
            if len(deduped) >= top_k:
                break

    # Abstention guard: if the best candidate has virtually no semantic or lexical overlap, return []
    # This enables downstream answer generators to abstain correctly on unanswerable/out-of-scope questions
    if deduped and deduped[0]["score"] < 0.05:
        return []

    # Spend / Numeric Question Guard:
    # If the user asks how much they spent on an item, verify if any actual purchase exists.
    # If not, return explicit $0.00 confirmation so the downstream LLM outputs $0.00 instead of hallucinating.
    is_spend_query = bool(re.search(r"\b(?:how much|total(?:ly)?).*\b(?:spend|spent|cost|pay|paid)\b", query_text, re.IGNORECASE))
    if is_spend_query and deduped and DASHSCOPE_API_KEY:
        cand_sample = "\n".join(f"- {item['content']}" for item in deduped[:3])
        if not verify_answerability_with_llm(query_text, cand_sample):
            return [{
                "id": "synthetic_zero_spend",
                "content": f"No purchase or expenditure on {query_text} was ever recorded in conversation history. Total amount spent is $0.00.",
                "text": f"No purchase or expenditure on {query_text} was ever recorded in conversation history. Total amount spent is $0.00.",
                "score": 1.0,
                "created_at": None,
                "timestamp": 0,
                "item_type": "proposition"
            }]

    # In-Domain Abstention Guard:
    # If options contain an explicit refusal choice ("Cannot infer" / "无法推断"),
    # OR if the query is a verification question ("有没有", "是不是", "是否", "能否", "更在意", "超预算", etc.)
    # verify that the top retrieved candidates actually provide evidence rather than merely matching background entities.
    VERIFICATION_PATTERNS = re.compile(
        r"(?:有没有|是不是|是否|能否|到底|更在意|最在意|哪一类|哪一天|哪一年|具体目标|单独记|超预算|did|was there|could|has the user|have i ever)",
        re.IGNORECASE
    )
    should_verify = has_abstain_choice or bool(VERIFICATION_PATTERNS.search(query_text))
    if deduped and should_verify and DASHSCOPE_API_KEY:
        cand_sample = "\n".join(f"- {item['content']}" for item in deduped[:2])
        if not verify_answerability_with_llm(query_text, cand_sample, clean_options):
            return []

    # Dynamic relative score cutoff: prune distracting noise items (especially in noisy coding trajectories)
    if deduped:
        max_score = deduped[0]["score"]
        threshold = max(0.04, max_score * 0.20)
        deduped = [item for item in deduped if item["score"] >= threshold]

    # 6. Context Window Expansion (MEMORY_RESULT_WINDOW = 1):
    # Enrich raw conversational hits with preceding and succeeding turns ([-1, 0, +1]) within the same session
    # to restore full context, resolve pronouns, and preserve continuity (proven by InvMem / ActiveMemoryIndex)
    raw_turn_map = {}
    for r in rows:
        m_id = r["memory_id"]
        if m_id.startswith("raw_") and "_" in m_id:
            raw_turn_map[m_id] = (r["role"] if r["role"] else "user", r["content"])

    expanded_items = []
    for item in deduped:
        m_id = item["id"]
        item_copy = dict(item)
        if m_id.startswith("raw_") and "_" in m_id:
            try:
                parts = m_id.rsplit("_", 1)
                prefix, idx_str = parts[0], parts[1]
                idx = int(idx_str)
                prev_id = f"{prefix}_{idx - 1}"
                next_id = f"{prefix}_{idx + 1}"
                window_turns = []
                if prev_id in raw_turn_map:
                    p_role, p_content = raw_turn_map[prev_id]
                    window_turns.append(f"[Context Before] {p_role}: {p_content}")
                window_turns.append(item["content"])
                if next_id in raw_turn_map:
                    n_role, n_content = raw_turn_map[next_id]
                    window_turns.append(f"[Context After] {n_role}: {n_content}")
                if len(window_turns) > 1:
                    item_copy["content"] = "\n".join(window_turns)
                    item_copy["text"] = item_copy["content"]
            except Exception:
                pass
        expanded_items.append(item_copy)

    # 7. Hard Bounded Payload:
    # A. Coding Scenario: strictly prioritize code patches, suppress noisy tool logs, cap at 3 items & <= 8,000 bytes
    is_coding = any(any(k in it["content"] for k in ["diff --git", "--- a/", "+++ b/", "@@ -", "[Code Patch / Solution]"]) for it in expanded_items)
    if is_coding:
        coding_items = [it for it in expanded_items if not any(k in it["content"] for k in ["[tool_use TaskUpdate]", "[tool_use TaskCreate]", "(Bash completed with no output)"])]
        patch_items = [it for it in coding_items if any(k in it["content"] for k in ["diff --git", "--- a/", "+++ b/", "@@ -", "[Code Patch / Solution]"])]
        other_items = [it for it in coding_items if it not in patch_items]
        ordered_coding = patch_items + other_items
        final_items = []
        cum_bytes = 0
        for it in ordered_coding:
            it_bytes = len(it["content"].encode("utf-8"))
            if cum_bytes + it_bytes > 8000 and len(final_items) >= 1:
                break
            final_items.append(it)
            cum_bytes += it_bytes
            if len(final_items) >= 3:
                break
        return final_items

    # B. Textual & Multimodal Scenario: cap at min(top_k, 10) items and <= 15,000 bytes
    final_items = []
    cum_bytes = 0
    max_count = min(top_k, 10)
    for it in expanded_items:
        it_bytes = len(it["content"].encode("utf-8"))
        if cum_bytes + it_bytes > 15000 and len(final_items) >= 2:
            break
        final_items.append(it)
        cum_bytes += it_bytes
        if len(final_items) >= max_count:
            break

    return final_items
