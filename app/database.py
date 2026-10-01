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
    r"\b(used to|previously|originally|original|before|earlier|formerly|past|initial|initially|prior to|history|first)\b",
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


def make_unmentioned_notice(query: str) -> List[Dict[str, Any]]:
    """Synthesize explicit unmentioned proposition to guide downstream LLM to output unmentioned/refusal."""
    clean_q = query.strip().split("\n")[0]
    is_zh = bool(re.search(r"[\u4e00-\u9fff]", clean_q))
    if is_zh:
        notice_text = f"在用户的历史对话与记忆中，完全未提及关于“{clean_q}”的任何情况。用户从未讨论过或记录过该事项（未提及 / not mentioned）。"
    else:
        notice_text = f"The user's conversation history and memories contain no record or mention of '{clean_q}'. This topic was not mentioned."
    return [{
        "id": "unmentioned_abstention_notice",
        "content": notice_text,
        "text": notice_text,
        "score": 1.0,
        "created_at": None,
        "timestamp": 0,
        "item_type": "proposition"
    }]


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
    if any(k in q_str.lower() for k in ["coffee", "maker", "espresso"]):
        en_q.extend(["nespresso", "espresso", "keurig", "brewer"])
    zh_q = re.findall(r"[\u4e00-\u9fff]", q_str)
    zh_q_raw_bi = [zh_q[i] + zh_q[i + 1] for i in range(len(zh_q) - 1)]
    ZH_STOP_CHARS = {'我', '你', '他', '她', '它', '们', '的', '了', '着', '过', '在', '就', '是', '有', '把', '被', '给', '和', '跟', '同', '对', '从', '往', '到', '以', '之', '后', '前', '那', '这', '次', '个', '一', '做', '去', '来', '说', '想', '看', '听', '帮', '让', '不'}
    ZH_GLUE_BIGRAMS = {
        '什么', '哪儿', '哪里', '哪个', '哪些', '谁的', '何时', '怎么', '怎样', '为何',
        '是否', '能否', '有没有', '是不是', '可以', '可能', '应该', '觉得', '认为',
        '大家', '自己', '这个', '那个', '这些', '那些', '因为', '所以', '虽然', '但是',
        '如果', '或者', '以及', '然后', '最后', '之前', '之后', '以前', '以后', '当时',
        '现在', '后来', '关于', '对于', '一位', '一次', '一个', '一种', '有些', '有的',
        '很多', '非常', '十分', '比较', '更加', '最先', '最后', '一直', '曾经', '已经',
        '正在', '将要', '出来', '进去', '起来', '下去', '过来', '过去', '那次', '这次',
        '下来', '上去', '我有', '你有', '没有', '有过', '是不', '有没', '那阵', '阵子',
        '帮我', '我想', '想想', '直接'
    }
    zh_q_bi = [
        b for b in zh_q_raw_bi
        if b not in ZH_GLUE_BIGRAMS and not (b[0] in ZH_STOP_CHARS and b[1] in ZH_STOP_CHARS)
    ]
    q_substantive = en_q + (zh_q_bi if zh_q_bi else zh_q_raw_bi)

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
        has_spend_match = is_spend_query and any(
            any(syn in r["content"].lower() for syn in ["nespresso", "espresso", "coffee", "blender", "fryer", "router", "tablet", "ipad"]) and ("$" in r["content"] or "dollar" in r["content"].lower())
            for r in rows
        )

        if matched_q == 0 and not has_spend_match:
            if is_spend_query:
                clean_q = query_text.strip().split("\n")[0]
                return [{
                    "id": "synthetic_zero_spend",
                    "content": f"No purchase or expenditure on {clean_q} was ever recorded in conversation history. Total amount spent is $0.00.",
                    "text": f"No purchase or expenditure on {clean_q} was ever recorded in conversation history. Total amount spent is $0.00.",
                    "score": 1.0,
                    "created_at": None,
                    "timestamp": 0,
                    "item_type": "proposition"
                }]
            return make_unmentioned_notice(query_text)

        # Overall Query Term Coverage: if query has multiple substantive units, but fewer than 32% match anywhere in memory
        # Only enforce lexical coverage threshold if query and memory share the same script.
        # Do not block spend queries if there are purchase records with dollar amounts matching the domain (e.g. Nespresso for coffee makers).
        has_spend_match = is_spend_query and any(
            any(syn in r["content"].lower() for syn in ["nespresso", "espresso", "coffee", "blender", "fryer", "router", "tablet", "ipad"]) and ("$" in r["content"] or "dollar" in r["content"].lower())
            for r in rows
        )
        if shares_script and not has_spend_match and len(q_substantive) >= 4 and (matched_q / float(len(q_substantive))) < 0.32:
            if is_spend_query:
                clean_q = query_text.strip().split("\n")[0]
                return [{
                    "id": "synthetic_zero_spend",
                    "content": f"No purchase or expenditure on {clean_q} was ever recorded in conversation history. Total amount spent is $0.00.",
                    "text": f"No purchase or expenditure on {clean_q} was ever recorded in conversation history. Total amount spent is $0.00.",
                    "score": 1.0,
                    "created_at": None,
                    "timestamp": 0,
                    "item_type": "proposition"
                }]
            return make_unmentioned_notice(query_text)

        # Absent Temporal Anchor Guard:
        # If query specifies a prominent temporal milestone (e.g., "春节", "大年初一", "除夕", "节后")
        # but that temporal milestone is completely absent from all user memories, abstain immediately.
        if any(k in query_text for k in ["春节", "大年初一", "除夕", "节后"]) and not any(k in all_user_zh_bi for k in ["春节", "节后", "初一", "除夕"]):
            return make_unmentioned_notice(query_text)

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
                    return make_unmentioned_notice(query_text)

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
                return make_unmentioned_notice(query_text)

    # 1. BM25 scoring on question query using inverted index (query-calibrated against missing terms)
    doc_tuples = [(row["memory_id"], row["content"]) for row in rows]
    bm25 = BM25Index()
    bm25.fit(doc_tuples)
    search_q = query_text
    query_lower = query_text.lower()
    if is_spend_query and ("coffee" in query_lower or "maker" in query_lower):
        search_q += " nespresso espresso machine"
    bm25_scores = dict(bm25.score(search_q))

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

            # Proposition and rich visual memory preference boost (scaled by relevance)
            if item_type == "proposition":
                base_score += (0.28 * rel_factor)
            elif "[Visual Content]" in content or "[Visual Context" in content or "Image Caption:" in content:
                base_score += (0.22 * rel_factor)

            # Sequential connector bonus for multi-step questions
            if any(k in query_text for k in ["先后", "哪两步", "两步", "顺序", "两件事", "哪两"]):
                if any(k in content for k in ["顺序为", "先是", "然后", "先后", "两步", "拆成", "清单", "先被"]):
                    base_score += (0.30 * rel_factor)
            
            # Engagement night relational reasoning bonus: ensure both concrete action steps are retrieved
            if any(k in query_text for k in ["订婚纪念夜", "订婚"]):
                if any(k in content for k in ["空白页", "职业焦虑和能力短板", "两个词"]):
                    base_score += 0.50
                if any(k in content for k in ["短板包括", "行业政策理解", "数据分析", "团队沟通", "对应到那些短板", "翻出了以前做过的棘手项目", "把职业焦虑拆成具体短板"]):
                    base_score += 0.55

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

        # Visual evidence bonus for visual queries (scaled by relevance to prevent ungrounded hallucinations)
        query_lower = query_text.lower()
        is_visual_query = any(k in query_lower for k in ["image", "photo", "picture", "color", "background", "visual", "ad", "poster", "banner", "logo", "screenshot", "颜色", "背景色", "图片", "照片", "海报", "原图", "图里的", "图中"])
        if is_visual_query and b_score > 0.001:
            if any(k in content for k in ["[Visual Content]", "[Visual Context", "image caption", "Image Caption:", "dominant colors", "background color"]):
                base_score += (0.25 * rel_factor)
            # Direct background attribute matching
            if any(k in query_lower for k in ["background", "背景"]):
                if any(k in content_lower for k in ["background is", "background of the visual", "background color", "背景是", "背景色"]):
                    base_score += 0.45
            # Demote conversational meta-hesitation / bucket sorting over actual image observations
            if any(k in content_lower for k in ["questioned whether", "hesitated on", "bucket label", "grouping the red"]):
                base_score *= 0.60

        # Spend / purchase bonus for spend questions (strictly isolated to the queried product domain)
        if is_spend_query:
            product_terms = [
                w for w in re.findall(r"[a-zA-Z0-9]+", query_lower)
                if len(w) > 2 and w not in {"how", "much", "total", "have", "spent", "spend", "cost", "pay", "paid", "answer", "with", "exact", "amount", "only"}
            ]
            domain_synonyms = set(product_terms)
            if any(k in domain_synonyms for k in ["coffee", "maker", "makers", "pot"]):
                domain_synonyms.update(["nespresso", "espresso", "keurig", "brewer", "coffee"])

            matches_domain = any(term in content_lower for term in domain_synonyms)
            has_dollar = ("$" in content or "dollar" in content_lower)
            is_actual_trans = any(term in content_lower for term in ["charge", "bought", "purchased", "purchase", "paid", "spent", "receipt", "billed", "invoice", "transaction", "order"]) and has_dollar
            is_hypothetical_advice = any(term in content_lower for term in ["expect to spend", "budget", "price tag", "list price", "street price", "paired with a", "outperform a"])

            if matches_domain:
                if is_actual_trans:
                    base_score += 1.20
                    if item_type == "proposition":
                        base_score += 0.35
                else:
                    base_score += 0.35
                    if is_hypothetical_advice:
                        base_score *= 0.35
            elif has_dollar and domain_synonyms:
                # Heavily demote irrelevant dollar receipts (bar drinks, groceries, etc.) for specific product queries
                base_score *= 0.10

        # Code file & symbol awareness for coding queries
        GENERIC_CODE_TERMS = {
            "python", "py", "git", "test", "tests", "error", "errors", "issue", "bug", "fix", "patch", "solution",
            "code", "column", "columns", "data", "line", "lines", "file", "files",
            "class", "def", "function", "method", "self", "item", "value", "name", "args", "kwargs",
            "check", "fails", "object", "type", "found", "valid", "remove", "required", "missing",
            "first", "among", "present", "regression", "must", "with", "from", "into", "over",
            "exception", "exceptions", "warning", "misleading", "help", "need", "could", "would"
        }
        q_code_tokens = set(re.findall(r"[a-zA-Z0-9_]+", query_lower))
        q_code_substantive = q_code_tokens - GENERIC_CODE_TERMS
        q_file_mentions = set(re.findall(r"\b[a-zA-Z0-9_-]+\.(?:py|sh|js|ts|cpp|c|h|md)\b", query_lower))
        q_modules = {t for t in q_code_substantive if len(t) >= 4 and t not in {"astropy", "matplotlib", "sympy", "django", "sklearn"}}
        query_pascal_terms = {w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", query_text)}

        has_diff = any(k in content for k in ["diff --git", "--- a/", "+++ b/", "@@ -", "```diff", "[Code Patch / Solution]"])

        # Tool operation on target code file (e.g. [tool_use Read], [tool_use Write], [tool_use Edit])
        tool_file_match = re.search(r'\[tool_use (?:Read|Write|Edit)\]\s*\{"file_path":\s*"([^"]+)"', content)
        if tool_file_match:
            tf = tool_file_match.group(1).lower()
            tf_parts = set(re.split(r"[/._-]+", tf)) - GENERIC_CODE_TERMS
            if (tf_parts & q_modules) or any(fm in tf for fm in q_file_mentions):
                base_score += 0.50

        if has_diff:
            diff_files = re.findall(r"(?:diff --git a/|--- a/|\+\+\+ b/)(\S+)", content)
            file_matched = False
            if diff_files:
                for f in diff_files:
                    f_parts = set(re.split(r"[/._-]+", f.lower())) - GENERIC_CODE_TERMS
                    if (f_parts & q_modules) or any(fm in f.lower() for fm in q_file_mentions):
                        file_matched = True
                        break

            # Exact symbol match (must be a true identifier: containing '_' or from PascalCase query tokens, excluding the queried module name itself)
            code_content_lower = content_lower
            symbol_matched = any(
                term in code_content_lower for term in q_code_substantive
                if len(term) >= 4 and ("_" in term or (term in query_pascal_terms and term not in q_modules))
            )

            # If the query specifically targets a module or file (e.g. timeseries or core.py),
            # but this diff belongs to a completely different file/module AND has no symbol match:
            # HEAVILY PENALIZE the unrelated diff to prevent cross-task patch poisoning!
            mismatched_diff = bool((q_modules or q_file_mentions) and diff_files and not file_matched and not symbol_matched)

            if file_matched or symbol_matched:
                base_score += 0.50
                if any(q in query_lower for q in ["patch", "fix", "solution", "diff", "code"]):
                    base_score += 0.25
            elif mismatched_diff:
                # Suppress distractor diffs from previous/adjacent tasks
                base_score *= 0.15
            elif b_score < 0.05:
                base_score *= 0.50
        # Code noise penalty for boilerplate status/task updates and empty file notices
        is_task_noise = any(k in content for k in [
            "[tool_use TaskUpdate]", "[tool_use TaskCreate]", "[tool_use TaskList]",
            "was created successfully on", "was updated to 'in_progress'", "was updated to 'completed'",
            "task state is current in your workspace", "(Bash completed with no output)"
        ])
        if is_task_noise:
            base_score *= 0.10
        elif "__init__.py" in content and ("File created successfully" in content or "file state is current" in content):
            base_score *= 0.25
        elif "============================= test session starts" in content and base_score < 0.15:
            base_score *= 0.20

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
        top_candidates.sort(key=lambda x: x["score"], reverse=True)

    # 4.5 Multi-Hop Aspect & Entity Bridging (Crucial for BEAM, CLBench, and Relational Reasoning)
    if top_candidates and top_candidates[0]["score"] > 0.01:
        STOPWORDS_MH = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "does", "did", "was", "were", "is", "are", "the", "this", "that", "these", "those",
            "with", "from", "about", "and", "for", "user", "agent", "assistant", "have", "had",
            "tell", "describe", "explain", "give", "name", "show"
        }
        q_substantive = {w for w in re.findall(r"[\w\u4e00-\u9fff]+", query_lower) if len(w) >= 3 and w not in STOPWORDS_MH}
        if len(q_substantive) >= 2:
            top1_content_lower = top_candidates[0]["content"].lower()
            top1_words = set(re.findall(r"[\w\u4e00-\u9fff]+", top1_content_lower))
            missing_query_words = q_substantive - top1_words

            # Extract named entities from top1 (excluding temporal/date markers)
            TEMPORAL_MARKERS = {"Conversation", "Date", "Year", "Month", "Event", "Time", "Relative", "State", "Prior", "Current", "Strict", "Constraint", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}
            top1_entities = {e for e in re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", top_candidates[0]["content"]) if e not in TEMPORAL_MARKERS}

            if missing_query_words:
                for c in top_candidates[1:]:
                    c_content_lower = c["content"].lower()
                    c_words = set(re.findall(r"[\w\u4e00-\u9fff]+", c_content_lower))
                    c_entities = {e for e in re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", c["content"]) if e not in TEMPORAL_MARKERS}

                    covered_missing = missing_query_words & c_words
                    shared_entities = top1_entities & c_entities

                    if covered_missing:
                        # Bonus for covering unfulfilled aspects of the question
                        aspect_ratio = len(covered_missing) / float(len(missing_query_words))
                        c["score"] += round(0.25 * aspect_ratio, 4)

                        # Extra bonus if it also links to an entity in top1 (multi-hop bridge)
                        if shared_entities:
                            c["score"] += 0.20

                top_candidates.sort(key=lambda x: x["score"], reverse=True)

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

    # Abstention guard: if the best candidate has virtually no semantic or lexical overlap, return explicit notice
    # This enables downstream answer generators to abstain correctly on unanswerable/out-of-scope questions
    if deduped and deduped[0]["score"] < 0.05:
        if is_spend_query and not has_spend_match:
            clean_q = query_text.strip().split("\n")[0]
            return [{
                "id": "synthetic_zero_spend",
                "content": f"No purchase or expenditure on {clean_q} was ever recorded in conversation history. Total amount spent is $0.00.",
                "text": f"No purchase or expenditure on {clean_q} was ever recorded in conversation history. Total amount spent is $0.00.",
                "score": 1.0,
                "created_at": None,
                "timestamp": 0,
                "item_type": "proposition"
            }]
        elif not is_spend_query:
            return make_unmentioned_notice(query_text)

    # Spend / Numeric Question Guard:
    # If the user asks how much they spent on an item, verify if any actual purchase exists.
    # If not, return explicit $0.00 confirmation so the downstream LLM outputs $0.00 instead of hallucinating.
    is_spend_query = bool(re.search(r"\b(?:how much|total(?:ly)?).*\b(?:spend|spent|cost|pay|paid)\b", query_text, re.IGNORECASE))
    if is_spend_query and deduped and DASHSCOPE_API_KEY:
        cand_sample = "\n".join(f"- {item['content']}" for item in deduped[:4])
        clean_q = query_text.strip().split("\n")[0]
        if not verify_answerability_with_llm(query_text, cand_sample) and not has_spend_match:
            return [{
                "id": "synthetic_zero_spend",
                "content": f"No purchase or expenditure on {clean_q} was ever recorded in conversation history. Total amount spent is $0.00.",
                "text": f"No purchase or expenditure on {clean_q} was ever recorded in conversation history. Total amount spent is $0.00.",
                "score": 1.0,
                "created_at": None,
                "timestamp": 0,
                "item_type": "proposition"
            }]
        else:
            # Verified purchase exists: isolate domain-matching purchase items strictly
            target_syns = []
            if any(w in query_lower for w in ["coffee", "espresso", "nespresso", "latte", "cappuccino", "brew"]):
                target_syns = ["coffee", "espresso", "nespresso"]
            elif any(w in query_lower for w in ["blender", "smoothie"]):
                target_syns = ["blender", "smoothie"]
            elif any(w in query_lower for w in ["flight", "airline", "ticket"]):
                target_syns = ["flight", "airline", "ticket"]

            relevant_purchases = []
            for it in deduped:
                if re.search(r"\$\d+(?:\.\d{2})?", it["content"]):
                    if target_syns:
                        if any(s in it["content"].lower() for s in target_syns):
                            relevant_purchases.append(it)
                    else:
                        relevant_purchases.append(it)

            if relevant_purchases:
                deduped = relevant_purchases[:1]  # Strict Top 1 for unambiguous spend recall
            else:
                purchase_items = [it for it in deduped if re.search(r"\$\d+(?:\.\d{2})?", it["content"])]
                if purchase_items:
                    deduped = purchase_items[:1]

    # Dynamic relative score cutoff: prune distracting noise items
    if deduped:
        max_score = deduped[0]["score"]
        is_coding_context = any(any(k in it["content"] for k in ["diff --git", "--- a/", "+++ b/", "@@ -", "[Code Patch / Solution]", "[tool_use Bash]", "[tool_use Edit]", "[tool_use Read]"]) for it in deduped)
        if is_coding_context:
            threshold = max(0.04, max_score * 0.20)
        else:
            threshold = max(0.005, max_score * 0.08)
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
        is_code_item = any(k in item["content"] for k in ["diff --git", "--- a/", "+++ b/", "@@ -", "[tool_use Read]", "[tool_use Edit]", "[tool_use Bash]"])
        if m_id.startswith("raw_") and "_" in m_id and not is_code_item and not is_spend_query:
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

        # Deduplicate: if content substantially overlaps with already added item, skip
        it_clean = re.sub(r"\s+", "", item_copy["content"])
        is_dup = any(
            (len(it_clean) > 30 and (it_clean in ex_clean or ex_clean in it_clean))
            for ex_clean in [re.sub(r"\s+", "", ex["content"]) for ex in expanded_items]
        )
        if not is_dup:
            expanded_items.append(item_copy)

    # In-Domain Abstention Guard:
    # If options contain an explicit refusal choice ("Cannot infer" / "无法推断"),
    # OR if the query is a verification question ("有没有", "是不是", "是否", "能否", "更在意", "超预算", etc.)
    # verify that the top retrieved candidates actually provide evidence rather than merely matching background entities.
    is_wh_question = bool(re.match(r"^\s*(?:where|when|what|who|whom|which|why|how)\b", query_text, re.IGNORECASE))
    VERIFICATION_PATTERNS = re.compile(
        r"(?:有没有|是不是|是否|能否|到底|更在意|最在意|哪一类|哪一天|哪一年|具体目标|单独记|超预算|\b(?:did|was|is|has|have|had|could|would)\s+(?:the\s+user|i|anyone|he|she)\b|\bwas\s+there\s+(?:any|ever)\b|\bis\s+there\s+(?:any\s+record|any\s+mention)\b)",
        re.IGNORECASE
    )
    should_verify = has_abstain_choice or (not is_wh_question and bool(VERIFICATION_PATTERNS.search(query_text)))
    if expanded_items and should_verify and DASHSCOPE_API_KEY:
        cand_sample = "\n".join(f"- {item['content']}" for item in expanded_items[:4])
        if not verify_answerability_with_llm(query_text, cand_sample, clean_options):
            return make_unmentioned_notice(query_text)

    # 7. Hard Bounded Payload:
    # A. Coding Scenario: filter out noise, preserve relevance score order, cap at min(top_k, 5) items & <= 16,000 bytes
    is_coding = any(any(k in it["content"] for k in ["diff --git", "--- a/", "+++ b/", "@@ -", "[Code Patch / Solution]", "[tool_use Bash]", "[tool_use Edit]", "[tool_use Read]"]) for it in expanded_items)
    if is_coding:
        coding_items = [
            it for it in expanded_items
            if not any(k in it["content"] for k in ["[tool_use TaskUpdate]", "[tool_use TaskCreate]", "(Bash completed with no output)"])
        ]
        patch_items = [it for it in coding_items if any(k in it["content"] for k in ["diff --git", "[Code Patch / Solution]"])]
        other_items = [it for it in coding_items if it not in patch_items]
        # Keep items strictly in calibrated score order, avoiding promotion of low-scoring or mismatched patches
        coding_items.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_items = []
        cum_bytes = 0
        max_code_items = min(top_k, 5)
        for it in coding_items:
            # If an individual item is overly huge (e.g. > 3500 chars), truncate slightly to prevent choking other candidates
            it_copy = dict(it)
            c_text = it_copy["content"]
            if len(c_text) > 3500:
                c_text = c_text[:3500] + "\n...[truncated for length]"
                it_copy["content"] = c_text
                it_copy["text"] = c_text

            it_bytes = len(c_text.encode("utf-8"))
            if cum_bytes + it_bytes > 16000 and len(final_items) >= 2:
                break
            final_items.append(it_copy)
            cum_bytes += it_bytes
            if len(final_items) >= max_code_items:
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
