import math
import json
import logging
import time
import urllib.request
import urllib.error
from typing import List, Optional, Dict

from app.config import DASHSCOPE_API_KEY, DASHSCOPE_EMBED_URL

logger = logging.getLogger("agent0mem.embeddings")

import threading

# In-memory thread-safe cache for embeddings to speed up repetitive queries and options
_CACHE: Dict[str, List[float]] = {}
_MAX_CACHE_SIZE = 10000
_CACHE_LOCK = threading.Lock()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


_F16_MAGIC = b"F16:"


def encode_vector(vec: List[float]) -> bytes:
    """Pack an embedding as little-endian float16 (2 bytes per dimension)."""
    import struct
    return _F16_MAGIC + struct.pack(f"<{len(vec)}e", *vec)


def decode_vector(raw) -> Optional[List[float]]:
    """Decode a stored vector: float16 blob (new) or JSON text (legacy rows)."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        b = bytes(raw)
        if b.startswith(_F16_MAGIC):
            import struct
            body = b[len(_F16_MAGIC):]
            return list(struct.unpack(f"<{len(body) // 2}e", body))
        try:
            raw = b.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        v = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return v if isinstance(v, list) else None


def get_embeddings(texts: List[str], timeout: float = 3.0, max_retries: int = 2) -> List[Optional[List[float]]]:
    """Fetch text-embedding-v3 embeddings from DashScope with thread-safe caching and fast failover."""
    if not texts:
        return []
    if not DASHSCOPE_API_KEY:
        logger.warning("No DASHSCOPE_API_KEY provided; dense embeddings disabled.")
        return [None] * len(texts)

    results: List[Optional[List[float]]] = [None] * len(texts)
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    # Thread-safe cache check
    with _CACHE_LOCK:
        for idx, text in enumerate(texts):
            key = text.strip()
            if key in _CACHE:
                results[idx] = _CACHE[key]
            else:
                missing_indices.append(idx)
                missing_texts.append(key[:2048] if key else " ")

    if not missing_texts:
        return results

    from concurrent.futures import ThreadPoolExecutor

    chunks = []
    batch_size = 8  # DashScope text-embedding-v3 hard limit is 10
    for i in range(0, len(missing_texts), batch_size):
        chunk_texts = missing_texts[i:i + batch_size]
        chunk_indices = missing_indices[i:i + batch_size]
        chunks.append((chunk_texts, chunk_indices, i // batch_size))

    def fetch_chunk(chunk_texts, chunk_indices, batch_num):
        payload = json.dumps({
            "model": "text-embedding-v3",
            "input": {
                "texts": chunk_texts
            }
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }

        # Fast retry policy: configurable attempts and timeout to prevent thread starvation
        for attempt in range(max_retries):
            throttled = False
            req = urllib.request.Request(DASHSCOPE_EMBED_URL, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        embeds = data.get("output", {}).get("embeddings", [])
                        embeds_sorted = sorted(embeds, key=lambda x: x.get("text_index", 0))
                        with _CACHE_LOCK:
                            for sub_idx, item in enumerate(embeds_sorted):
                                orig_idx = chunk_indices[sub_idx]
                                emb_vec = item.get("embedding")
                                results[orig_idx] = emb_vec
                                if emb_vec and len(_CACHE) < _MAX_CACHE_SIZE:
                                    _CACHE[chunk_texts[sub_idx]] = emb_vec
                        return True
                    else:
                        logger.warning(f"DashScope embeddings attempt {attempt + 1} HTTP {resp.status}")
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8", errors="ignore")
                throttled = e.code == 429
                logger.warning(f"DashScope embeddings attempt {attempt + 1} HTTP {e.code}: {err_msg}")
            except Exception as e:
                logger.warning(f"DashScope embeddings attempt {attempt + 1} error: {str(e)}")
            time.sleep(0.5 if throttled else 0.15 * (attempt + 1))

        logger.warning(f"Dense embeddings skipped for batch {batch_num} after retries; falling back cleanly to lexical.")
        return False

    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
        futures = [executor.submit(fetch_chunk, ct, ci, bn) for ct, ci, bn in chunks]
        for f in futures:
            f.result()

    return results
