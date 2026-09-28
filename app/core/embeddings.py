import math
import json
import logging
import time
import urllib.request
import urllib.error
from typing import List, Optional, Dict

from app.config import DASHSCOPE_API_KEY, DASHSCOPE_EMBED_URL

logger = logging.getLogger("agent0mem.embeddings")

# In-memory cache for embeddings to speed up repetitive queries and options
_CACHE: Dict[str, List[float]] = {}
_MAX_CACHE_SIZE = 10000


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


def get_embeddings(texts: List[str]) -> List[Optional[List[float]]]:
    """Fetch text-embedding-v3 embeddings from DashScope with caching and retries."""
    if not texts:
        return []
    if not DASHSCOPE_API_KEY:
        logger.warning("No DASHSCOPE_API_KEY provided; dense embeddings disabled.")
        return [None] * len(texts)

    results: List[Optional[List[float]]] = [None] * len(texts)
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    # Check cache first
    for idx, text in enumerate(texts):
        key = text.strip()
        if key in _CACHE:
            results[idx] = _CACHE[key]
        else:
            missing_indices.append(idx)
            missing_texts.append(key[:2048] if key else " ")

    if not missing_texts:
        return results

    batch_size = 8  # DashScope text-embedding-v3 hard limit is 10
    for i in range(0, len(missing_texts), batch_size):
        chunk_texts = missing_texts[i:i + batch_size]
        chunk_indices = missing_indices[i:i + batch_size]

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

        # Retry with exponential backoff up to 3 times
        success = False
        for attempt in range(3):
            req = urllib.request.Request(DASHSCOPE_EMBED_URL, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        embeds = data.get("output", {}).get("embeddings", [])
                        embeds_sorted = sorted(embeds, key=lambda x: x.get("text_index", 0))
                        for sub_idx, item in enumerate(embeds_sorted):
                            orig_idx = chunk_indices[sub_idx]
                            emb_vec = item.get("embedding")
                            results[orig_idx] = emb_vec
                            # Cache vector
                            if emb_vec and len(_CACHE) < _MAX_CACHE_SIZE:
                                _CACHE[chunk_texts[sub_idx]] = emb_vec
                        success = True
                        break
                    else:
                        logger.warning(f"DashScope embeddings attempt {attempt + 1} HTTP {resp.status}")
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8", errors="ignore")
                logger.warning(f"DashScope embeddings attempt {attempt + 1} HTTP {e.code}: {err_msg}")
            except Exception as e:
                logger.warning(f"DashScope embeddings attempt {attempt + 1} error: {str(e)}")
            time.sleep(0.5 * (attempt + 1))

        if not success:
            logger.error(f"Failed to generate embeddings after retries for batch {i // batch_size}")

    return results
