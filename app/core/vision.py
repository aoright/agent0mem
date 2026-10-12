import hashlib
import json
import logging
import threading
import time
import urllib.request
import urllib.error
from typing import Optional, Dict

from app.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

logger = logging.getLogger("agent0mem.vision")

# In-memory thread-safe cache for image descriptions
_IMAGE_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 4096
_IMAGE_CACHE_LOCK = threading.Lock()

VISION_PROMPT = (
    "Analyze this image comprehensively for multimodal memory retrieval in both English and Chinese. "
    "1. Explicitly identify and state the background color and dominant colors in both languages "
    "(e.g., 'Background color: white / 背景颜色: 白色'). "
    "2. Transcribe verbatim all visible text, numbers, prices, brand labels, and timestamps. "
    "3. Concisely describe key people, animals, objects, attire, and actions in 2-3 factual sentences."
)


def _compute_image_key(image_data: str) -> str:
    """Compute a short MD5 hash for image caching."""
    if len(image_data) > 256:
        # Hash prefix + length + suffix to avoid full hash on huge base64
        h = hashlib.md5(f"{image_data[:128]}_{len(image_data)}_{image_data[-128:]}".encode("utf-8")).hexdigest()
    else:
        h = hashlib.md5(image_data.encode("utf-8")).hexdigest()
    return h


def describe_image(image_url: str, timeout: float = 8.0) -> Optional[str]:
    """
    Extract a detailed factual description from an image using Qwen-VL.
    Works with inline data URIs (data:image/...;base64,...) and public URLs.
    Includes thread-safe caching and retry resilience against network jitter.
    """
    if not image_url or not DASHSCOPE_API_KEY:
        return None

    # Check cache with thread lock
    cache_key = _compute_image_key(image_url)
    with _IMAGE_CACHE_LOCK:
        if cache_key in _IMAGE_CACHE:
            return _IMAGE_CACHE[cache_key]

    url = f"{DASHSCOPE_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-vl-plus",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.1
    }

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    description = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if description:
                        with _IMAGE_CACHE_LOCK:
                            if len(_IMAGE_CACHE) >= _MAX_CACHE_SIZE:
                                _IMAGE_CACHE.pop(next(iter(_IMAGE_CACHE)))
                            _IMAGE_CACHE[cache_key] = description
                        return description
                else:
                    logger.warning(f"Qwen-VL returned HTTP {resp.status} on attempt {attempt + 1}")
        except Exception as e:
            logger.warning(f"Qwen-VL image description attempt {attempt + 1} failed: {str(e)}")
        time.sleep(0.5 * (attempt + 1))

    return None
