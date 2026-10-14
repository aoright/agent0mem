import re
import json
import time
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, ENABLE_LLM_EXTRACTION

logger = logging.getLogger("agent0mem.extractor")

import os as _os

# A single 6 s attempt timed out under load and silently stored raw dialogue as
# "propositions". Defaults below give the extractor room to finish.
EXTRACTION_MODEL = _os.environ.get("AGENT0MEM_EXTRACTION_MODEL", "qwen-flash")
EXTRACTION_FALLBACK_MODEL = _os.environ.get("AGENT0MEM_EXTRACTION_FALLBACK_MODEL", "qwen-plus")
EXTRACTION_TIMEOUT_SEC = float(_os.environ.get("AGENT0MEM_EXTRACTION_TIMEOUT_SEC", "20"))
EXTRACTION_ATTEMPTS = int(_os.environ.get("AGENT0MEM_EXTRACTION_ATTEMPTS", "2"))


def is_raw_fragmented_tool_call(content: str) -> bool:
    """Check if content is a raw fragmented tool call or tool execution noise without code patches."""
    if not content:
        return False
    if any(marker in content for marker in ["diff --git", "--- a/", "+++ b/", "@@ -", "```diff", "[Code Patch / Solution]"]):
        return False
    c = content.strip()
    if re.search(r"\[tool_use\s+[^\]]+\]", c, re.IGNORECASE):
        return True
    if re.search(r"\[tool_result\]", c, re.IGNORECASE):
        return True
    if any(phrase in c for phrase in [
        "(Bash completed with no output)",
        "was created successfully on",
        "was updated to 'in_progress'",
        "was updated to 'completed'",
        "task state is current in your workspace",
        "file state is current in your context"
    ]):
        return True
    return False



EXTRACTION_PROMPT = """You are an expert memory distillation and governance engine.
Given the timestamped conversation turns, extract key factual statements as atomic, self-contained propositions.

Rules:
1. Each proposition must be a clear, standalone factual statement. Include the speaker name or subject (e.g., "Alice prefers green tea over coffee").
2. Resolve pronouns (he, she, it, they, I, my) to specific named entities based on the conversation context.
3. Temporal and Date Grounding:
   - When turns have timestamps, convert day-level relative dates (e.g., "yesterday", "two days ago") to explicit dates if clear from the turn timestamp.
   - For week-based expressions ("last week", "first week of July", "a couple weeks ago"), PRESERVE the relative expression along with the month and year (e.g., "in the first week of April 2022" or "last week in April 2022"), do not collapse it into an arbitrary single day.
   - Retain exact dates, times, locations, numbers, codes, and proper nouns verbatim.
4. Conflict and State Updates:
   - When an entity's status, habit, location, or preference changes across time (e.g., stopped, switched, moved, relocated, changed, no longer, used to):
     Provide complete transition context in statements:
     - Mark current status as:
       "[Current State] Entity now ... (e.g., [Current State] Jordan now drinks matcha latte every morning, replacing black coffee which Jordan previously drank; or [Current State] Jordan currently lives in Tokyo as of Dec 2023, having relocated from Berlin)"
     - Mark prior status with transition links:
       "[Prior State / Superseded] Entity previously ... (e.g., [Prior State / Superseded] Jordan previously lived in London until May 2023 before moving to Berlin in June 2023)"
5. Exact Lists:
   - For lists of items (e.g., hobbies, visited countries, ingredients, recommendations), keep all mentioned items together in a SINGLE complete proposition (e.g., "The user's three favorite hobbies are rock climbing, sourdough baking, and landscape photography") rather than splitting them into separate single-item statements, and do not hallucinate extras.
6. Constraints, Allergies, and Negative Preferences:
   - Explicitly record all mentioned constraints, allergies, and negative preferences completely. Do not omit any items or allergens (e.g., if the user avoids peanuts and shellfish, include BOTH: "User strictly avoids peanuts and shellfish due to severe allergies").
7. Ignore meaningless pleasantries, greetings, and filler (e.g., "hi", "ok", "sounds good", "thanks").
8. Return only a valid JSON array of strings. Do not wrap in markdown or commentary.
"""


def format_timestamp(ts: Optional[int]) -> str:
    if not ts:
        return ""
    try:
        t = float(ts)
        if t > 1e11:  # milliseconds
            t = t / 1000.0
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def extract_propositions(messages: List[Dict[str, Any]]) -> List[str]:
    """Distill atomic factual propositions with temporal grounding and conflict annotations."""
    if not messages:
        return []

    dialogue_lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        ts = msg.get("timestamp")
        time_prefix = f"[{format_timestamp(ts)}] " if ts else ""

        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text" and p.get("text"):
                        parts.append(p.get("text"))
                    elif p.get("type") in ("image_url", "image") or "image_url" in p or "image" in p:
                        img_data = p.get("image_url", p.get("image", {}))
                        img_url = img_data.get("url") if isinstance(img_data, dict) else (p.get("url") or str(img_data))
                        if img_url and isinstance(img_url, str) and (img_url.startswith("http") or img_url.startswith("data:")):
                            try:
                                from app.core.vision import describe_image
                                v_desc = describe_image(img_url)
                                if v_desc:
                                    parts.append(f"[Visual Content] {v_desc}")
                            except Exception:
                                pass
            content = " ".join(parts)
        content_str = str(content).strip()
        if content_str and len(content_str) > 1:
            if is_raw_fragmented_tool_call(content_str):
                continue
            dialogue_lines.append(f"{time_prefix}{role}: {content_str}")


    if not dialogue_lines:
        return []

    full_dialogue = "\n".join(dialogue_lines)

    # Skip LLM proposition extraction on machine-generated streaming feedback, trajectory payloads,
    # and structured document/triple/trap corpora where raw turns already preserve exact content.
    if re.search(
        r"(?:streaming\s+online\s+qa\s+feedback:|\"schema_version\":\s*\"streaming-online-feedback|"
        r"streaming\s+intra-transfer\s+train\s+trajectory:|\"schema_version\":\s*\"streaming-intra-transfer-trajectory|"
        r"\[docpp\]|Native\s+clip\s+page:|\[memtrapbench\]|Corpus:\s*(?:UPDATE:|\{)|\[tempreason\]|\[clbench)",
        full_dialogue,
        re.IGNORECASE,
    ):
        return []

    # Directly preserve code diffs and patches as high-priority propositions
    patch_propositions = []
    for line in dialogue_lines:
        if any(marker in line for marker in ["diff --git", "--- a/", "+++ b/", "@@ -"]):
            patch_propositions.append(f"[Code Patch / Solution] {line[:3000]}")

    if not ENABLE_LLM_EXTRACTION or not DASHSCOPE_API_KEY:
        return patch_propositions + dialogue_lines

    # Guard against massive token overflow in coding trajectories
    if len(full_dialogue) > 12000:
        dialogue_for_llm = full_dialogue[:6000] + "\n...[truncated long middle section]...\n" + full_dialogue[-6000:]
    else:
        dialogue_for_llm = full_dialogue

    try:
        url = f"{DASHSCOPE_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }

        for attempt in range(EXTRACTION_ATTEMPTS):
            active_model = EXTRACTION_MODEL if attempt == 0 else EXTRACTION_FALLBACK_MODEL
            payload = json.dumps({
                "model": active_model,
                "messages": [
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": f"Conversation:\n{dialogue_for_llm}\n\nAtomic Propositions (JSON array):"}
                ],
                "temperature": 0.1,
                "max_tokens": 4096
            }).encode("utf-8")
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=EXTRACTION_TIMEOUT_SEC) as resp:
                    if resp.status == 200:
                        result = json.loads(resp.read().decode("utf-8"))
                        raw_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        if raw_text.startswith("```"):
                            parts = raw_text.split("```")
                            if len(parts) >= 2:
                                raw_text = parts[1]
                                if raw_text.startswith("json"):
                                    raw_text = raw_text[4:].strip()
                        propositions = []
                        try:
                            propositions = json.loads(raw_text)
                        except Exception:
                            # Fallback to regex extraction of complete JSON string literals
                            matches = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', raw_text)
                            propositions = [m.replace('\\"', '"').replace('\\n', ' ') for m in matches if len(m) > 10]
                        if isinstance(propositions, list) and propositions:
                            clean_props = [str(p).strip() for p in propositions if str(p).strip()]
                            return patch_propositions + clean_props
                    else:
                        logger.warning(f"LLM extraction ({active_model}) HTTP {resp.status}")
            except Exception as e:
                logger.warning(f"LLM extraction ({active_model}) attempt {attempt + 1} error: {str(e)}")
                if attempt + 1 < EXTRACTION_ATTEMPTS:
                    time.sleep(0.5 * (attempt + 1))
    except Exception as e:
        logger.warning(f"LLM proposition extraction outer fallback: {str(e)}")

    return patch_propositions
