"""Query-focused evidence notes built at search time.

After retrieval, an LLM reads the retrieved snippets and writes faithful, query-focused
notes (relevant facts only, every list item enumerated once, the original time
expression plus the recording date, and a flag when a fact belongs to someone else).
The notes are returned as the first item; the original snippets follow unchanged, so
the downstream answer model can always fall back to the raw evidence.

Nothing here is dataset- or question-specific. On any failure or timeout the function
returns None and search results are returned without notes.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import List, Optional

logger = logging.getLogger("agent0mem.query_notes")

NOTES_MODEL = os.environ.get("AGENT0MEM_NOTES_MODEL", "qwen-plus")
NOTES_TIMEOUT_SEC = float(os.environ.get("AGENT0MEM_NOTES_TIMEOUT_SEC", "25"))
NOTES_MAX_INPUT_BYTES = int(os.environ.get("AGENT0MEM_NOTES_MAX_INPUT_BYTES", "40000"))

NOTES_PROMPT = """You are the retrieval component of a long-term memory system. Below are snippets from a user's past conversations; each starts with the date it was recorded in brackets. A separate assistant will answer the question using your notes together with the snippets.

Write notes that collect the evidence needed to answer the question:
- Include only facts supported by the snippets. Quote names, numbers, titles and places exactly.
- Keep the time expression used in the conversation (for example "last Friday", "next month") and add the snippet's recording date in parentheses.
- If the question asks for several items, a list, or a count, enumerate every distinct supported item exactly once.
- If a relevant fact is about a different person than the one in the question, state who it is about.
- If facts conflict, list both with their dates.
- If nothing relevant is present, write exactly: No relevant memory found.
- Do not speculate and do not answer beyond the evidence.
Write at most 15 short bullet points.

<snippets>
{snippets}
</snippets>

Question: {question}
Notes:"""


def build_notes(question: str, items: List[dict]) -> Optional[str]:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key or not items:
        return None
    base = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    parts, size = [], 0
    for it in items:
        c = str(it.get("content", ""))
        b = len(c.encode("utf-8"))
        if size + b > NOTES_MAX_INPUT_BYTES:
            break
        parts.append(c)
        size += b
    prompt = NOTES_PROMPT.format(snippets="\n\n".join(parts), question=question)
    body = json.dumps({"model": NOTES_MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": 600}).encode()
    for attempt in range(2):
        try:
            req = urllib.request.Request(base + "/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {key}",
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=NOTES_TIMEOUT_SEC) as r:
                text = json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
            return text or None
        except Exception as e:  # noqa: BLE001
            logger.warning("query notes attempt %d failed: %s", attempt + 1, e)
    return None


def with_notes(question: str, items: List[dict]) -> List[dict]:
    notes = build_notes(question, items)
    if not notes:
        return items
    content = f"[Evidence notes for this question, compiled from the memories below]\n{notes}"
    return [{"id": "query_notes", "content": content, "text": content, "score": 1.0,
             "created_at": None, "timestamp": None}] + items
