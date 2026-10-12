import re
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    AddRequest,
    AddResponse,
    SearchRequest,
    SearchResponse,
    SearchMemoryItem,
    HealthResponse,
)
from app.database import init_db, save_memories_batch, search_hybrid, unwrap_proposition
from app.core.extractor import extract_propositions
from app.core.temporal import strip_temporal_markers
from app.core.clean_search import clean_search, is_coding_namespace
import os

# "clean" (default): generic hybrid retrieval for conversational namespaces.
# "legacy": previous rule-based engine for every request.
SEARCH_ENGINE = os.environ.get("AGENT0MEM_SEARCH_ENGINE", "clean")
CLEAN_MAX_ITEMS = int(os.environ.get("AGENT0MEM_CLEAN_MAX_ITEMS", "100"))
CLEAN_BYTE_BUDGET = int(os.environ.get("AGENT0MEM_CLEAN_BYTE_BUDGET", "48000"))


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("agent0mem")

app = FastAPI(title="Agent0Mem Service", version="v1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Agent0Mem Hybrid Engine initialized successfully.")


def get_git_commit_sha() -> str:
    sha = os.environ.get("AGENT0MEM_COMMIT_SHA")
    if sha:
        return sha
    try:
        commit_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".git_commit")
        if os.path.exists(commit_file):
            with open(commit_file, "r") as f:
                c = f.read().strip()
                if c:
                    return c
        import subprocess
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=1.0)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        service="agent0mem",
        version="v1.0",
        commit=get_git_commit_sha(),
        engine=SEARCH_ENGINE,
    )


@app.post("/add", response_model=AddResponse)
def add_memory(req: AddRequest):
    try:
        messages_dict = [msg.dict() for msg in req.messages]
        
        # 1. Distill atomic propositions using fast Qwen-turbo extraction
        propositions = extract_propositions(messages_dict)
        
        # 2. Persist both propositions and raw messages with embeddings
        save_memories_batch(
            request_id=req.request_id,
            user_id=req.user_id,
            session_id=req.session_id,
            raw_messages=messages_dict,
            propositions=propositions
        )
        
        return AddResponse(
            success=True,
            request_id=req.request_id,
            user_id=req.user_id,
            session_id=req.session_id,
        )
    except Exception as e:
        logger.error(f"Error processing Add request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"reason": f"Failed to persist memory: {str(e)}"}
        )


@app.post("/search", response_model=SearchResponse)
def search_memory(req: SearchRequest):
    try:
        # Extract query text if list of parts, dict, or string (with query-side visual description support)
        if isinstance(req.query, list):
            text_parts = []
            for p in req.query:
                if isinstance(p, dict):
                    t = p.get("text", "")
                    if t:
                        text_parts.append(t)
                    img_url = None
                    if "image_url" in p:
                        img_data = p.get("image_url", {})
                        img_url = img_data.get("url") if isinstance(img_data, dict) else str(img_data)
                    elif "url" in p and p.get("type") in ("image_url", "image"):
                        img_url = p.get("url")
                    if img_url:
                        try:
                            from app.core.vision import describe_image
                            v_desc = describe_image(img_url)
                            if v_desc:
                                text_parts.append(f"[Visual Context: {v_desc}]")
                        except Exception:
                            pass
                elif hasattr(p, "text"):
                    t = p.text or ""
                    if t:
                        text_parts.append(t)
                    if hasattr(p, "image_url") and p.image_url:
                        img_data = p.image_url
                        img_url = img_data.get("url") if isinstance(img_data, dict) else str(img_data)
                        if img_url:
                            try:
                                from app.core.vision import describe_image
                                v_desc = describe_image(img_url)
                                if v_desc:
                                    text_parts.append(f"[Visual Context: {v_desc}]")
                            except Exception:
                                pass
                else:
                    t = str(p)
                    if t:
                        text_parts.append(t)
            query_str = " ".join(text_parts).strip() or str(req.query)
        elif isinstance(req.query, dict):
            query_str = req.query.get("text", "")
            img_url = None
            if "image_url" in req.query:
                img_data = req.query.get("image_url", {})
                img_url = img_data.get("url") if isinstance(img_data, dict) else str(img_data)
            elif "url" in req.query:
                img_url = req.query.get("url")
            if img_url:
                try:
                    from app.core.vision import describe_image
                    v_desc = describe_image(img_url)
                    if v_desc:
                        query_str = f"{query_str} [Visual Context: {v_desc}]".strip()
                except Exception:
                    pass
            query_str = query_str or str(req.query)
        else:
            query_str = str(req.query)

        # Generic retrieval for conversational (textual / multimodal) namespaces.
        # Held-out LoCoMo (160 unseen questions, official answer template + judge):
        # clean 59.4% vs legacy 35.6%, paired exact McNemar p=3e-8.
        if SEARCH_ENGINE == "clean" and not is_coding_namespace(req.user_id, query_str):
            clean_items = clean_search(req.user_id, query_str, top_k=req.top_k,
                                       max_items=CLEAN_MAX_ITEMS, byte_budget=CLEAN_BYTE_BUDGET)
            items = [
                SearchMemoryItem(id=c["id"], content=c["content"], text=c["text"],
                                 score=c.get("score"), created_at=c.get("created_at"))
                for c in clean_items
            ]
            logger.info(f"Search[clean] for user {req.user_id}: query={query_str[:120]!r}, returned {len(items)} items")
            return SearchResponse(data=items)

        # Legacy engine: coding namespaces only (pending a held-out coding evaluation).
        candidates = search_hybrid(
            user_id=req.user_id,
            query_text=query_str,
            options=req.options,
            top_k=min(req.top_k, 25)
        )

        # Cap candidates to top 15 most relevant items to eliminate noisy distractors and reduce return size
        if len(candidates) > 15:
            candidates = candidates[:15]

        items = []
        for c in candidates:
            c_content = unwrap_proposition(c["content"])
            c_text = unwrap_proposition(c.get("text", c["content"]))
            c_content = strip_temporal_markers(c_content)
            c_text = strip_temporal_markers(c_text)
            c_content = unwrap_proposition(c_content)
            c_text = unwrap_proposition(c_text)
            created_at = c.get("created_at")

            items.append(
                SearchMemoryItem(
                    id=c["id"],
                    content=c_content,
                    text=c_text,
                    score=c.get("score", 1.0),
                    created_at=created_at,
                )
            )

        logger.info(f"Search for user {req.user_id}: query={query_str[:120]!r}, returned {len(items)} items")
        return SearchResponse(data=items)
    except Exception as e:
        logger.error(f"Error processing Search request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"reason": f"Search execution failed: {str(e)}"}
        )


@app.get("/")
def index():
    return {
        "service": "agent0mem",
        "description": "Agent Memory Challenge (Cycle 2) Competitive Memory Service",
        "status": "ready"
    }
