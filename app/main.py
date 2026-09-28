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
from app.database import init_db, save_memories_batch, search_hybrid
from app.core.extractor import extract_propositions

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


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", service="agent0mem", version="v1.0")


@app.post("/add", response_model=AddResponse)
async def add_memory(req: AddRequest):
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
async def search_memory(req: SearchRequest):
    try:
        # Extract query text if list of parts or string
        if isinstance(req.query, list):
            text_parts = [
                p.get("text", "") for p in req.query
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            query_str = " ".join(text_parts) if text_parts else str(req.query)
        else:
            query_str = str(req.query)

        # Execute hybrid search with option expansion and recency boost
        candidates = search_hybrid(
            user_id=req.user_id,
            query_text=query_str,
            options=req.options,
            top_k=req.top_k
        )

        items = [
            SearchMemoryItem(
                id=c["id"],
                content=c["content"],
                text=c.get("text", c["content"]),
                score=c.get("score", 1.0),
                created_at=c.get("created_at"),
            )
            for c in candidates
        ]
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
