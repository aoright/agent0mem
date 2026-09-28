# Agent0Mem

Agent0Mem is an open-source, high-precision agent memory system engineered for the Agent Memory Leaderboard (AML) and long-term agent collaboration benchmarks.

It addresses the fundamental limitations of standard RAG and memory buffers: context pollution, lost-in-the-middle degradation, temporal state conflicts, and distractor susceptibility.

## Architectural Highlights

1. **Proposition Distillation with Conflict Governance**
   - Raw dialogue streams are distilled into atomic, declarative propositions.
   - Entity grounding: Resolves ambiguous personal pronouns to explicit entity names.
   - Temporal anchor resolution: Converts relative references ("yesterday", "next month") into deterministic timestamps based on dialogue metadata.
   - Conflict-aware state tracking: Symmetrically tags `[Current State]` and `[Prior State / Superseded]` records so downstream LLMs never confuse historical facts with present reality.

2. **Temporal Intent Routing**
   - Automatically detects whether a retrieval query targets the present/latest state or historical/superseded context.
   - Dynamically boosts current states while downweighting superseded memories for standard queries, and selectively surfaces superseded states when the user explicitly queries past history.

3. **Hybrid Retrieval with Multi-Aspect Fusion**
   - BM25 Indexing: Zero-dependency tokenization supporting multilingual and CJK tokens for exact entity, numerical, and date recall.
   - Dense Embeddings: High-dimensional semantic embeddings for robust conceptual generalization and paraphrase matching.
   - Option-Aware Reinforcement: Reinforces candidate memory evidence matching multiple-choice options without distractor pollution.

4. **Production State Separation & Lightweight Footprint**
   - Zero heavyweight vector database dependencies: Runs on high-throughput SQLite with WAL (Write-Ahead Logging) and connection pooling.
   - Clean separation of mutable database state from application code.

## Protocol Implementation

Agent0Mem implements the standard AML Add/Search REST specification:

### 1. Health Check
- `GET /health`
- Response: `{"status": "ok", "service": "agent0mem", "version": "v1.0"}`

### 2. Add Memory
- `POST /add`
- Request:
```json
{
  "request_id": "req_12345",
  "messages": [
    {
      "role": "user",
      "content": "I usually drink black coffee every morning, but recently switched to matcha latte.",
      "timestamp": 1685000000
    }
  ],
  "user_id": "usr_9988",
  "session_id": "sess_001"
}
```
- Response:
```json
{
  "success": true,
  "request_id": "req_12345",
  "user_id": "usr_9988",
  "session_id": "sess_001"
}
```

### 3. Search Memory
- `POST /search`
- Request:
```json
{
  "query": "What does the user drink in the morning?",
  "options": ["black coffee", "matcha latte", "green tea", "espresso"],
  "user_id": "usr_9988",
  "top_k": 5
}
```
- Response:
```json
{
  "data": [
    {
      "id": "prop_req_12345_0",
      "content": "[Current State] User now drinks matcha latte every morning, replacing black coffee which User previously drank",
      "text": "[Current State] User now drinks matcha latte every morning, replacing black coffee which User previously drank",
      "score": 1.2364,
      "created_at": "2023-05-25T07:33:20Z"
    }
  ]
}
```

## Quick Start

### Installation
```bash
git clone https://github.com/aoright/agent0mem.git
cd agent0mem
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root:
```ini
DASHSCOPE_API_KEY=your_dashscope_api_key_here
AGENT0MEM_DATA_DIR=/var/lib/agent0mem/data
```

### Run Tests
```bash
python tests/test_engine.py
```

### Start Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8288 --workers 2
```

## License
MIT License.
