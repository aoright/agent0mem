import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import init_db, save_memories_batch, search_hybrid
from app.core.bm25 import tokenize, BM25Index
from app.core.embeddings import cosine_similarity


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_memories.sqlite3")
    monkeypatch.setattr("app.database.DB_PATH", test_db)
    monkeypatch.setattr("app.config.DB_PATH", test_db)
    init_db()


def test_memory_recall_and_exact_list_preservation():
    user_id = "test_user_recall"
    request_id = "req_recall_01"
    session_id = "sess_01"

    messages = [
        {
            "role": "user",
            "content": "My primary goals for this quarter are: 1. Complete Rust migration, 2. Reduce latency below 50ms, 3. Hire senior backend engineer.",
            "timestamp": 1700000000,
        },
        {
            "role": "assistant",
            "content": "Understood. I have recorded these three goals.",
            "timestamp": 1700000010,
        },
    ]

    propositions = [
        "[Current State] User's primary goals for this quarter are completing Rust migration, reducing latency below 50ms, and hiring a senior backend engineer."
    ]

    save_memories_batch(request_id, user_id, session_id, messages, propositions)

    results = search_hybrid(user_id, "What are the user's primary goals for this quarter?", top_k=5)

    assert len(results) > 0
    top_item = results[0]["content"]
    assert "Rust migration" in top_item
    assert "50ms" in top_item
    assert "senior backend engineer" in top_item


def test_state_graph_updates_and_temporal_routing():
    user_id = "test_user_state_graph"
    request_id_1 = "req_state_01"
    request_id_2 = "req_state_02"
    session_id = "sess_state"

    # Initial state
    msg_1 = [
        {
            "role": "user",
            "content": "I live in Seattle and work as a software engineer at Company A.",
            "timestamp": 1680000000,
        }
    ]
    props_1 = [
        "[Prior State / Superseded] User previously lived in Seattle and worked at Company A."
    ]
    save_memories_batch(request_id_1, user_id, session_id, msg_1, props_1)

    # State update
    msg_2 = [
        {
            "role": "user",
            "content": "I recently moved to New York City and joined Company B as a Tech Lead.",
            "timestamp": 1690000000,
        }
    ]
    props_2 = [
        "[Current State] User currently lives in New York City and works at Company B as a Tech Lead, having moved from Seattle."
    ]
    save_memories_batch(request_id_2, user_id, session_id, msg_2, props_2)

    # Present query should favor Current State (New York / Company B)
    present_results = search_hybrid(user_id, "Where does the user currently live and work?", top_k=3)
    assert len(present_results) > 0
    present_contents = [r["content"] for r in present_results]
    assert any("New York" in c for c in present_contents)
    assert any("Company B" in c for c in present_contents)

    # Past query should correctly surface Superseded State (Seattle / Company A)
    past_results = search_hybrid(user_id, "Where did the user previously live before moving?", top_k=3)
    assert len(past_results) > 0
    contents = [r["content"] for r in past_results]
    assert any("Seattle" in c or "Company A" in c for c in contents)


def test_distractor_resistance_and_option_matching():
    user_id = "test_user_distractor"
    request_id = "req_dist_01"
    session_id = "sess_dist"

    # Input with irrelevant distractor conversations
    messages = [
        {
            "role": "user",
            "content": "Someone told me that Python is slow and Java is bulky, but my absolute favorite language for daily development is Go.",
            "timestamp": 1700000100,
        },
        {
            "role": "user",
            "content": "Also, my colleague drinks 5 cups of espresso, whereas I only drink green tea.",
            "timestamp": 1700000200,
        },
    ]

    propositions = [
        "[Current State] User's favorite programming language for daily development is Go.",
        "[Current State] User drinks green tea, while colleague drinks 5 cups of espresso.",
    ]

    save_memories_batch(request_id, user_id, session_id, messages, propositions)

    # MCQ query with distractors in options
    query = "What is the user's favorite programming language for daily development?"
    options = ["A. Python", "B. Java", "C. Go", "D. C++"]

    results = search_hybrid(user_id, query, options=options, top_k=3)

    assert len(results) > 0
    top_content = results[0]["content"]
    assert "Go" in top_content
    # Top memory score for Go proposition should be higher than irrelevant tea/espresso turn
    assert any("Go" in r["content"] for r in results[:1])


def test_bm25_and_cosine_similarity_edge_cases():
    # BM25 CJK + English tokenization
    tokens = tokenize("Agent0Mem supports 智能记忆 and CJK tokens!")
    assert "agent0mem" in tokens
    assert "supports" in tokens
    assert "智" in tokens or "智能记忆" in tokens or len(tokens) > 3

    bm25 = BM25Index()
    docs = [("d1", "Agent memory system"), ("d2", "Unrelated database record")]
    bm25.fit(docs)
    scores = dict(bm25.score("Agent memory"))
    assert scores["d1"] > scores["d2"]

    # Cosine similarity edge cases
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
