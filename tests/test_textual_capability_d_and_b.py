import pytest
from app.database import init_db, save_memories_batch, search_hybrid


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_natural_language_knowledge_update():
    """Verify that natural language state updates demote prior facts and boost current state."""
    uid = "test_user_knowledge_update_nl"

    turns = [
        {"role": "user", "content": "I work at DeepMind London as an AI research engineer.", "timestamp": 1665792000},
        {"role": "user", "content": "Update: On July 1, 2024, I transitioned to lead the Long-Term Memory team at DeepMind.", "timestamp": 1719792000},
        {"role": "user", "content": "Falcon cluster initially had 128 H100 GPUs, upgraded to 512 H100 GPUs on September 20, 2024.", "timestamp": 1726790400},
        {"role": "user", "content": "On January 10, 2025, the Falcon cluster was expanded further to 1024 H100 GPUs.", "timestamp": 1736467200},
    ]

    save_memories_batch(
        request_id="req_up_1",
        user_id=uid,
        session_id="sess_up_1",
        raw_messages=turns,
        propositions=[
            "User works at DeepMind London as an AI research engineer. [Conversation Date: 2022-10-15]",
            "[Current State] User transitioned to lead the Long-Term Memory team at DeepMind. [Conversation Date: 2024-07-01]",
            "[Prior State / Superseded] Falcon cluster initially had 128 H100 GPUs, upgraded to 512 H100 GPUs. [Conversation Date: 2024-09-20]",
            "[Current State] Falcon cluster was expanded further to 1024 H100 GPUs. [Conversation Date: 2025-01-10]",
        ]
    )

    # 1. Current State Query: Falcon cluster GPUs
    res_current = search_hybrid(uid, "How many H100 GPUs does the Falcon cluster currently have?", top_k=3)
    assert len(res_current) > 0
    assert "1024" in res_current[0]["content"], f"Current state should be ranked #1 with 1024 GPUs, got: {res_current[0]['content']}"

    # 2. Past State Query: Falcon cluster initial GPUs
    res_past = search_hybrid(uid, "How many H100 GPUs did the Falcon cluster initially have?", top_k=3)
    assert len(res_past) > 0
    assert "128" in res_past[0]["content"] or "512" in res_past[0]["content"], f"Past state should be retrieved for initial GPUs, got: {res_past[0]['content']}"

    # 3. Current Role Query:
    res_role = search_hybrid(uid, "What is the user's current role at DeepMind?", top_k=3)
    assert len(res_role) > 0
    assert "lead the Long-Term Memory team" in res_role[0]["content"], f"Current role should be lead Long-Term Memory, got: {res_role[0]['content']}"


def test_persona_negative_constraint_governance():
    """Verify that strict allergies and dietary constraints are preserved and boosted for recommendation queries."""
    uid = "test_user_persona_neg_gov"

    turns = [
        {"role": "user", "content": "I visited a seafood restaurant in Boston last summer.", "timestamp": 1690000000},
        {"role": "user", "content": "Health update: I strictly follow a vegan diet and have severe allergy to shellfish. Never recommend seafood or meat.", "timestamp": 1720000000},
        {"role": "user", "content": "I enjoy listening to jazz music while eating dinner.", "timestamp": 1720100000},
    ]

    save_memories_batch(
        request_id="req_neg_1",
        user_id=uid,
        session_id="sess_neg_1",
        raw_messages=turns,
        propositions=[
            "User visited a seafood restaurant in Boston last summer. [Conversation Date: 2023-07-22]",
            "[Strict Constraint] User strictly follows a vegan diet and has severe allergy to shellfish. Never recommend seafood or meat. [Conversation Date: 2024-07-03]",
            "User enjoys listening to jazz music while eating dinner. [Conversation Date: 2024-07-04]",
        ]
    )

    res = search_hybrid(uid, "Can you recommend a place for dinner? Please recall my related preferences from our conversation history to give personalized responses.", top_k=3)
    assert len(res) > 0
    # Strict constraint must be at Rank 1
    assert any(k in res[0]["content"] for k in ["vegan", "shellfish", "Strict Constraint", "Never recommend"]), f"Rank 1 must be the constraint, got: {res[0]['content']}"
