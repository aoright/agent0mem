import pytest
from app.database import init_db, save_memories_batch
from app.core.clean_search import clean_search


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_streaming_noise_filtering():
    """Verify that streaming QA feedback logs and irrelevant trajectories are filtered from conversational search."""
    uid = "test_user_streaming_noise"

    # User has real conversational memories with Jon and Gina
    turns = [
        {"role": "user", "content": "Gina asked Jon about his new contemporary dance studio opening timeline.", "timestamp": 1680000000},
        {"role": "user", "content": "Jon said it took him four months of renovation before opening the studio.", "timestamp": 1680001000},
    ]

    # Injected streaming feedback and trajectories from previous turns/tasks
    noise_turns = [
        {
            "role": "user",
            "content": (
                'Streaming online QA feedback:\n'
                '{"method": "locomo_llm_judge", "predicted_answer": "The memories do not specify how long it took for Jon to open his studio.", '
                '"qa_id": "locomo1:q0000:step01", "question": "How long did it take for Jon to open his studio?", '
                '"schema_version": "streaming-online-feedback-v1", "score": 0.0}'
            ),
            "timestamp": 1680002000,
        },
        {
            "role": "user",
            "content": (
                'Streaming intra-transfer train trajectory:\n'
                '{"dataset": "ALF", "task_index": 86, "trajectory": [{"role": "system", "content": "Interact with a household..."}, '
                '{"role": "user", "content": "examine cd 2 with desklamp 1"}]}'
            ),
            "timestamp": 1680003000,
        },
    ]

    save_memories_batch(
        request_id="req_stream_1",
        user_id=uid,
        session_id="sess_dialogue",
        raw_messages=turns,
        propositions=[
            "Jon spent four months renovating his contemporary dance studio before opening it. [Conversation Date: 2023-03-28]",
        ]
    )

    save_memories_batch(
        request_id="req_stream_noise",
        user_id=uid,
        session_id="sess_noise",
        raw_messages=noise_turns,
        propositions=[]
    )

    # Search for Jon's studio opening timeline
    res = clean_search(uid, "How long did it take for Jon to open his studio?", top_k=5)
    assert len(res) > 0

    # Rank 1 must be the genuine memory, NOT the failed streaming feedback
    assert not any("Streaming online QA feedback" in r["content"] for r in res), "Streaming QA feedback must be filtered out!"
    assert not any("Streaming intra-transfer train trajectory" in r["content"] for r in res), "Irrelevant ALF trajectory must be filtered out!"
    assert "four months" in res[0]["content"], f"Genuine dialogue memory must be Rank 1, got: {res[0]['content']}"


def test_streaming_trajectory_retrieval():
    """Verify that ALF/DB task trajectory queries correctly retrieve the targeted task trajectory."""
    uid = "test_user_trajectory_task"

    traj_alf_86 = (
        'Streaming intra-transfer train trajectory:\n'
        '{"dataset": "ALF", "task_index": 86, "schema_version": "streaming-intra-transfer-trajectory-v1", '
        '"trajectory": [{"role": "system", "content": "Examine the CD with the desklamp"}, {"role": "assistant", "content": "take cd 2 from desk 1"}]}'
    )
    traj_db_5 = (
        'Streaming intra-transfer train trajectory:\n'
        '{"dataset": "DB", "task_index": 5, "schema_version": "streaming-intra-transfer-trajectory-v1", '
        '"trajectory": [{"role": "system", "content": "Execute SQL query on customers table"}]}'
    )

    save_memories_batch(
        request_id="req_traj_1",
        user_id=uid,
        session_id="sess_traj",
        raw_messages=[
            {"role": "user", "content": traj_alf_86, "timestamp": 1690000000},
            {"role": "user", "content": traj_db_5, "timestamp": 1690001000},
        ],
        propositions=[]
    )

    res = clean_search(uid, "What was the action in ALF task index 86?", top_k=3)
    assert len(res) > 0
    assert "task_index\": 86" in res[0]["content"] or "ALF" in res[0]["content"]
    assert "desklamp" in res[0]["content"]


def test_event_date_chronological_ordering():
    """Verify that event dates occurring in the past sort chronologically even when discussed in later conversations."""
    uid = "test_user_event_date_ordering"

    # Both conversations took place in 2024, but Event A happened in 2021 and Event B happened in 2023
    turns = [
        {"role": "user", "content": "I remember visiting the Louvre in Paris back in March 2021.", "timestamp": 1710000000},
        {"role": "user", "content": "Last summer in August 2023, I climbed Mount Rainier in Washington.", "timestamp": 1710001000},
    ]

    save_memories_batch(
        request_id="req_ev_ord_1",
        user_id=uid,
        session_id="sess_ev",
        raw_messages=turns,
        propositions=[
            "User climbed Mount Rainier in Washington. [Event Date: 2023-08-15] [Conversation Date: 2024-03-10]",
            "User visited the Louvre in Paris. [Event Date: 2021-03-01] [Conversation Date: 2024-03-10]",
        ]
    )

    res = clean_search(uid, "What order did I visit the Louvre and climb Mount Rainier?", top_k=5)
    assert len(res) >= 2
    # In chronological order, Louvre (2021) MUST come before Mount Rainier (2023)
    assert "Louvre" in res[0]["content"], f"Rank 1 must be 2021 Louvre event, got: {res[0]['content']}"
    assert "Rainier" in res[1]["content"], f"Rank 2 must be 2023 Rainier event, got: {res[1]['content']}"


def test_conversational_negative_preference():
    """Verify that natural negative expressions ('I really dislike horror movies') elevate to active rules and suppress distractors."""
    uid = "test_user_conv_neg"

    turns = [
        {"role": "user", "content": "I watched the movie The Conjuring at a Halloween party last year.", "timestamp": 1698000000},
        {"role": "user", "content": "I really dislike horror movies and find them stressful. Please don't recommend any horror films.", "timestamp": 1720000000},
        {"role": "user", "content": "I enjoy lighthearted animated comedies and heartwarming family films.", "timestamp": 1720100000},
    ]

    save_memories_batch(
        request_id="req_neg_conv_1",
        user_id=uid,
        session_id="sess_neg_conv",
        raw_messages=turns,
        propositions=[
            "User watched the movie The Conjuring at a Halloween party. [Conversation Date: 2023-10-23]",
            "User really dislikes horror movies and finds them stressful. Please do not recommend any horror films. [Conversation Date: 2024-07-03]",
            "User enjoys lighthearted animated comedies and heartwarming family films. [Conversation Date: 2024-07-04]",
        ]
    )

    res = clean_search(uid, "Can you recommend a great movie for movie night based on my preferences?", top_k=5)
    assert len(res) > 0
    # Active negative constraint must be Rank 1
    assert "Active User Behavioral Rule" in res[0]["content"]
    assert "dislikes horror" in res[0]["content"] or "horror" in res[0]["content"]
    # Positive horror movie mention (The Conjuring) must be suppressed from top results
    assert not any("The Conjuring" in r["content"] and "Active User" not in r["content"] for r in res[:2])


def test_preference_evolution_present_tense():
    """Verify that present-tense questions ('What is Alice's dietary preference?') boost latest state and demote prior state."""
    uid = "test_user_pref_evol_present"

    turns = [
        {"role": "user", "content": "Alice mentioned that she used to eat steak and barbecue every week.", "timestamp": 1680000000},
        {"role": "user", "content": "Update: In June 2024, Alice switched to a strictly plant-based vegan diet.", "timestamp": 1718000000},
    ]

    save_memories_batch(
        request_id="req_evol_1",
        user_id=uid,
        session_id="sess_evol",
        raw_messages=turns,
        propositions=[
            "[Prior State / Superseded] Alice used to eat steak and barbecue every week. [Conversation Date: 2023-03-28]",
            "[Current State] In June 2024, Alice switched to a strictly plant-based vegan diet. [Conversation Date: 2024-06-10]",
        ]
    )

    res = clean_search(uid, "What is Alice's dietary preference?", top_k=3)
    assert len(res) > 0
    assert "vegan" in res[0]["content"] or "plant-based" in res[0]["content"]
    assert "Prior State" not in res[0]["content"]


def test_attribute_aspect_retrieval():
    """Verify that synonym-based attribute questions ('What is Sarah's occupation?') match employment memories."""
    uid = "test_user_attr_aspect"

    turns = [
        {"role": "user", "content": "Sarah Garcia is employed as a principal aerospace propulsion engineer at Blue Horizon.", "timestamp": 1700000000},
        {"role": "user", "content": "Sarah likes hiking in national parks during vacations.", "timestamp": 1700001000},
    ]

    save_memories_batch(
        request_id="req_attr_1",
        user_id=uid,
        session_id="sess_attr",
        raw_messages=turns,
        propositions=[
            "Sarah Garcia is employed as a principal aerospace propulsion engineer at Blue Horizon. [Conversation Date: 2023-11-15]",
            "Sarah enjoys hiking in national parks during vacations. [Conversation Date: 2023-11-15]",
        ]
    )

    res = clean_search(uid, "What is Sarah's occupation?", top_k=3)
    assert len(res) > 0
    assert any(k in res[0]["content"] for k in ["aerospace", "propulsion engineer", "Blue Horizon"])
