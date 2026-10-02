import pytest
from app.core.temporal import extract_ts_from_content, enrich_temporal_text
from app.database import search_hybrid, make_unmentioned_notice

def test_extract_ts_from_content():
    t1 = "[Time: Sep 04, 2025, 09:16:04] My name is Sarah Garcia..."
    ts1 = extract_ts_from_content(t1)
    assert ts1 is not None
    assert ts1 == 1756977364000

    t2 = "CONVERSATION TRANSCRIPT\n[Turn 1 - 2025-03-01T14:02:00Z]\nMessage from: BRO"
    ts2 = extract_ts_from_content(t2)
    assert ts2 is not None
    assert ts2 == 1740837720000

    t3 = "User discussed a meeting on [Conversation Date: 2024-01-02]."
    ts3 = extract_ts_from_content(t3)
    assert ts3 is not None
    assert ts3 == 1704153600000

    t4 = "General chat without dates or timestamps"
    assert extract_ts_from_content(t4) is None

def test_enrich_temporal_text_no_real_ts():
    raw_text = "Sarah Garcia has a PhD in Political Science."
    enriched = enrich_temporal_text(raw_text, base_sec=1756977364.0, has_real_ts=False)
    assert enriched == raw_text
    assert "[Conversation Date:" not in enriched

def test_memtrap_24_impossible():
    rows = [{
        "memory_id": "m1",
        "user_id": "u_test_memtrap",
        "session_id": "s1",
        "item_type": "raw",
        "role": "assistant",
        "content": "[memtrapbench][session_1][D1:8][text] assistant: I tried, but I can't make 24 from these four numbers.",
        "timestamp": 1704153600000,
        "created_at": "2024-01-02T00:00:00Z"
    }]
    res = search_hybrid("u_test_memtrap", "Reach 24 using the numbers [5, 5, 5, 8].", rows=rows)
    assert len(res) == 1
    assert "I tried, but I can't make 24 from these four numbers." in res[0]["content"]

def test_absent_employer_entity_abstention():
    rows = [{
        "memory_id": "m1",
        "user_id": "u_test_absent",
        "session_id": "s1",
        "item_type": "proposition",
        "role": "system",
        "content": "Sarah Garcia lives in Miami and has a PhD in Political Science.",
        "timestamp": 1704153600000,
        "created_at": "2024-01-02T00:00:00Z"
    }]
    # Query mentions Microsoft which is absent from memory
    res = search_hybrid("u_test_absent", "On Sep 05, 2025, what was Sarah Garcia's job title at Microsoft?", rows=rows)
    assert len(res) == 1
    assert "not mentioned" in res[0]["content"].lower() or "未提及" in res[0]["content"]

def test_temporal_interval_matching():
    rows = [
        {
            "memory_id": "m_1978",
            "user_id": "u_test_interval",
            "session_id": "s1",
            "item_type": "proposition",
            "role": "system",
            "content": "Joaquim Molins is a member of the Centre Union of Catalonia from January 1978 to January 1980.",
            "timestamp": 1704153600000,
            "created_at": "2024-01-02T00:00:00Z"
        },
        {
            "memory_id": "m_1976",
            "user_id": "u_test_interval",
            "session_id": "s1",
            "item_type": "proposition",
            "role": "system",
            "content": "Joaquim Molins is a member of the Catalan Centre from January 1976 to January 1978.",
            "timestamp": 1704153610000,
            "created_at": "2024-01-02T00:00:10Z"
        }
    ]
    res = search_hybrid("u_test_interval", "Which political party did Joaquim Molins belong to in Jan, 1976?", rows=rows)
    assert len(res) >= 1
    assert "Catalan Centre" in res[0]["content"]
    assert "1976" in res[0]["content"]

def test_mquake_multi_hop_expansion():
    rows = [
        {
            "memory_id": "m_hop1",
            "user_id": "u_test_mquake",
            "session_id": "s1",
            "item_type": "proposition",
            "role": "system",
            "content": "Tarsem Singh is a citizen of India",
            "timestamp": 1704153600000,
            "created_at": "2024-01-02T00:00:00Z"
        },
        {
            "memory_id": "m_hop2_up",
            "user_id": "u_test_mquake",
            "session_id": "s2",
            "item_type": "proposition",
            "role": "system",
            "content": "India is now classified as part of Africa according to the updated fact.",
            "timestamp": 1704153700000,
            "created_at": "2024-01-02T00:01:40Z"
        },
        {
            "memory_id": "m_hop2_old",
            "user_id": "u_test_mquake",
            "session_id": "s1",
            "item_type": "proposition",
            "role": "system",
            "content": "[Prior State] India is located on the continent of Asia",
            "timestamp": 1704153600000,
            "created_at": "2024-01-02T00:00:00Z"
        },
        {
            "memory_id": "raw_update",
            "user_id": "u_test_mquake",
            "session_id": "s2",
            "item_type": "raw",
            "role": "user",
            "content": 'Corpus: UPDATE: replace the prior value of this subject and relation with the following fact.\n{"subject": "India", "relation": "continent", "object": "Africa"}',
            "timestamp": 1704153700000,
            "created_at": "2024-01-02T00:01:40Z"
        }
    ]
    res = search_hybrid("u_test_mquake", "What continent is the country located in, where Tarsem Singh is a citizen?", rows=rows)
    assert len(res) >= 2
    # Verify Rank 1 contains the updated fact (Africa) and Rank 2 contains the bridge fact (India)
    contents = [r["content"] for r in res]
    assert any("Africa" in c for c in contents)
    assert any("India" in c for c in contents)
    assert "Africa" in res[0]["content"]
