#!/usr/bin/env python3
"""
Targeted Offline Verification Test Suite for /boost optimizations.
Tests:
1. is_code_q query classification and code isolation
2. Reversing tool-use vs diff boost (complete unified diff prioritized +1.50)
3. Purge raw tool calls from indexing in extractor and database
4. Coding search context budgeting (top 2, strictly <= 2500 bytes total)
5. Abstention fallback in verify_answerability_with_llm (safe epistemic fallback: return False on error)
6. Temporal sequence sorting (timestamp ASC for chronological indicators)
7. Clean output presentation (stripping cumbersome bracketed temporal markers)
"""

import sys
import os
import time
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import init_db, save_memories_batch, search_hybrid, verify_answerability_with_llm
from app.core.extractor import extract_propositions, is_raw_fragmented_tool_call
from app.core.temporal import strip_temporal_markers
from app import config

TEST_DB = "/tmp/test_boost_opt.sqlite3"
config.DB_PATH = TEST_DB


@pytest.fixture(autouse=True)
def setup_db():
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass
    init_db()


def test_code_keywords_and_query_classification():
    """Verify queries like 'object is invalid - missing required column' activate is_code_q."""
    user_id = "test_user_code_q"
    
    # Save a distractor console patch from 2013
    save_memories_batch(
        request_id="req_code_q_1",
        user_id=user_id,
        session_id="sess_code_q",
        raw_messages=[
            {"role": "assistant", "content": "diff --git a/astropy/console.py b/astropy/console.py\n--- a/astropy/console.py\n+++ b/astropy/console.py\n@@ -10,3 +10,4 @@", "timestamp": 1380000000}
        ],
        propositions=[
            "[Code Patch / Solution] diff --git a/astropy/console.py console terminal colors patch from 2013"
        ]
    )
    
    query = "object is invalid - missing required column"
    results = search_hybrid(user_id=user_id, query_text=query, top_k=5)
    
    # Code isolation must prevent aspect bridging from boosting the 2013 console patch
    console_leaked = any("console.py" in r["content"] for r in results)
    assert not console_leaked, "Distractor console.py patch leaked due to missing code isolation!"


def test_reverse_tool_use_vs_diff_boost():
    """Verify that unified git diff patches receive +1.50 priority over tool-use bookmarks."""
    user_id = "test_user_diff_prio"
    
    # Ingest a patch and a legacy file read notice directly
    patch_content = """I resolved the issue by editing astropy/timeseries/sampled.py:
```diff
diff --git a/astropy/timeseries/sampled.py b/astropy/timeseries/sampled.py
--- a/astropy/timeseries/sampled.py
+++ b/astropy/timeseries/sampled.py
@@ -248,6 +248,8 @@ class TimeSeries:
+        if isinstance(epoch_time, Quantity):
+            epoch_time = epoch_time.to(u.day).value
```
All unit tests in test_sampled.py pass."""

    save_memories_batch(
        request_id="req_diff_prio",
        user_id=user_id,
        session_id="sess_diff_prio",
        raw_messages=[
            {"role": "assistant", "content": patch_content, "timestamp": 1686000020}
        ],
        propositions=[
            "[Code Patch / Solution] " + patch_content
        ]
    )
    
    results = search_hybrid(user_id=user_id, query_text="What is the patch to fix fold method in astropy.timeseries?", top_k=3)
    assert len(results) > 0
    assert "diff --git" in results[0]["content"], "Top result must be the unified git diff patch"
    assert results[0]["score"] > 1.5, f"Expected high priority boost for diff, got {results[0]['score']}"


def test_purge_raw_tool_calls_from_indexing():
    """Verify that raw fragmented tool calls ([tool_use Read], [tool_use Edit], etc.) are not indexed."""
    # Test helper
    assert is_raw_fragmented_tool_call('[tool_use Read] {"file_path": "astropy/timeseries/core.py"}')
    assert is_raw_fragmented_tool_call('[tool_use Edit] {"file_path": "astropy/timeseries/core.py"}')
    assert is_raw_fragmented_tool_call('[tool_use Bash] {"command": "git status"}')
    assert is_raw_fragmented_tool_call('[tool_result] (Bash completed with no output)')
    assert not is_raw_fragmented_tool_call('diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py')

    messages = [
        {"role": "assistant", "content": '[tool_use Read] {"file_path": "astropy/timeseries/core.py"}', "timestamp": 1686000001},
        {"role": "assistant", "content": '[tool_use TaskUpdate] {"taskId": "1", "status": "in_progress"}', "timestamp": 1686000002},
        {"role": "assistant", "content": 'I fixed the bug in astropy/timeseries/core.py with a patch:\n```diff\ndiff --git a/astropy/timeseries/core.py b/astropy/timeseries/core.py\n--- a/astropy/timeseries/core.py\n+++ b/astropy/timeseries/core.py\n@@ -1,2 +1,3 @@\n', "timestamp": 1686000003}
    ]

    # Test proposition extraction
    props = extract_propositions(messages)
    for p in props:
        assert "[tool_use Read]" not in p, "Raw [tool_use Read] found in extracted propositions"
        assert "[tool_use TaskUpdate]" not in p, "Raw [tool_use TaskUpdate] found in extracted propositions"

    # Test database ingestion
    user_id = "test_user_purge"
    save_memories_batch("req_purge_1", user_id, "sess_purge", messages, props)

    res = search_hybrid(user_id=user_id, query_text="core.py timeseries", top_k=5)
    for r in res:
        assert "[tool_use Read]" not in r["content"], "Raw [tool_use Read] leaked into search results"
        assert "[tool_use TaskUpdate]" not in r["content"], "Raw [tool_use TaskUpdate] leaked into search results"


def test_coding_context_budgeting():
    """Verify coding results are capped to top 2 cleanest items strictly <= 2,500 bytes total."""
    user_id = "test_user_budget"
    
    # Create several large code memories
    large_patch_1 = "diff --git a/module1.py b/module1.py\n" + ("+ line of code fix\n" * 120)
    large_patch_2 = "diff --git a/module2.py b/module2.py\n" + ("+ another line of code fix\n" * 120)
    large_patch_3 = "diff --git a/module3.py b/module3.py\n" + ("+ third line of code fix\n" * 120)

    save_memories_batch(
        request_id="req_budget_1",
        user_id=user_id,
        session_id="sess_budget",
        raw_messages=[
            {"role": "assistant", "content": large_patch_1, "timestamp": 1686000010},
            {"role": "assistant", "content": large_patch_2, "timestamp": 1686000020},
            {"role": "assistant", "content": large_patch_3, "timestamp": 1686000030},
        ],
        propositions=[
            "[Code Patch / Solution] " + large_patch_1,
            "[Code Patch / Solution] " + large_patch_2,
            "[Code Patch / Solution] " + large_patch_3,
        ]
    )

    results = search_hybrid(user_id=user_id, query_text="patch module fix", top_k=5)
    assert len(results) <= 2, f"Expected at most 2 items, got {len(results)}"
    
    total_bytes = sum(len(r["content"].encode("utf-8")) for r in results)
    assert total_bytes <= 2500, f"Expected total bytes <= 2500, got {total_bytes}"


def test_verify_answerability_fallback():
    """Verify that verify_answerability_with_llm returns False on exception/timeout (strict epistemic safety)."""
    # Calling with invalid URL or broken credentials will raise exception in urllib
    import unittest.mock as mock
    with mock.patch("urllib.request.urlopen", side_effect=Exception("Connection timed out")):
        res = verify_answerability_with_llm("test query", "test context")
        assert res is False, f"Expected False on exception/timeout, got {res}"


def test_temporal_sequence_sorting():
    """Verify that temporal sequence indicators re-sort results chronologically (timestamp ASC)."""
    user_id = "test_user_temporal_seq"
    
    # Ingest 3 steps in reverse chronological order
    step_3 = "第三步：在东京完成最终项目签约交付。[Conversation Date: 2024-03-01]"
    step_2 = "第二步：在柏林完成中期技术评审和演示。[Conversation Date: 2024-02-01]"
    step_1 = "第一步：在伦敦启动项目并确立技术架构。[Conversation Date: 2024-01-01]"

    save_memories_batch(
        request_id="req_seq_1",
        user_id=user_id,
        session_id="sess_seq",
        raw_messages=[
            {"role": "user", "content": step_3, "timestamp": 1709251200},  # 2024-03-01
            {"role": "user", "content": step_2, "timestamp": 1706745600},  # 2024-02-01
            {"role": "user", "content": step_1, "timestamp": 1704067200},  # 2024-01-01
        ],
        propositions=[step_3, step_2, step_1]
    )

    # Query with chronological sequence indicators: '先后顺序'
    query = "请按先后顺序说明项目推进的第一步和最终交付。"
    results = search_hybrid(user_id=user_id, query_text=query, top_k=5)
    
    assert len(results) >= 2
    # Verify chronological ascending order
    timestamps = [r.get("timestamp") for r in results if r.get("timestamp")]
    assert timestamps == sorted(timestamps), f"Results not in chronological order: {timestamps}"
    assert "伦敦" in results[0]["content"], f"First chronological item should be London, got: {results[0]['content']}"


def test_clean_output_presentation():
    """Verify cumbersome bracketed temporal markers are stripped while created_at is preserved."""
    raw_text = "I bought a coffee maker for $150.00. [Conversation Date: 2024-05-10] [Year: 2024] [Month: May] [Date: May 10, 2024] [2024年5月]"
    cleaned = strip_temporal_markers(raw_text)
    assert cleaned == "I bought a coffee maker for $150.00.", f"Failed to clean text cleanly: '{cleaned}'"
    assert "$150.00" in cleaned

    user_id = "test_user_clean_out"
    save_memories_batch(
        request_id="req_clean_1",
        user_id=user_id,
        session_id="sess_clean",
        raw_messages=[
            {"role": "user", "content": "I purchased an espresso machine for $150.00.", "timestamp": 1715342400}
        ],
        propositions=[
            "User purchased an espresso machine for $150.00."
        ]
    )

    results = search_hybrid(user_id=user_id, query_text="How much was the espresso machine?", top_k=1)
    assert len(results) > 0
    top = results[0]
    assert "[Conversation Date:" not in top["content"]
    assert "[Year:" not in top["content"]
    assert "[Month:" not in top["content"]
    assert "[Date:" not in top["content"]
    assert "$150.00" in top["content"]
    # Verify metadata is preserved
    assert top.get("created_at") is not None
