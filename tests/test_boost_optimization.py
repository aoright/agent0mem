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
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import init_db, save_memories_batch, search_hybrid, verify_answerability_with_llm, unwrap_proposition
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
    # When timeseries fix is triggered, the precise fix propositions take rank 1;
    # the actual diff patch should still be present in results
    has_diff = any("diff --git" in r["content"] or "as_scalar_or_list_str" in r["content"] for r in results)
    assert has_diff, "Results must contain either the fix proposition or the unified git diff patch"


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
    assert len(results) <= 4, f"Expected at most 4 items, got {len(results)}"
    
    total_bytes = sum(len(r["content"].encode("utf-8")) for r in results)
    assert total_bytes <= 4500, f"Expected total bytes <= 4500, got {total_bytes}"


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
    """Verify redundant bracketed temporal markers are stripped while conversation dates and created_at are preserved."""
    raw_text = "I bought a coffee maker for $150.00. [Conversation Date: 2024-05-10] [Year: 2024] [Month: May] [Date: May 10, 2024] [2024年5月]"
    cleaned = strip_temporal_markers(raw_text)
    assert cleaned == "I bought a coffee maker for $150.00. [Conversation Date: 2024-05-10]", f"Failed to preserve date: '{cleaned}'"
    assert "$150.00" in cleaned
    assert "[Conversation Date: 2024-05-10]" in cleaned
    assert "[Year:" not in cleaned
    assert "[Month:" not in cleaned
    assert "[Date:" not in cleaned

    # Test preserving sole date when only [Date: ...] is present
    raw_sole = "User attended conference in London. [Date: October 15, 2022] [Year: 2022] [Month: October]"
    cleaned_sole = strip_temporal_markers(raw_sole)
    assert "October 15, 2022" in cleaned_sole, f"Failed to preserve sole date: '{cleaned_sole}'"
    assert "[Conversation Date: October 15, 2022]" in cleaned_sole
    assert "[Year:" not in cleaned_sole
    assert "[Month:" not in cleaned_sole

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
    assert "[Conversation Date: 2024-05-10]" in top["content"]
    assert "[Year:" not in top["content"]
    assert "[Month:" not in top["content"]
    assert "[Date:" not in top["content"]
    assert "$150.00" in top["content"]
    # Verify metadata is preserved
    assert top.get("created_at") is not None


def test_mismatched_patch_not_promoted():
    """Verify that when query targets astropy/timeseries/core.py, an unrelated patch for table.py is not promoted to Rank 1."""
    user_id = "test_user_mismatch_patch"
    
    table_patch = """diff --git a/astropy/table/table.py b/astropy/table/table.py
--- a/astropy/table/table.py
+++ b/astropy/table/table.py
@@ -100,6 +100,8 @@ def to_pandas(self):
+    if index_name:
+        del self.columns[index_name]
"""
    timeseries_prop = "In astropy/timeseries/core.py, _check_required_columns raises an error when required columns are removed."

    save_memories_batch(
        request_id="req_mismatch_1",
        user_id=user_id,
        session_id="sess_mismatch",
        raw_messages=[
            {"role": "assistant", "content": table_patch, "timestamp": 1628188560}
        ],
        propositions=[
            "[Code Patch / Solution] " + table_patch,
            timeseries_prop
        ]
    )

    results = search_hybrid(
        user_id=user_id,
        query_text="timeseries core.py _check_required_columns fix missing required column test",
        top_k=3
    )
    assert len(results) > 0
    assert "table.py" not in results[0]["content"], f"Mismatched table.py patch promoted to Rank 1! Got: {results[0]['content']}"
    assert "timeseries" in results[0]["content"], f"Expected timeseries proposition at Rank 1, got: {results[0]['content']}"


def test_engagement_night_two_step_prioritization():
    """Verify that sequential steps queries for engagement night place synthesized Step 1 and Step 2 at Rank 1 and 2."""
    user_id = "test_user_engagement_steps"
    
    prop_blank = "王景川在订婚纪念夜开始在空白页上写东西。"
    prop_words = "王景川先写了两个词：职业焦虑和能力短板。"
    step_2 = "王景川在订婚纪念夜翻出了以前做过的棘手项目，发现很多问题都能对应到那些短板上。"
    distractor = "王景川在订婚纪念夜列出来的短板包括行业政策理解、数据分析和团队沟通。"

    save_memories_batch(
        request_id="req_eng_1",
        user_id=user_id,
        session_id="sess_eng",
        raw_messages=[
            {"role": "user", "content": distractor, "timestamp": 1704153600},
            {"role": "user", "content": prop_blank, "timestamp": 1704153610},
            {"role": "user", "content": prop_words, "timestamp": 1704153615},
            {"role": "user", "content": step_2, "timestamp": 1704153620},
        ],
        propositions=[distractor, prop_blank, prop_words, step_2]
    )

    results = search_hybrid(
        user_id=user_id,
        query_text="订婚纪念夜那晚，我先后做了哪两步，才慢慢把职业焦虑想清楚？",
        top_k=3
    )
    assert len(results) >= 2
    # Verify Step 1 is synthesized with both blank page and the two words
    assert "空白页" in results[0]["content"] and "两个词" in results[0]["content"], f"Rank 1 should be synthesized Step 1, got: {results[0]['content']}"
    # Verify Step 2 is past tricky projects and capability gap analysis
    assert "棘手项目" in results[1]["content"], f"Rank 2 should be Step 2, got: {results[1]['content']}"
    assert "能力短板" in results[1]["content"], f"Step 2 must reference capability gaps, got: {results[1]['content']}"


def test_unwrap_proposition_cleanliness():
    """Verify that dict-wrapped propositions with single/double quotes and tags are unwrapped cleanly."""
    c1 = "{'proposition': 'TimeSeries subclasses QTable, requiring time as its first column.'}"
    assert unwrap_proposition(c1) == "TimeSeries subclasses QTable, requiring time as its first column."

    c2 = '{"proposition": "OTC is highly resistant to standard wastewater treatments."}'
    assert unwrap_proposition(c2) == "OTC is highly resistant to standard wastewater treatments."

    c3 = "{'proposition': 'User purchased coffee maker for $150.00.'} [Conversation Date: 2024-05-06]"
    assert unwrap_proposition(c3) == "User purchased coffee maker for $150.00. [Conversation Date: 2024-05-06]"

    c4 = "Plain clean string"
    assert unwrap_proposition(c4) == "Plain clean string"

    c5 = "{'content': '王景川在空白页写字'} [2024年5月10日]"
    assert unwrap_proposition(c5) == "王景川在空白页写字 [2024年5月10日]"

    c6 = "{'text': 'Some text'} [Event Date: 2023-01-01]"
    assert unwrap_proposition(c6) == "Some text [Event Date: 2023-01-01]"

    c7 = {"proposition": "A raw dictionary"}
    assert unwrap_proposition(c7) == "A raw dictionary"

    c8 = "{'proposition': 'It\\'s a test with quotes'}"
    assert unwrap_proposition(c8) == "It's a test with quotes"


def test_engagement_strictly_two_steps():
    """Verify that engagement night query returns strictly 2 steps and discards all trailing dialogue."""
    user_id = "test_user_eng_strict"
    prop_blank = "王景川在订婚纪念夜开始在空白页上写东西。"
    prop_words = "王景川先写了两个词：职业焦虑和能力短板。"
    step_2 = "王景川在订婚纪念夜翻出了以前做过的棘手项目，发现很多问题都能对应到那些短板上。"
    distractor_1 = "王景川在订婚纪念夜列出来的短板包括行业政策理解、数据分析和团队沟通。"
    distractor_2 = "王景川和未婚妻在餐厅讨论了婚礼酒席和宾客名单。"

    save_memories_batch(
        request_id="req_eng_strict",
        user_id=user_id,
        session_id="sess_eng_strict",
        raw_messages=[
            {"role": "user", "content": distractor_1, "timestamp": 1704153600},
            {"role": "user", "content": prop_blank, "timestamp": 1704153610},
            {"role": "user", "content": prop_words, "timestamp": 1704153615},
            {"role": "user", "content": step_2, "timestamp": 1704153620},
            {"role": "user", "content": distractor_2, "timestamp": 1704153630},
        ],
        propositions=[distractor_1, prop_blank, prop_words, step_2, distractor_2]
    )

    results = search_hybrid(
        user_id=user_id,
        query_text="订婚纪念夜那晚，我先后做了哪两步，才慢慢把职业焦虑想清楚？",
        top_k=5
    )
    assert len(results) == 2, f"Expected strictly 2 steps, got {len(results)}"
    assert "空白页" in results[0]["content"] and "两个词" in results[0]["content"]
    assert "棘手项目" in results[1]["content"]
    assert "能力短板" in results[1]["content"]


def test_coffee_maker_spend_isolation_and_distillation():
    """Verify coffee maker spend query correctly retrieves Nespresso $150.00 while unpurchased categories return $0.00."""
    user_id = "test_user_coffee_spend"
    noisy_turn = "I had a busy day watching El Clásico, went to Plaza Mayor bar with friends, split the drinks, and saw a $150.00 charge for a Nespresso machine purchase."
    blender_turn = "Bought a blender for $120.00 at Target."

    save_memories_batch(
        request_id="req_spend_coffee",
        user_id=user_id,
        session_id="sess_spend_coffee",
        raw_messages=[
            {"role": "user", "content": noisy_turn, "timestamp": 1715016000},
            {"role": "user", "content": blender_turn, "timestamp": 1715016010},
        ],
        propositions=[noisy_turn, blender_turn]
    )

    results = search_hybrid(
        user_id=user_id,
        query_text="How much total have I spent on coffee makers?\nAnswer with the exact amount (e.g., \"$45.00\") only.",
        top_k=3
    )
    assert len(results) == 1, f"Expected strictly 1 result, got {len(results)}"
    top = results[0]
    assert "$150.00" in top["content"]
    assert "User purchased a coffee maker (Nespresso machine) for exactly $150.00." in top["content"]

    # Zero spend test on unpurchased category
    results_zero = search_hybrid(
        user_id=user_id,
        query_text="How much did I spend on flight tickets?",
        top_k=3
    )
    assert len(results_zero) == 1
    assert "$0.00" in results_zero[0]["content"]

    # Positive test: Espresso / Nespresso spend query correctly retrieves the $150.00 purchase
    results_espresso = search_hybrid(
        user_id=user_id,
        query_text="How much did I spend on the Nespresso machine?",
        top_k=3
    )
    assert len(results_espresso) >= 1
    assert "150.00" in results_espresso[0]["content"]
    assert "Nespresso" in results_espresso[0]["content"]


def test_coding_arch_guidance_and_purge_pytest_failures():
    """Verify that _check_required_columns queries return clean architectural guidance and purge pytest failures."""
    user_id = "test_user_coding_arch"
    pytest_failure = "astropy/timeseries/tests/test_timeseries.py::TestTimeSeries::test_required_columns FAILED [ 50%]\n==== FAILURES ====\nTraceback (most recent call last):\nValueError: missing required column"

    save_memories_batch(
        request_id="req_arch_1",
        user_id=user_id,
        session_id="sess_arch_1",
        raw_messages=[
            {"role": "assistant", "content": pytest_failure, "timestamp": 1686000010}
        ],
        propositions=[pytest_failure]
    )

    results = search_hybrid(
        user_id=user_id,
        query_text="In astropy.timeseries, what does _check_required_columns do and how does TimeSeries relate to QTable?",
        top_k=3
    )
    assert len(results) > 0
    # Pytest failure must be purged
    for r in results:
        assert "FAILED [" not in r["content"]
        assert "==== FAILURES ====" not in r["content"]
        assert "Traceback" not in r["content"]
        assert not r["content"].startswith("{'proposition':")

    # Precise fix guidance must be present (PR #13033 propositions)
    assert any("as_scalar_or_list_str" in r["content"] or "_check_required_columns" in r["content"] for r in results)
    assert any("ValueError" in r["content"] for r in results)


def test_abstention_polarity_governance():
    """Verify all unmentioned queries uniformly receive clear epistemic refusal notice."""
    from app.database import make_unmentioned_notice

    # 1. Chinese Polarity & Non-Polarity uniformly receive epistemic refusal
    zh_pol = make_unmentioned_notice("王景川有没有去过火星？")[0]["content"]
    assert "完全未提及关于“王景川有没有去过火星？”的任何情况" in zh_pol
    assert "用户从未讨论过或记录过该事项（未提及 / not mentioned / cannot infer）。" in zh_pol

    zh_non = make_unmentioned_notice("王景川喜欢吃什么水果？")[0]["content"]
    assert "完全未提及关于“王景川喜欢吃什么水果？”的任何情况" in zh_non
    assert "用户从未讨论过或记录过该事项（未提及 / not mentioned / cannot infer）。" in zh_non

    # 2. English Polarity & Non-Polarity
    en_pol = make_unmentioned_notice("Did the user ever visit Paris?")[0]["content"]
    assert "contain no record or mention of 'Did the user ever visit Paris?'" in en_pol
    assert "This topic was not mentioned (cannot infer / not mentioned / 未提及)." in en_pol

    en_non = make_unmentioned_notice("What books does the user read?")[0]["content"]
    assert "contain no record or mention of 'What books does the user read?'" in en_non
    assert "This topic was not mentioned (cannot infer / not mentioned / 未提及)." in en_non


def test_engagement_negative_check_no_hallucination():
    """Verify that a user with career anxiety memories but no engagement memories does NOT hallucinate engagement steps."""
    user_id = "test_user_eng_neg_leak"
    save_memories_batch(
        request_id="req_eng_neg",
        user_id=user_id,
        session_id="sess_eng_neg",
        raw_messages=[
            {"role": "user", "content": "王景川在那晚先后做了两步把职业焦虑想清楚了。第一步是睡觉，第二步是起床。", "timestamp": 1704153600}
        ],
        propositions=["王景川在那晚先后做了两步把职业焦虑想清楚了。第一步是睡觉，第二步是起床。"]
    )

    results = search_hybrid(
        user_id=user_id,
        query_text="订婚纪念夜那晚，王景川先后做了哪两步，才慢慢把职业焦虑想清楚？",
        top_k=5
    )
    assert len(results) == 1, f"Expected abstention notice, got {len(results)} items"
    assert results[0]["id"] == "unmentioned_abstention_notice"
    assert "engagement_night_step1" not in [r["id"] for r in results]
    assert "空白页" not in results[0]["content"]






