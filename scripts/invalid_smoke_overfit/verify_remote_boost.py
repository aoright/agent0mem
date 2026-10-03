#!/usr/bin/env python3
"""
Remote Production Verification Script for /boost optimizations.
Tests against http://47.97.127.223/agent0mem
Validates all 7 delivered optimization criteria live on production:
1. is_code_q query classification and distractor code isolation
2. Reversing tool-use vs diff boost (complete unified diff prioritized)
3. Purge raw tool calls from indexing
4. Coding search context budgeting (top 2, strictly <= 2,500 bytes total)
5. Abstention verification fallback
6. Temporal sequence sorting (chronological timestamp ASC)
7. Clean output presentation (stripped bracketed temporal tags, preserved created_at)
"""

import time
import json
import urllib.request
import urllib.error

REMOTE_BASE = "http://47.97.127.223/agent0mem"


def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{REMOTE_BASE}/{endpoint.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_1_code_keywords_and_isolation():
    print("=" * 65)
    print("Test 1: is_code_q Classification & Distractor Patch Isolation")
    print("=" * 65)
    user_id = f"prod_test_code_q_{int(time.time())}"
    
    # Ingest a 2013 console patch
    post_json("add", {
        "request_id": f"req_cq_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_cq",
        "messages": [
            {"role": "assistant", "content": "diff --git a/astropy/console.py b/astropy/console.py\n--- a/astropy/console.py\n+++ b/astropy/console.py\n@@ -10,3 +10,4 @@", "timestamp": 1380000000}
        ]
    })
    
    # Query with expanded code keyword 'missing required column'
    res = post_json("search", {
        "user_id": user_id,
        "query": "object is invalid - missing required column",
        "top_k": 5
    })
    data = res.get("data", [])
    console_leaked = any("console.py" in r["content"] for r in data)
    assert not console_leaked, "Distractor console.py patch leaked!"
    print(f"PASS: Query classified as code, distractor console.py properly isolated. Returned items: {len(data)}")


def test_2_and_3_diff_priority_and_purge_tool_calls():
    print("=" * 65)
    print("Test 2 & 3: Unified Diff Priority & Purge Raw Tool Calls")
    print("=" * 65)
    user_id = f"prod_test_diff_prio_{int(time.time())}"
    
    # Send trajectory with raw tool calls and 1 diff patch
    post_json("add", {
        "request_id": f"req_dp_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_dp",
        "messages": [
            {"role": "assistant", "content": '[tool_use Read] {"file_path": "astropy/timeseries/sampled.py"}', "timestamp": 1686000010},
            {"role": "assistant", "content": '[tool_use TaskUpdate] {"taskId": "1", "status": "in_progress"}', "timestamp": 1686000020},
            {"role": "assistant", "content": """I resolved the issue by editing astropy/timeseries/sampled.py:
```diff
diff --git a/astropy/timeseries/sampled.py b/astropy/timeseries/sampled.py
--- a/astropy/timeseries/sampled.py
+++ b/astropy/timeseries/sampled.py
@@ -248,6 +248,8 @@ class TimeSeries:
+        if isinstance(epoch_time, Quantity):
+            epoch_time = epoch_time.to(u.day).value
```
All unit tests in test_sampled.py pass.""", "timestamp": 1686000030}
        ]
    })
    
    res = post_json("search", {
        "user_id": user_id,
        "query": "What is the patch or solution to fix the fold method TypeError in astropy.timeseries?",
        "top_k": 3
    })
    data = res.get("data", [])
    assert len(data) > 0, "No results returned"
    top_c = data[0]["content"]
    assert "diff --git" in top_c, "Top item must be the unified git diff patch"
    for r in data:
        assert "[tool_use Read]" not in r["content"], "Raw [tool_use Read] leaked into search results"
        assert "[tool_use TaskUpdate]" not in r["content"], "Raw [tool_use TaskUpdate] leaked into search results"
    print("PASS: Unified git diff prioritized at Rank 1, raw tool calls purged.")


def test_4_coding_context_budgeting():
    print("=" * 65)
    print("Test 4: Coding Context Budgeting (Top 2 Cleanest <= 2500 bytes)")
    print("=" * 65)
    user_id = f"prod_test_budget_{int(time.time())}"
    
    p1 = "diff --git a/mod1.py b/mod1.py\n" + ("+ line of code fix\n" * 100)
    p2 = "diff --git a/mod2.py b/mod2.py\n" + ("+ line of code fix\n" * 100)
    p3 = "diff --git a/mod3.py b/mod3.py\n" + ("+ line of code fix\n" * 100)
    
    post_json("add", {
        "request_id": f"req_bg_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_bg",
        "messages": [
            {"role": "assistant", "content": p1, "timestamp": 1686000010},
            {"role": "assistant", "content": p2, "timestamp": 1686000020},
            {"role": "assistant", "content": p3, "timestamp": 1686000030},
        ]
    })
    
    res = post_json("search", {
        "user_id": user_id,
        "query": "patch line of code fix mod",
        "top_k": 5
    })
    data = res.get("data", [])
    assert len(data) <= 2, f"Expected <= 2 items, got {len(data)}"
    total_bytes = sum(len(r["content"].encode("utf-8")) for r in data)
    assert total_bytes <= 2500, f"Expected total bytes <= 2500, got {total_bytes}"
    print(f"PASS: Capped at {len(data)} items, total bytes {total_bytes} <= 2500 budget.")


def test_5_temporal_sequence_sorting():
    print("=" * 65)
    print("Test 5: Temporal Sequence Sorting (Timestamp ASC)")
    print("=" * 65)
    user_id = f"prod_test_seq_{int(time.time())}"
    
    # Ingest 3 steps in reverse chronological order
    post_json("add", {
        "request_id": f"req_sq_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_sq",
        "messages": [
            {"role": "user", "content": "第三步：在东京完成最终项目签约交付。", "timestamp": 1709251200},
            {"role": "user", "content": "第二步：在柏林完成中期技术评审和演示。", "timestamp": 1706745600},
            {"role": "user", "content": "第一步：在伦敦启动项目并确立技术架构。", "timestamp": 1704067200},
        ]
    })
    
    res = post_json("search", {
        "user_id": user_id,
        "query": "请按先后顺序说明项目推进的第一步和最终交付。",
        "top_k": 3
    })
    data = res.get("data", [])
    assert len(data) >= 2, f"Expected at least 2 items, got {len(data)}"
    assert "伦敦" in data[0]["content"], f"First chronological item must be London (step 1), got: {data[0]['content']}"
    print(f"PASS: Sequence indicators re-sorted results in chronological order (Rank 1: {data[0]['content'][:40]}...).")


def test_6_clean_output_presentation():
    print("=" * 65)
    print("Test 6: Clean Output Presentation (No bracketed temporal tags)")
    print("=" * 65)
    user_id = f"prod_test_clean_{int(time.time())}"
    
    post_json("add", {
        "request_id": f"req_cl_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_cl",
        "messages": [
            {"role": "user", "content": "I purchased an espresso machine for $150.00.", "timestamp": 1715342400}
        ]
    })
    
    res = post_json("search", {
        "user_id": user_id,
        "query": "How much was the espresso machine?",
        "top_k": 1
    })
    data = res.get("data", [])
    assert len(data) > 0, "No result returned"
    top = data[0]
    c = top["content"]
    assert "[Conversation Date:" in c, "Preserved [Conversation Date:] missing from content"
    assert "[Year:" not in c, "Cumbersome [Year:] leaked into content"
    assert "[Month:" not in c, "Cumbersome [Month:] leaked into content"
    assert "$150.00" in c, "Exact entity $150.00 missing from clean content"
    assert top.get("created_at") is not None, "Metadata created_at missing"
    print(f"PASS: Clean output presentation verified: '{c}' (created_at={top.get('created_at')})")


def test_7_mismatched_patch_isolation():
    print("=" * 65)
    print("Test 7: Mismatched Patch Poisoning Prevention")
    print("=" * 65)
    user_id = f"prod_test_mismatch_{int(time.time())}"
    
    table_patch = "diff --git a/astropy/table/table.py b/astropy/table/table.py\n--- a/astropy/table/table.py\n+++ b/astropy/table/table.py\n@@ -100,6 +100,8 @@ def to_pandas(self):\n+    if index_name:\n+        del self.columns[index_name]\n"
    timeseries_prop = "In astropy/timeseries/core.py, _check_required_columns raises an error when required columns are removed."

    post_json("add", {
        "request_id": f"req_mm_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_mm",
        "messages": [
            {"role": "assistant", "content": table_patch, "timestamp": 1628188560},
            {"role": "assistant", "content": timeseries_prop, "timestamp": 1628188570},
        ]
    })

    res = post_json("search", {
        "user_id": user_id,
        "query": "timeseries core.py _check_required_columns fix missing required column test",
        "top_k": 3
    })
    data = res.get("data", [])
    assert len(data) > 0, "No result returned"
    assert "table.py" not in data[0]["content"], f"Mismatched table.py patch promoted to Rank 1! Got: {data[0]['content']}"
    assert "timeseries" in data[0]["content"], f"Expected timeseries proposition at Rank 1, got: {data[0]['content']}"
    print(f"PASS: Mismatched table.py patch isolated from timeseries query. Rank 1: '{data[0]['content'][:60]}...'")


def test_8_engagement_night_two_steps():
    print("=" * 65)
    print("Test 8: Engagement Night Two-Step Prioritization & Synthesis")
    print("=" * 65)
    uid_wang = "u_401b916b2a68e0bcecb4006d37fe0cc86ba8b773415bb93c96ed6c436b8d38d1"
    res = post_json("search", {
        "user_id": uid_wang,
        "query": "订婚纪念夜那晚，我先后做了哪两步，才慢慢把职业焦虑想清楚？",
        "top_k": 3
    })
    data = res.get("data", [])
    assert len(data) >= 2, f"Expected at least 2 items, got {len(data)}"
    # Step 1 should synthesize blank page and the two words
    assert "空白页" in data[0]["content"] and "两个词" in data[0]["content"], f"Rank 1 must be synthesized Step 1, got: {data[0]['content']}"
    assert "棘手项目" in data[1]["content"], f"Rank 2 must be Step 2, got: {data[1]['content']}"
    print(f"PASS: Two steps cleanly prioritized at Rank 1 and Rank 2:\n  Rank 1: {data[0]['content'][:60]}...\n  Rank 2: {data[1]['content'][:60]}...")


def test_9_atomic_retrieval_date_preservation():
    print("=" * 65)
    print("Test 9: Atomic Retrieval Date Preservation (atom_15)")
    print("=" * 65)
    uid_v1 = "u_56c736bdf518e9f006716c551aa8c796aece8ed9d7551dbd83f333a71b312b8b"
    res = post_json("search", {
        "user_id": uid_v1,
        "query": "What date is recorded for the conversation discussing the Ethos Water deck piece?",
        "top_k": 3
    })
    data = res.get("data", [])
    assert len(data) > 0, "No results returned for atom_15"
    top = data[0]
    assert "2024-01-22" in top["content"], f"Expected '2024-01-22' in content, got: {top['content']}"
    assert "2024-01-22" in top.get("text", ""), f"Expected '2024-01-22' in text, got: {top.get('text')}"
    print(f"PASS: Date '2024-01-22' preserved in content and text: '{top['content'][:70]}...'")


def test_10_direct_recall_coffee_maker_isolation():
    print("=" * 65)
    print("Test 10: Direct Recall Spend Isolation (Coffee Maker $150.00)")
    print("=" * 65)
    uid_spend = "u_068888a05600c99405f9b4d4d21216ba5c5b9ffb87c3340db5c33c47876729f4"
    res = post_json("search", {
        "user_id": uid_spend,
        "query": "How much total have I spent on coffee makers?",
        "top_k": 3
    })
    data = res.get("data", [])
    assert len(data) == 1, f"Expected strictly 1 item for coffee maker spend, got {len(data)}"
    top = data[0]
    assert "150.00" in top["content"], f"Expected $150.00 in content, got: {top['content']}"
    assert "120" not in top["content"], f"Contaminated with blender amount: {top['content']}"
    assert "270" not in top["content"], f"Contaminated with combined amount: {top['content']}"
    print(f"PASS: Exact $150.00 isolated without blender/combined contamination: '{top['content']}'")


def test_11_in_domain_epistemic_abstention_all_9():
    print("=" * 65)
    print("Test 11: In-Domain Epistemic Abstention (All 9 Verification Queries)")
    print("=" * 65)
    uid_wang = "u_401b916b2a68e0bcecb4006d37fe0cc86ba8b773415bb93c96ed6c436b8d38d1"
    verification_queries = [
        ("gov_1", "我那次帮舅妈看货时，后来有没有再推荐过别的批发渠道？"),
        ("gov_2", "大年初一帮舅妈进货之后，我有没有把那次经验单独记下来？"),
        ("gov_3", "春节前我给阿拉斯加寄养时，最后选店最看重的到底是哪个条件？"),
        ("gov_4", "我以前是不是有过一次晨跑后就直接放弃打卡的经历？"),
        ("gov_5", "节后第一周晚上记账时，我更在意的是存款还是还款压力？"),
        ("gov_6", "我春节后第一次恢复早起，是哪一天开始的？"),
        ("gov_7", "帮我想想，春节后我有没有给自己设过更具体的跑步目标？"),
        ("gov_8", "我春节前后给爸妈和自己准备的花费里，哪一类最容易超预算？"),
        ("gov_9", "春节后那阵子，我最先想调整的是作息还是运动习惯？"),
    ]
    for qid, q in verification_queries:
        res = post_json("search", {
            "user_id": uid_wang,
            "query": q,
            "top_k": 3
        })
        data = res.get("data", [])
        assert len(data) > 0, f"No response returned for {qid}"
        c = data[0]["content"]
        assert "完全未提及" in c, f"Expected explicit epistemic abstention statement for {qid}, got: '{c}'"
        print(f"PASS: {qid} cleanly abstained: '{c[:60]}...'")


def main():
    print("\nStarting Live Remote Production Optimization Verification...\n")
    test_1_code_keywords_and_isolation()
    test_2_and_3_diff_priority_and_purge_tool_calls()
    test_4_coding_context_budgeting()
    test_5_temporal_sequence_sorting()
    test_6_clean_output_presentation()
    test_7_mismatched_patch_isolation()
    test_8_engagement_night_two_steps()
    test_9_atomic_retrieval_date_preservation()
    test_10_direct_recall_coffee_maker_isolation()
    test_11_in_domain_epistemic_abstention_all_9()
    print("\n" + "=" * 65)
    print("ALL REMOTE PRODUCTION VERIFICATIONS PASSED WITH 100% PRECISION!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()


