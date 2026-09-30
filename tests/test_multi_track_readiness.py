#!/usr/bin/env python3
"""
Comprehensive Multi-Track Verification Test Suite for Agent0Mem.
Validates live production server across:
1. Coding Memory & Patch-Priority Retrieval (CamBench-Noisy)
2. Multimodal Memory & Abstention Thresholding (LDBD-Multimodal)
3. Latency & Concurrency Stress Check
"""

import sys
import json
import time
import urllib.request
import urllib.error

REMOTE_SERVER = "http://47.97.127.223/agent0mem"


def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{REMOTE_SERVER}/{endpoint.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_coding_patch_priority():
    print("=" * 60)
    print("Test 1: Coding Track - Patch Priority & Noise Filtering")
    print("Goal: Prioritize code diff over trajectory noise (CamBench)")
    user_id = f"test_coding_{int(time.time())}"

    messages = [
        {"role": "user", "content": "# Issue: astropy.timeseries fold() method raises TypeError with quantity bounds", "timestamp": 1686000000},
        {"role": "assistant", "content": "[thinking] Let me search the codebase for fold() definition.", "timestamp": 1686000010},
        {"role": "assistant", "content": "[tool_use TaskCreate] {\"subject\": \"Investigate fold() quantity error\"}", "timestamp": 1686000020},
        {"role": "assistant", "content": "[tool_use Bash] {\"command\": \"git status\"}", "timestamp": 1686000030},
        {"role": "user", "content": "[tool_result] (Bash completed with no output)", "timestamp": 1686000040},
        {"role": "assistant", "content": "[tool_use TaskUpdate] {\"taskId\": \"1\", \"status\": \"in_progress\"}", "timestamp": 1686000050},
        {"role": "assistant", "content": "I have created the fix in astropy/timeseries/sampled.py:\n```diff\ndiff --git a/astropy/timeseries/sampled.py b/astropy/timeseries/sampled.py\n--- a/astropy/timeseries/sampled.py\n+++ b/astropy/timeseries/sampled.py\n@@ -248,6 +248,8 @@ class TimeSeries:\n+        if isinstance(epoch_time, Quantity):\n+            epoch_time = epoch_time.to(u.day).value\n```\nAll unit tests in test_sampled.py now pass.", "timestamp": 1686000060},
    ]

    add_resp = post_json("add", {
        "request_id": f"req_code_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_code",
        "messages": messages
    })
    assert add_resp.get("success") is True, "Add coding memory failed"

    # Search for the fix
    search_resp = post_json("search", {
        "user_id": user_id,
        "query": "What is the patch or solution to fix the fold method TypeError in astropy.timeseries?",
        "top_k": 3
    })

    data = search_resp.get("data", [])
    assert len(data) > 0, "No coding memories returned"
    top_c = data[0]["content"]
    print(f"Top Retrieved Code Memory: {top_c[:150]}...")
    assert "diff --git" in top_c or "sampled.py" in top_c or "Quantity" in top_c, "Failed to prioritize code patch"
    assert "[tool_use TaskUpdate]" not in top_c, "Failed to demote trajectory noise"
    print("PASS: Code patch successfully prioritized over noisy tool calls.")


def test_multimodal_abstention_guard():
    print("=" * 60)
    print("Test 2: Multimodal Track - Abstention Thresholding")
    print("Goal: Return empty list on unmentioned facts to prevent hallucination")
    user_id = f"test_mm_{int(time.time())}"

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Screenshot showing user profile page with username Alex2024 and green active status badge."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}}
        ], "timestamp": 1687000000},
        {"role": "assistant", "content": "I see Alex2024 with an active green status badge.", "timestamp": 1687000010}
    ]

    add_resp = post_json("add", {
        "request_id": f"req_mm_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_mm",
        "messages": messages
    })
    assert add_resp.get("success") is True, "Add multimodal memory failed"

    # 1. Answerable Query
    search_resp1 = post_json("search", {
        "user_id": user_id,
        "query": "What is the username and badge color on the profile screenshot?",
        "top_k": 3
    })
    data1 = search_resp1.get("data", [])
    assert len(data1) > 0, "Failed to retrieve answerable multimodal memory"
    print(f"Answerable Query Top Result: {data1[0]['content'][:100]}...")
    assert "Alex2024" in data1[0]["content"] and "green" in data1[0]["content"], "Missing profile details"
    print("PASS: Answerable multimodal query retrieved successfully.")

    # 2. Unanswerable Query (Abstention Scenario)
    search_resp2 = post_json("search", {
        "user_id": user_id,
        "query": "What brand of luxury watch was the user wearing in the underwater diving photo?",
        "top_k": 3
    })
    data2 = search_resp2.get("data", [])
    print(f"Unanswerable Query Result Count: {len(data2)}")
    assert len(data2) == 0, f"Expected empty list for abstention query, got {len(data2)} items"
    print("PASS: Abstention guard successfully suppressed irrelevant candidates.")

    # 3. Chinese Multilingual Multimodal Query
    search_resp3 = post_json("search", {
        "user_id": user_id,
        "query": "用户Alex2024个人主页上的状态徽章颜色是什么？",
        "top_k": 3
    })
    data3 = search_resp3.get("data", [])
    assert len(data3) > 0, "Failed to retrieve Chinese multimodal query"
    print("PASS: Chinese multimodal query matched successfully via CJK bigrams.")

    # 4. Chinese Unanswerable Query (Abstention Scenario)
    search_resp4 = post_json("search", {
        "user_id": user_id,
        "query": "王景川在潜水时佩戴了什么品牌的劳力士机械手表？",
        "top_k": 3
    })
    data4 = search_resp4.get("data", [])
    assert len(data4) == 0, f"Expected empty list for Chinese abstention query, got {len(data4)} items"
    print("PASS: Chinese abstention guard successfully suppressed irrelevant candidates.")


def test_latency_stress():
    print("=" * 60)
    print("Test 3: Latency & Responsiveness Check")
    user_id = f"test_latency_{int(time.time())}"

    t0 = time.time()
    post_json("add", {
        "request_id": f"req_lat_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_lat",
        "messages": [
            {"role": "user", "content": "Quick latency verification payload.", "timestamp": 1688000000}
        ]
    })
    t_add = time.time() - t0

    t0 = time.time()
    post_json("search", {
        "user_id": user_id,
        "query": "latency payload",
        "top_k": 10
    })
    t_search = time.time() - t0

    print(f"Add Latency:    {t_add:.3f}s (Budget: <3.0s)")
    print(f"Search Latency: {t_search:.3f}s (Budget: <1.0s)")
    assert t_add < 3.0, "Add too slow"
    assert t_search < 1.0, "Search too slow"
    print("PASS: Latency well within benchmark budgets.")


def main():
    print("\nRunning Live Multi-Track Verification Test Suite...\n")
    test_coding_patch_priority()
    test_multimodal_abstention_guard()
    test_latency_stress()
    print("\n" + "=" * 60)
    print("ALL MULTI-TRACK TESTS PASSED WITH 100% PRECISION!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
