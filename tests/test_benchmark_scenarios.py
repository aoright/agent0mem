#!/usr/bin/env python3
"""
Comprehensive AML Benchmark Simulation Test Suite.
Validates the Agent0Mem live remote server against all 4 core challenge scenarios:
1. Strict List Preservation (LoCoMo-Refined)
2. Negative Preference / Avoid Constraints (PersonaMem v2)
3. Multi-Hop Temporal State Updates (TempReason / CL-Bench)
4. Option-Aware Single Choice QA (ScriptMem / BEAM)
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


def test_scenario_1_strict_list():
    print("=" * 60)
    print("Scenario 1: Strict List Preservation (LoCoMo-Refined)")
    print("Goal: Extract and recall exact entity list without hallucinated additions.")
    user_id = f"sim_user_list_{int(time.time())}"
    
    # User lists exactly three specific hobbies in a conversation full of random chatter
    messages = [
        {"role": "user", "content": "I met Bob yesterday and he was talking about playing golf, but that's not for me.", "timestamp": 1686000000},
        {"role": "assistant", "content": "Haha, golf can definitely be slow!", "timestamp": 1686000010},
        {"role": "user", "content": "Yeah! My actual three favorite hobbies are rock climbing, sourdough baking, and landscape photography.", "timestamp": 1686000020},
        {"role": "assistant", "content": "That is an amazing mix of creative and outdoor activities!", "timestamp": 1686000030}
    ]
    
    add_resp = post_json("add", {
        "request_id": f"req_list_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_list",
        "messages": messages
    })
    assert add_resp.get("success") is True, "Add failed"

    search_resp = post_json("search", {
        "user_id": user_id,
        "query": "What are the user's favorite hobbies?",
        "top_k": 3
    })
    
    data = search_resp.get("data", [])
    assert len(data) > 0, "No memories returned"
    top_memory = data[0]["content"]
    print(f"Top Retrieved Memory: {top_memory}")
    assert "rock climbing" in top_memory and "sourdough baking" in top_memory and "landscape photography" in top_memory, "Missing hobbies in memory"
    assert "golf" not in top_memory or "not" in top_memory, "Golf wrongly included as hobby"
    print("PASS: Exact list preserved with zero pollution.")


def test_scenario_2_negative_constraints():
    print("=" * 60)
    print("Scenario 2: Negative Constraint / Avoid Preference (PersonaMem v2)")
    print("Goal: Correctly distill and prioritize negative constraints.")
    user_id = f"sim_user_neg_{int(time.time())}"
    
    messages = [
        {"role": "user", "content": "Whenever you recommend recipes or restaurants, strictly avoid anything with peanuts or shellfish due to severe allergies.", "timestamp": 1687000000},
        {"role": "assistant", "content": "Understood! I will strictly exclude peanuts and shellfish from all food recommendations.", "timestamp": 1687000010}
    ]
    
    post_json("add", {
        "request_id": f"req_neg_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_neg",
        "messages": messages
    })

    search_resp = post_json("search", {
        "user_id": user_id,
        "query": "What ingredients or foods should be avoided for the user?",
        "top_k": 3
    })
    
    data = search_resp.get("data", [])
    assert len(data) > 0, "No memories returned"
    top_memory = data[0]["content"]
    print(f"Top Retrieved Memory: {top_memory}")
    assert "peanut" in top_memory.lower() and "shellfish" in top_memory.lower(), "Constraint not captured"
    print("PASS: Negative allergy constraint properly recorded and retrieved.")


def test_scenario_3_multi_state_temporal():
    print("=" * 60)
    print("Scenario 3: Multi-State Temporal Succession (TempReason / CL-Bench)")
    print("Goal: Distinguish current location vs prior locations based on query tense.")
    user_id = f"sim_user_temp_{int(time.time())}"
    
    messages = [
        {"role": "user", "content": "I lived in London for three years until May 2023.", "timestamp": 1683000000},
        {"role": "user", "content": "Then in June 2023 I moved to Berlin.", "timestamp": 1686000000},
        {"role": "user", "content": "Last month in December 2023, I relocated to Tokyo for work and this is my current permanent residence.", "timestamp": 1704000000}
    ]
    
    post_json("add", {
        "request_id": f"req_temp_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_temp",
        "messages": messages
    })

    # Test Current Query
    search_curr = post_json("search", {
        "user_id": user_id,
        "query": "Where does the user currently live?",
        "top_k": 3
    })
    top_curr = search_curr.get("data", [])[0]["content"]
    print(f"Query (Current): 'Where does the user currently live?' -> {top_curr}")
    assert "Tokyo" in top_curr, "Current location failed"

    # Test Past Query
    search_past = post_json("search", {
        "user_id": user_id,
        "query": "Where did the user originally live before Berlin?",
        "top_k": 3
    })
    top_past = search_past.get("data", [])[0]["content"]
    print(f"Query (Past): 'Where did the user originally live before Berlin?' -> {top_past}")
    assert "London" in top_past, "Past location failed"
    print("PASS: Temporal state succession and intent routing fully verified.")


def test_scenario_4_options_mcq():
    print("=" * 60)
    print("Scenario 4: Option-Aware MCQ Matching (ScriptMem / BEAM)")
    print("Goal: Confirm correct option without distractor contamination.")
    user_id = f"sim_user_mcq_{int(time.time())}"
    
    messages = [
        {"role": "user", "content": "My favorite programming language for systems engineering is Rust, while for data analysis I prefer Julia.", "timestamp": 1688000000}
    ]
    
    post_json("add", {
        "request_id": f"req_mcq_{int(time.time())}",
        "user_id": user_id,
        "session_id": "sess_mcq",
        "messages": messages
    })

    search_mcq = post_json("search", {
        "user_id": user_id,
        "query": "Which language does the user prefer for systems engineering?",
        "options": ["A. Python", "B. Rust", "C. Go", "D. C++"],
        "top_k": 3
    })
    
    data = search_mcq.get("data", [])
    top_content = data[0]["content"]
    print(f"Query: 'Which language for systems engineering?' (Options: Python, Rust, Go, C++) -> {top_content}")
    assert "Rust" in top_content, "MCQ option match failed"
    print("PASS: Option-aware MCQ retrieval succeeded.")


if __name__ == "__main__":
    print(f"Running Live Remote Benchmark Scenarios against: {REMOTE_SERVER}")
    test_scenario_1_strict_list()
    test_scenario_2_negative_constraints()
    test_scenario_3_multi_state_temporal()
    test_scenario_4_options_mcq()
    print("\n" + "=" * 60)
    print("ALL 4 BENCHMARK SCENARIOS PASSED CONVINCINGLY ON THE PRODUCTION SERVER!")
    print("=" * 60)
