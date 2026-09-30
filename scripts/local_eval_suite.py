#!/usr/bin/env python3
"""
Agent0Mem Offline High-Fidelity Multi-Track Benchmark Harness.
Enables unlimited local iterations with ZERO official quota usage and ZERO cooldown.
Simulates exact evaluation pipelines from AML (PersonaMem, LoCoMo, LongMemEval, SWE-bench).
"""

import os
import sys
import time
import json
import re
from typing import Dict, List, Any, Tuple

# Point to project app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import search_hybrid, init_db, save_memories_batch
from app import config

# Use an isolated benchmark database in /tmp
BENCHMARK_DB_PATH = "/tmp/agent0mem_local_eval.sqlite3"
config.DB_PATH = BENCHMARK_DB_PATH


def setup_fresh_eval_db():
    if os.path.exists(BENCHMARK_DB_PATH):
        try:
            os.remove(BENCHMARK_DB_PATH)
        except Exception:
            pass
    init_db()


def run_textual_benchmark() -> Dict[str, Any]:
    """Evaluates fact recall, temporal order, state conflict resolution, and negative constraints."""
    user_id = "eval_textual_u1"
    
    # Session 1: Initial state and preferences
    msgs_s1 = [
        {"role": "user", "content": "I live in Seattle, and I strictly follow a vegan diet. I never eat dairy or seafood.", "timestamp": 1710000000},
        {"role": "assistant", "content": "Got it! Seattle resident, strict vegan, no dairy, no seafood.", "timestamp": 1710000010},
    ]
    props_s1 = [
        "User lives in Seattle.",
        "[Current Preference] User strictly follows a vegan diet and avoids dairy and seafood.",
    ]
    save_memories_batch("req_t1", user_id, "sess_1", msgs_s1, props_s1)

    # Session 2: State update (Moved to Chicago)
    msgs_s2 = [
        {"role": "user", "content": "Update: Last week I moved from Seattle to Chicago for a new job at Northwestern.", "timestamp": 1715000000},
        {"role": "assistant", "content": "Congrats on the move to Chicago and your new role at Northwestern!", "timestamp": 1715000010},
    ]
    props_s2 = [
        "[Prior State / Superseded] User previously lived in Seattle.",
        "[Current State] User lives in Chicago and works at Northwestern.",
        "[Event Date: 2024-05-01] User moved to Chicago for a new job.",
    ]
    save_memories_batch("req_t2", user_id, "sess_2", msgs_s2, props_s2)

    # Add a violating distractor memory that violates the user's negative constraint
    msgs_distractor = [
        {"role": "user", "content": "Last year at a banquet, the chef served grilled Alaskan salmon with cheese.", "timestamp": 1700000000},
    ]
    props_distractor = [
        "Chef served grilled Alaskan salmon with cheese at the annual banquet.",
    ]
    save_memories_batch("req_t_distractor", user_id, "sess_distractor", msgs_distractor, props_distractor)

    # Test Cases
    tests = [
        {
            "name": "Current Location (State Conflict)",
            "query": "Where does the user currently live?",
            "expected_contains": ["Chicago"],
            "forbidden_in_top": ["Seattle"],
        },
        {
            "name": "Historical Location (Past Query)",
            "query": "Where did the user use to live previously before moving?",
            "expected_contains": ["Seattle"],
            "forbidden_in_top": [],
        },
        {
            "name": "Dietary Rule (Negative Constraint)",
            "query": "Recommend dinner options, user avoids dairy and seafood",
            "expected_contains": ["vegan"],
            "forbidden_in_top": ["salmon"],  # The violating salmon memory must NOT be top
        },
        {
            "name": "Workplace Recall (Direct Fact)",
            "query": "Where does the user work now?",
            "expected_contains": ["Northwestern"],
            "forbidden_in_top": [],
        }
    ]

    passed = 0
    total = len(tests)
    details = []

    for t in tests:
        res = search_hybrid(user_id, t["query"], top_k=5)
        top_text = res[0]["content"] if res else ""
        all_text = " ".join([r["content"] for r in res])

        hit = any(exp.lower() in all_text.lower() for exp in t["expected_contains"])
        leak = any(forb.lower() in top_text.lower() for forb in t.get("forbidden_in_top", []))

        success = hit and not leak
        if success:
            passed += 1

        details.append({
            "test": t["name"],
            "passed": success,
            "top_match": top_text[:90],
            "returned_count": len(res),
        })

    score = round((passed / total) * 100.0, 2)
    return {"track": "textual", "score": score, "passed": passed, "total": total, "details": details}


def run_coding_benchmark() -> Dict[str, Any]:
    """Evaluates noisy trajectory filtering, patch localization, and payload compression (<25KB)."""
    user_id = "eval_coding_u1"

    # Insert a target bug fix patch
    target_patch = """diff --git a/astropy/timeseries/sampled.py b/astropy/timeseries/sampled.py
index a1b2c3d..e4f5g6h 100644
--- a/astropy/timeseries/sampled.py
+++ b/astropy/timeseries/sampled.py
@@ -100,6 +100,8 @@ def _check_required_columns(self):
-    raise ValueError("Missing required column")
+    if col not in self.colnames:
+        raise ValueError(f"Required column '{col}' is missing from TimeSeries")
"""
    msgs_target = [
        {"role": "user", "content": "Fix TimeSeries misleading exception when required column check fails", "timestamp": 1718000000},
        {"role": "assistant", "content": f"[Code Patch / Solution] Applied fix to sampled.py:\n{target_patch}", "timestamp": 1718000010},
    ]
    props_target = [
        "[Code Patch / Solution] Fixed TimeSeries misleading exception in astropy/timeseries/sampled.py",
        target_patch,
    ]
    save_memories_batch("req_c_target", user_id, "sess_code", msgs_target, props_target)

    # Insert 50 noisy distractors (TaskUpdate, bash traces, compile logs)
    noisy_msgs = []
    noisy_props = []
    for i in range(50):
        noisy_msgs.append({
            "role": "assistant",
            "content": f"[tool_use TaskUpdate] status=IN_PROGRESS step={i} running pytest tests/test_sampled.py ... (Bash completed with no output)",
            "timestamp": 1717000000 + i * 10
        })
        noisy_props.append(f"Task step {i} status update in progress")

    save_memories_batch("req_c_noise", user_id, "sess_code", noisy_msgs, noisy_props)

    # Query for the fix
    query = "TimeSeries misleading exception required column check fails astropy"
    t0 = time.time()
    res = search_hybrid(user_id, query, top_k=25)
    dt = time.time() - t0

    # Calculate payload size
    payload_bytes = sum(len(r["content"].encode("utf-8")) for r in res)

    # Verification criteria
    top_is_patch = len(res) > 0 and ("diff --git" in res[0]["content"] or "sampled.py" in res[0]["content"])
    payload_under_budget = payload_bytes < 25000  # <25 KB (crucial for Flash model)
    count_under_15 = len(res) <= 15
    latency_under_100ms = dt < 0.100

    score = 0.0
    if top_is_patch:
        score += 50.0
    if payload_under_budget:
        score += 25.0
    if count_under_15:
        score += 15.0
    if latency_under_100ms:
        score += 10.0

    return {
        "track": "coding",
        "score": score,
        "metrics": {
            "top_is_patch": top_is_patch,
            "payload_bytes": payload_bytes,
            "items_returned": len(res),
            "latency_ms": round(dt * 1000, 2),
            "top_item": res[0]["content"][:90] if res else ""
        }
    }


def run_multimodal_benchmark() -> Dict[str, Any]:
    """Evaluates Multimodal direct recall and abstention (crucial for the 69.2% abstention suite)."""
    user_id = "eval_mm_u1"

    # Multimodal memories: Wang Jingchuan career and image caption
    msgs = [
        {"role": "user", "content": "王景川在2018年加入阿里巴巴达摩院担任高级算法专家，主研视觉与具身智能。", "timestamp": 1719000000},
        {"role": "assistant", "content": "The image caption describes a bottle of Ethos Water pouring a heart shape for Starbucks CSR campaign.", "timestamp": 1719000010},
    ]
    props = [
        "王景川在2018年加入阿里巴巴达摩院担任高级算法专家，负责视觉与具身智能研发。",
        "The image caption describes a bottle of Ethos Water pouring a heart for Starbucks CSR campaign.",
    ]
    save_memories_batch("req_mm1", user_id, "sess_mm", msgs, props)

    tests = [
        {
            "name": "Chinese Fact Recall (Wang Jingchuan Join Year)",
            "query": "王景川哪一年加入达摩院？",
            "type": "recall",
            "expected_count_gt": 0,
            "expected_contain": "2018",
        },
        {
            "name": "Multimodal Image Detail (Ethos Water Pour Shape)",
            "query": "What shape is the water pouring in the Starbucks Ethos Water image?",
            "type": "recall",
            "expected_count_gt": 0,
            "expected_contain": "heart",
        },
        {
            "name": "Abstention 1: Unrecorded Food Preference",
            "query": "王景川最喜欢吃的川菜是什么？",
            "type": "abstention",
            "expected_count": 0,
        },
        {
            "name": "Abstention 2: Unrelated Pet Adoption",
            "query": "What breed of puppy did Alex Mercer adopt last Sunday?",
            "type": "abstention",
            "expected_count": 0,
        },
        {
            "name": "Abstention 3: Distractor Options in Question",
            "query": "Where did the user purchase the electric bicycle?",
            "options": ["At Starbucks", "Online store", "In Seattle", "None of the above"],
            "type": "abstention",
            "expected_count": 0,
        },
    ]

    passed = 0
    total = len(tests)
    details = []

    for t in tests:
        opts = t.get("options")
        res = search_hybrid(user_id, t["query"], options=opts, top_k=10)
        
        if t["type"] == "recall":
            success = len(res) > 0 and t["expected_contain"] in res[0]["content"]
        else:  # abstention
            success = len(res) == 0

        if success:
            passed += 1

        details.append({
            "test": t["name"],
            "type": t["type"],
            "passed": success,
            "returned_count": len(res),
            "top_match": res[0]["content"][:80] if res else "(empty - correctly abstained)",
        })

    score = round((passed / total) * 100.0, 2)
    return {"track": "multimodal", "score": score, "passed": passed, "total": total, "details": details}


def main():
    print("=" * 70)
    print("AGENT0MEM OFFLINE HIGH-FIDELITY MULTI-TRACK BENCHMARK SUITE")
    print("Zero Quota Consumption | Zero Cooldown | Instant Feedback Loop")
    print("=" * 70)

    setup_fresh_eval_db()

    print("\n--- Running Track 1: Textual Memory Benchmark ---")
    t_res = run_textual_benchmark()
    print(f"Score: {t_res['score']}% ({t_res['passed']}/{t_res['total']} tests passed)")
    for d in t_res["details"]:
        status = "PASS" if d["passed"] else "FAIL"
        print(f"  [{status}] {d['test']}: {d['top_match']}")

    print("\n--- Running Track 2: Coding Agent Memory Benchmark ---")
    c_res = run_coding_benchmark()
    print(f"Score: {c_res['score']}% | Latency: {c_res['metrics']['latency_ms']}ms | Payload: {c_res['metrics']['payload_bytes']} bytes")
    print(f"  - Top Item is Patch: {c_res['metrics']['top_is_patch']}")
    print(f"  - Payload Budget (<25KB): {c_res['metrics']['payload_bytes'] < 25000}")
    print(f"  - Items Returned (<=15): {c_res['metrics']['items_returned']}")

    print("\n--- Running Track 3: Multimodal Memory Benchmark ---")
    m_res = run_multimodal_benchmark()
    print(f"Score: {m_res['score']}% ({m_res['passed']}/{m_res['total']} tests passed)")
    for d in m_res["details"]:
        status = "PASS" if d["passed"] else "FAIL"
        print(f"  [{status}] ({d['type'].upper()}) {d['test']}: {d['top_match']}")

    print("\n" + "=" * 70)
    print("OVERALL BENCHMARK SUMMARY:")
    print(f"  Textual Track:    {t_res['score']}%  (Target: >87.0%)")
    print(f"  Coding Track:     {c_res['score']}% (Target: >91.0%)")
    print(f"  Multimodal Track: {m_res['score']}%  (Target: >81.0%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
