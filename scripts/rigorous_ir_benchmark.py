#!/usr/bin/env python3
"""
Agent0Mem Rigorous Multi-Track Information Retrieval & Memory Governance Benchmark.

Evaluates memory retrieval performance against key empirical dimensions identified in official AML benchmarks:
1. Retrieval Ranking: MRR, Recall@1, Recall@5 across multi-session dialogues.
2. Evidence Governance & Abstention: Precision, Recall, F1 on unanswerable/out-of-domain queries (69.2% of Multimodal track).
3. Payload & Token Budget: Response payload size distribution (<15 KB safe ceiling for downstream coding agents).
4. Temporal & Conflict Inversion: Recency boost and state transition resolution.
5. Constraint & Negative Preference Filtering: Filtering out forbidden/disliked entities.
"""

import os
import sys
import time
import json
import math
from typing import Dict, List, Any, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import search_hybrid, init_db, save_memories_batch
from app import config

BENCHMARK_DB_PATH = "/tmp/agent0mem_rigorous_eval.sqlite3"
config.DB_PATH = BENCHMARK_DB_PATH


def setup_fresh_eval_db():
    if os.path.exists(BENCHMARK_DB_PATH):
        try:
            os.remove(BENCHMARK_DB_PATH)
        except Exception:
            pass
    init_db()


def compute_metrics(ranks: List[int], total_queries: int) -> Dict[str, float]:
    """Computes MRR, Recall@1, and Recall@5 given 1-based ranks of gold answers (or 0 if not found)."""
    if total_queries == 0:
        return {"mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0}
    
    mrr = sum(1.0 / r for r in ranks if r > 0) / total_queries
    r1 = sum(1 for r in ranks if r == 1) / total_queries
    r5 = sum(1 for r in ranks if 1 <= r <= 5) / total_queries
    return {
        "mrr": round(mrr, 4),
        "recall@1": round(r1, 4),
        "recall@5": round(r5, 4),
    }


def compute_binary_f1(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Computes Precision, Recall, and F1 for binary classification (e.g. Abstention)."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
    }


def run_ir_benchmark() -> Dict[str, Any]:
    print("=" * 75)
    print("RUNNING AGENT0MEM RIGOROUS IR & GOVERNANCE BENCHMARK")
    print("=" * 75)
    
    setup_fresh_eval_db()
    
    # -------------------------------------------------------------------------
    # Track 1: Textual Memory Corpus (Conversations, Updates, Temporal, Distractors)
    # -------------------------------------------------------------------------
    print("\n[Track 1: Textual Memory Dataset Ingestion]")
    textual_users = ["user_text_1", "user_text_2", "user_text_3"]
    
    # User 1: Career transitions and temporal events across 4 sessions
    save_memories_batch("t1_s1", "user_text_1", "s1", 
        [{"role": "user", "content": "I joined Google Seattle as a software engineer in July 2021.", "timestamp": 1625097600}],
        ["User worked as a software engineer at Google Seattle starting July 2021."]
    )
    save_memories_batch("t1_s2", "user_text_1", "s2", 
        [{"role": "user", "content": "Update: In August 2023, I transferred to Google Zurich to lead the infrastructure team.", "timestamp": 1690848000}],
        ["[Superseded] User previously worked at Google Seattle.", "User transferred to Google Zurich in August 2023 as an infrastructure tech lead."]
    )
    save_memories_batch("t1_s3", "user_text_1", "s3", 
        [{"role": "user", "content": "Big news: I left Google in January 2025 and founded an AI agents startup in San Francisco.", "timestamp": 1735689600}],
        ["[Current State] User left Google and founded an AI agents startup in San Francisco in January 2025."]
    )
    # Add 15 realistic distractor memories for User 1
    for i in range(15):
        save_memories_batch(f"t1_dist_{i}", "user_text_1", f"s_dist_{i}",
            [{"role": "user", "content": f"Random chat note {i}: discussed weather in London, bought coffee #{i}, and read an article about distributed consensus.", "timestamp": 1650000000 + i * 100000}],
            [f"Distractor note {i}: conversation about London weather, coffee #{i}, and distributed consensus."]
        )

    # User 2: Dietary, Medical, and Negative constraints
    save_memories_batch("t2_s1", "user_text_2", "s1",
        [{"role": "user", "content": "I am strictly allergic to peanuts and avoid shellfish. I follow a Mediterranean diet.", "timestamp": 1700000000}],
        ["[Strict Medical Constraint] User is allergic to peanuts.", "[Dietary Preference] User avoids shellfish and follows a Mediterranean diet."]
    )
    save_memories_batch("t2_s2", "user_text_2", "s2",
        [{"role": "user", "content": "My favorite vacation was sailing in Greece in September 2022.", "timestamp": 1705000000}],
        ["User's favorite vacation was sailing in Greece in September 2022."]
    )
    # Add distractor memories with conflicting words
    save_memories_batch("t2_dist_1", "user_text_2", "s_dist_1",
        [{"role": "user", "content": "My colleague had peanut butter toast and lobster bisque for lunch.", "timestamp": 1706000000}],
        ["Colleague of user had peanut butter toast and lobster bisque."]
    )

    # User 3: Complex multi-session project planning
    save_memories_batch("t3_s1", "user_text_3", "s1",
        [{"role": "user", "content": "Project Apollo is budgeted at $500,000 and targeted for Q2 2026 launch.", "timestamp": 1710000000}],
        ["Project Apollo budget is $500,000 and target launch date is Q2 2026."]
    )
    save_memories_batch("t3_s2", "user_text_3", "s2",
        [{"role": "user", "content": "Budget revision: Project Apollo budget increased to $750,000 due to GPU compute costs.", "timestamp": 1715000000}],
        ["[Update] Project Apollo budget increased to $750,000 due to GPU compute costs."]
    )

    # Evaluation Queries for Textual Track
    textual_eval_cases = [
        {
            "user_id": "user_text_1",
            "query": "Where does the user currently work and what city are they in?",
            "gold_substring": "San Francisco",
            "negative_check": "Seattle",
        },
        {
            "user_id": "user_text_1",
            "query": "What was the user's role when they transferred to Google Zurich?",
            "gold_substring": "Zurich",
            "negative_check": None,
        },
        {
            "user_id": "user_text_1",
            "query": "When did the user first join Google Seattle as a software engineer?",
            "gold_substring": "Seattle",
            "negative_check": None,
        },
        {
            "user_id": "user_text_2",
            "query": "Suggest dinner recommendations for user diet and allergy constraints",
            "gold_substring": "Mediterranean",
            "negative_check": "peanut butter toast",
        },
        {
            "user_id": "user_text_2",
            "query": "Where did the user have their favorite sailing vacation?",
            "gold_substring": "Greece",
            "negative_check": None,
        },
        {
            "user_id": "user_text_3",
            "query": "What is the latest revised budget for Project Apollo?",
            "gold_substring": "$750,000",
            "negative_check": None,
        },
    ]

    textual_ranks = []
    textual_negative_pass = 0
    textual_negative_total = 0
    latencies_ms = []

    for case in textual_eval_cases:
        t0 = time.perf_counter()
        candidates = search_hybrid(case["user_id"], case["query"], top_k=15)
        dt = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt)

        # Find gold rank
        rank = 0
        for i, c in enumerate(candidates):
            if case["gold_substring"].lower() in c["content"].lower():
                rank = i + 1
                break
        textual_ranks.append(rank)

        # Check negative constraint
        if case.get("negative_check"):
            textual_negative_total += 1
            if len(candidates) > 0 and case["negative_check"].lower() not in candidates[0]["content"].lower():
                textual_negative_pass += 1

    textual_ir_metrics = compute_metrics(textual_ranks, len(textual_eval_cases))
    textual_negative_rate = (textual_negative_pass / textual_negative_total) if textual_negative_total > 0 else 1.0

    print(f"  - MRR:       {textual_ir_metrics['mrr']:.4f}")
    print(f"  - Recall@1:  {textual_ir_metrics['recall@1']:.4f}")
    print(f"  - Recall@5:  {textual_ir_metrics['recall@5']:.4f}")
    print(f"  - Neg Avoid: {textual_negative_rate:.4f} ({textual_negative_pass}/{textual_negative_total})")
    print(f"  - Latency:   p50={sorted(latencies_ms)[len(latencies_ms)//2]:.1f}ms, max={max(latencies_ms):.1f}ms")

    # -------------------------------------------------------------------------
    # Track 2: Coding Memory Corpus (SWE-bench Distractors, Patches, Payload Bloat)
    # -------------------------------------------------------------------------
    print("\n[Track 2: Coding Memory Dataset & Payload Stress Test]")
    coding_user = "user_coding_bench"
    
    # Add real patch solutions and test logs
    patch_content = """diff --git a/sklearn/linear_model/_coordinate_descent.py b/sklearn/linear_model/_coordinate_descent.py
--- a/sklearn/linear_model/_coordinate_descent.py
+++ b/sklearn/linear_model/_coordinate_descent.py
@@ -1025,8 +1025,12 @@ def elastic_net_path(X, y, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None):
-    if check_input:
-        X = check_array(X, 'csc', dtype=[np.float64, np.float32], order='F')
+    if check_input:
+        X = check_array(X, ['csc', 'csr'], dtype=[np.float64, np.float32], order='F')
+        if y is not None:
+            y = check_array(y, dtype=X.dtype.type, ensure_2d=False)"""

    save_memories_batch("c_gold_patch", coding_user, "sess_gold",
        [{"role": "user", "content": f"Fix TypeError in elastic_net_path when y dtype does not match X dtype.\n{patch_content}", "timestamp": 1720000000}],
        [f"[Code Patch / Solution] elastic_net_path dtype check fix:\n{patch_content}"]
    )

    # Ingest 40 realistic noisy traceback and compiler log distractors
    for i in range(40):
        log_content = f"""Build Traceback #{i}:
Traceback (most recent call last):
  File "test_runner.py", line {100 + i}, in execute
    result = run_benchmark_matrix(mode='test_{i}', seed={i*17})
  File "sklearn/utils/estimator_checks.py", line 452, in check_estimator
    check_parameters_default_constructible(name, Estimator)
AssertionError: Default parameter mismatch in mock_estimator_{i} at line {200+i}."""
        save_memories_batch(f"c_dist_{i}", coding_user, f"sess_dist_{i}",
            [{"role": "user", "content": log_content, "timestamp": 1720000000 + i * 100}],
            [f"[Compiler / Test Log #{i}] {log_content[:200]}..."]
        )

    # Coding Evaluation Queries
    coding_eval_cases = [
        {"query": "elastic_net_path TypeError dtype mismatch between X and y", "is_relevant": True, "gold": "elastic_net_path"},
        {"query": "fix TypeError in elastic_net_path coordinate descent", "is_relevant": True, "gold": "elastic_net_path"},
        {"query": "where is check_parameters_default_constructible defined?", "is_relevant": True, "gold": "estimator_checks"},
        {"query": "pytest test_unrelated_feature_not_in_repo.py", "is_relevant": False, "gold": None},
    ]

    coding_payload_sizes = []
    coding_candidate_counts = []
    coding_ranks = []
    coding_abstain_tp = 0
    coding_abstain_fp = 0
    coding_abstain_tn = 0
    coding_abstain_fn = 0

    for case in coding_eval_cases:
        candidates = search_hybrid(coding_user, case["query"], top_k=15)
        # Serialize as API response to measure byte size
        serialized_bytes = len(json.dumps([{"id": c["id"], "content": c["content"], "score": c["score"]} for c in candidates]))
        coding_payload_sizes.append(serialized_bytes)
        coding_candidate_counts.append(len(candidates))

        if case["is_relevant"]:
            rank = 0
            for i, c in enumerate(candidates):
                if case["gold"].lower() in c["content"].lower():
                    rank = i + 1
                    break
            coding_ranks.append(rank)
            if len(candidates) > 0:
                coding_abstain_tn += 1
            else:
                coding_abstain_fn += 1
        else:
            # Irrelevant query should return 0 candidates
            if len(candidates) == 0:
                coding_abstain_tp += 1
            else:
                coding_abstain_fp += 1

    coding_ir_metrics = compute_metrics(coding_ranks, 3)
    max_payload_kb = max(coding_payload_sizes) / 1024
    mean_payload_kb = (sum(coding_payload_sizes) / len(coding_payload_sizes)) / 1024

    print(f"  - MRR:         {coding_ir_metrics['mrr']:.4f}")
    print(f"  - Recall@1:    {coding_ir_metrics['recall@1']:.4f}")
    print(f"  - Max Payload: {max_payload_kb:.2f} KB (Target: <15.0 KB)")
    print(f"  - Mean Payload:{mean_payload_kb:.2f} KB")
    print(f"  - Max Items:   {max(coding_candidate_counts)} items (Target: <=15)")

    # -------------------------------------------------------------------------
    # Track 3: Multimodal Memory Corpus (Evidence Governance & Rejection/Abstention)
    # -------------------------------------------------------------------------
    print("\n[Track 3: Multimodal Evidence Governance & Abstention Benchmark]")
    mm_user = "user_multimodal_bench"

    # Ingest distinct multimodal interaction scenes
    mm_scenes = [
        ("mm_1", "Photograph of a receipt from Blue Bottle Coffee in Tokyo dated 2024-03-15 for 1,200 JPY.", "Blue Bottle Coffee Tokyo receipt 1200 JPY on 2024-03-15"),
        ("mm_2", "Floor plan diagram of a 3-bedroom apartment on 4th floor showing balcony facing south.", "Floor plan diagram: 3 bedrooms, 4th floor, south facing balcony"),
        ("mm_3", "Chart comparing battery life of Drone Model X (42 mins) versus Drone Model Y (35 mins).", "Battery comparison chart: Drone X has 42 mins, Drone Y has 35 mins"),
        ("mm_4", "Screenshot of software error dialog: 'Error 0x80070005 Access is denied in C:\\Windows\\System32'.", "Windows error dialog 0x80070005 access denied in System32"),
    ]

    for req_id, msg_content, prop_content in mm_scenes:
        save_memories_batch(req_id, mm_user, "sess_mm",
            [{"role": "user", "content": msg_content, "timestamp": 1710000000}],
            [prop_content]
        )

    # Multimodal test suite: 4 in-domain answerable queries + 6 out-of-domain/abstention queries (60% abstention ratio)
    mm_eval_cases = [
        # In-domain (Positive)
        {"query": "How much was the receipt from Blue Bottle Coffee in Tokyo?", "should_recall": True, "gold": "1,200 JPY"},
        {"query": "Which direction does the balcony face in the apartment floor plan?", "should_recall": True, "gold": "south"},
        {"query": "What is the battery flight duration of Drone Model X?", "should_recall": True, "gold": "42 mins"},
        {"query": "What error code appeared in the System32 dialog screenshot?", "should_recall": True, "gold": "0x80070005"},
        # Out-of-Domain / Abstention queries (Negative / Evidence Governance)
        {"query": "What was the brand of car parked in front of the supermarket?", "should_recall": False, "gold": None},
        {"query": "Show me the flight ticket confirmation for Paris from May 2023.", "should_recall": False, "gold": None},
        {"query": "What did the doctor prescribe in the medical diagnosis note?", "should_recall": False, "gold": None},
        {"query": "What is the WiFi password for the hotel lobby in Berlin?", "should_recall": False, "gold": None},
        {"query": "Where did the user store the backup cryptographic private key?", "should_recall": False, "gold": None},
        {"query": "What was the score of the Champions League final match?", "should_recall": False, "gold": None},
    ]

    mm_tp, mm_fp, mm_fn, mm_tn = 0, 0, 0, 0
    mm_ranks = []

    for case in mm_eval_cases:
        candidates = search_hybrid(mm_user, case["query"], top_k=15)
        has_results = len(candidates) > 0

        if case["should_recall"]:
            if has_results:
                # True positive (recalled memory when expected)
                mm_tp += 1
                rank = 0
                for i, c in enumerate(candidates):
                    if case["gold"].lower() in c["content"].lower():
                        rank = i + 1
                        break
                mm_ranks.append(rank)
            else:
                # False negative (empty when should recall)
                mm_fn += 1
                mm_ranks.append(0)
        else:
            if not has_results:
                # True negative (successfully abstained / rejected unanswerable query)
                mm_tn += 1
            else:
                # False positive (hallucinated / retrieved irrelevant memory for unanswerable query)
                mm_fp += 1

    mm_ranking_metrics = compute_metrics(mm_ranks, 4)
    # Abstention classification: Positive class = Successfully rejecting / abstaining
    abstention_f1_metrics = compute_binary_f1(tp=mm_tn, fp=mm_fn, fn=mm_fp, tn=mm_tp)

    print(f"  - In-Domain Recall@1:  {mm_ranking_metrics['recall@1']:.4f}")
    print(f"  - In-Domain MRR:       {mm_ranking_metrics['mrr']:.4f}")
    print(f"  - Abstention Prec:     {abstention_f1_metrics['precision']:.4f}")
    print(f"  - Abstention Recall:   {abstention_f1_metrics['recall']:.4f} ({mm_tn}/6 abstained)")
    print(f"  - Abstention F1:       {abstention_f1_metrics['f1']:.4f}")
    print(f"  - False Positive Rec:  {mm_fp}/6 irrelevant queries triggered recall")

    # -------------------------------------------------------------------------
    # Overall Benchmark Synthesis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY AND STATISTICAL CONFIDENCE ASSESSMENT")
    print("=" * 75)
    summary = {
        "textual": {
            "mrr": textual_ir_metrics["mrr"],
            "recall@1": textual_ir_metrics["recall@1"],
            "negative_avoidance_rate": textual_negative_rate,
            "latency_p50_ms": round(sorted(latencies_ms)[len(latencies_ms)//2], 2),
        },
        "coding": {
            "mrr": coding_ir_metrics["mrr"],
            "recall@1": coding_ir_metrics["recall@1"],
            "max_payload_kb": round(max_payload_kb, 2),
            "payload_budget_satisfied": max_payload_kb <= 15.0,
            "candidate_cap_satisfied": max(coding_candidate_counts) <= 15,
        },
        "multimodal": {
            "in_domain_recall@1": mm_ranking_metrics["recall@1"],
            "abstention_precision": abstention_f1_metrics["precision"],
            "abstention_recall": abstention_f1_metrics["recall"],
            "abstention_f1": abstention_f1_metrics["f1"],
            "zero_hallucination_abstention": mm_fp == 0,
        }
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_ir_benchmark()
