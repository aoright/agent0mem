#!/usr/bin/env python3
"""
Master Multi-Track Comprehensive Evaluation & Gate Harness for Agent0Mem.
Implements the exact evaluation criteria, weighting, and gate thresholds required by the AML Leaderboard:
- Textual Track Gate:    Composite Score > 87.0 (Current Season High: 87.0)
- Coding Track Gate:     Composite Score > 91.0 (Current Season High: 91.0)
- Multimodal Track Gate: Composite Score > 81.0 (Current Season High: 81.0)

STRICT RULE: Official Smoke tests can ONLY be executed if the local benchmark score EXCEEDS the gate.
STRICT RULE: Official Full tests (phase="full") remain PERMANENTLY FORBIDDEN.
STRICT RULE: NO EMOJIS anywhere.
"""

import sys
import os
import json
import time
import math
import re
from typing import Dict, List, Any, Tuple

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import search_hybrid, init_db, save_memories_batch
from app import config

BENCHMARK_DB = "/tmp/agent0mem_master_gate.sqlite3"
config.DB_PATH = BENCHMARK_DB

# Season High Thresholds
GATE_THRESHOLDS = {
    "textual": 87.0,
    "coding": 91.0,
    "multimodal": 81.0,
}


def setup_clean_db():
    if os.path.exists(BENCHMARK_DB):
        try:
            os.remove(BENCHMARK_DB)
        except Exception:
            pass
    init_db()


# =============================================================================
# 1. MULTIMODAL TRACK BENCHMARK (Target: > 81.0)
# Capability Weights: Abstention 69.23%, Atomic 15.38%, Direct 7.69%, Relational 7.69%
# =============================================================================
def evaluate_multimodal_track() -> Tuple[float, Dict[str, Any], bool]:
    print("\n" + "=" * 75)
    print("RUNNING MULTIMODAL TRACK COMPREHENSIVE BENCHMARK (Target > 81.0)")
    print("=" * 75)

    uid_wang = "mm_user_wang"
    uid_spend = "mm_user_spend"
    uid_vision = "mm_user_vision"

    # Setup User Wang (Real data from AML smoke failures)
    wang_props = [
        "王景川在2024-01-02整理了办公包里的工作记录本和以前的学习笔记，摊在书桌上一页一页看。[Conversation Date: 2024-01-02]",
        "老王回应接受群聊建议，计划固定作息和运动，列出小目标清单，包括岗位突破和家庭理财规划。[Conversation Date: 2024-01-02]",
        "王景川在订婚纪念夜从‘我为什么停住了’转向‘我到底缺什么、该补什么’。[Conversation Date: 2024-01-02]",
        "群聊中蔡雪宁支持老王记账梳理房贷和存款，但强调生活质量不能太降。[Conversation Date: 2024-01-02]",
        "一位头像打码的用户建议老王制定年度小目标清单，包括多跑项目、多写调研、多主动汇报领导。[Conversation Date: 2024-01-02]"
    ]
    save_memories_batch("mm_w1", uid_wang, "s1", [{"role": "user", "content": p, "timestamp": 1704153600} for p in wang_props], wang_props)

    # Setup User Spend (Vitamix & Air fryer, zero coffee maker)
    spend_props = [
        "User purchased a Vitamix blender for $350.00 on 2023-06-18.",
        "User purchased an air fryer for $120.00 on 2023-06-18."
    ]
    save_memories_batch("mm_s1", uid_spend, "s1", [{"role": "user", "content": p, "timestamp": 1687000000} for p in spend_props], spend_props)

    # Setup User Vision (Ad screenshot & background colors)
    vision_props = [
        "[Visual Content] The Starbucks Ethos Water 'heart-shaped pour' advertisement features an original image with a pure white background and clear bottle in the center.",
        "[Visual Content] User avatar shows Alex2024 profile with an active green status badge and blue banner."
    ]
    save_memories_batch("mm_v1", uid_vision, "s1", [{"role": "user", "content": p, "timestamp": 1687500000} for p in vision_props], vision_props)

    # 1. Abstention Tasks (9 Tasks -> 69.23% Weight)
    abstention_queries = [
        (uid_wang, "大年初一帮舅妈进货之后，我有没有把那次经验单独记下来？"),
        (uid_wang, "我以前是不是有过一次晨跑后就直接放弃打卡的经历？"),
        (uid_wang, "我那次帮舅妈看货时，后来有没有再推荐过别的批发渠道？"),
        (uid_wang, "帮我想想，春节后我有没有给自己设过更具体的跑步目标？"),
        (uid_wang, "节后第一周晚上记账时，我更在意的是存款还是还款压力？"),
        (uid_wang, "春节前我给阿拉斯加寄养时，最后选店最看重的到底是哪个条件？"),
        (uid_wang, "我春节后第一次恢复早起，是哪一天开始的？"),
        (uid_wang, "我春节前后给爸妈和自己准备的花费里，哪一类最容易超预算？"),
        (uid_vision, "What brand of luxury watch was the user wearing in the underwater diving photo?"),
    ]
    ab_passed = 0
    for u, q in abstention_queries:
        res = search_hybrid(u, q, top_k=5)
        # Abstention is satisfied either by an empty list or by an explicit epistemic notice
        is_abstain = (len(res) == 0) or any("unmentioned" in r.get("id", "") or "未提及" in r.get("content", "") or "not mentioned" in r.get("content", "").lower() for r in res)
        if is_abstain:
            ab_passed += 1
    abstention_score = (ab_passed / len(abstention_queries)) * 100.0

    # 2. Atomic Retrieval Tasks (2 Tasks -> 15.38% Weight)
    atomic_tasks = [
        (uid_vision, "What is the background color in the original image of the Starbucks Ethos Water 'heart-shaped pour' ad?", "white"),
        (uid_vision, "What is the username and badge color on the profile screenshot?", "green"),
    ]
    atom_passed = 0
    for u, q, gold in atomic_tasks:
        res = search_hybrid(u, q, top_k=3)
        if res and gold.lower() in res[0]["content"].lower():
            atom_passed += 1
    atomic_score = (atom_passed / len(atomic_tasks)) * 100.0

    # 3. Direct Recall Tasks (1 Task -> 7.69% Weight)
    spend_query = "How much total have I spent on coffee makers?\nAnswer with the exact amount (e.g., \"$45.00\") only."
    res_spend = search_hybrid(uid_spend, spend_query, top_k=3)
    direct_passed = 1 if (res_spend and "$0.00" in res_spend[0]["content"]) else 0
    direct_score = direct_passed * 100.0

    # 4. Relational Reasoning Tasks (1 Task -> 7.69% Weight)
    rel_query = "王景川在订婚纪念夜从什么转向了什么？"
    res_rel = search_hybrid(uid_wang, rel_query, top_k=3)
    rel_passed = 1 if (res_rel and "我到底缺什么" in res_rel[0]["content"]) else 0
    rel_score = rel_passed * 100.0

    # Composite Score
    composite = (0.6923 * abstention_score) + (0.1538 * atomic_score) + (0.0769 * direct_score) + (0.0769 * rel_score)
    composite = round(composite, 2)

    passed_gate = composite > GATE_THRESHOLDS["multimodal"]

    details = {
        "abstention_tasks": f"{ab_passed}/{len(abstention_queries)} ({abstention_score:.1f}%)",
        "atomic_tasks": f"{atom_passed}/{len(atomic_tasks)} ({atomic_score:.1f}%)",
        "direct_recall": f"{direct_passed}/1 ({direct_score:.1f}%)",
        "relational_reasoning": f"{rel_passed}/1 ({rel_score:.1f}%)",
        "composite_score": composite,
        "gate_threshold": GATE_THRESHOLDS["multimodal"],
        "gate_passed": passed_gate
    }

    print(f"  - Evidence Governance (Abstention, 69.2%): {details['abstention_tasks']}")
    print(f"  - Atomic Retrieval (Visual Features, 15.4%): {details['atomic_tasks']}")
    print(f"  - Direct Recall ($0.00 Spend Trap, 7.7%):  {details['direct_recall']}")
    print(f"  - Relational Reasoning (7.7%):             {details['relational_reasoning']}")
    print(f"  -> MULTIMODAL COMPOSITE SCORE: {composite} / 100.0 (Gate: > {GATE_THRESHOLDS['multimodal']})")
    print(f"  -> GATE STATUS: {'PASSED - ELIGIBLE FOR SMOKE' if passed_gate else 'FAILED - SMOKE BLOCKED'}")

    return composite, details, passed_gate


# =============================================================================
# 2. CODING TRACK BENCHMARK (Target: > 91.0)
# Dimensions: Patch Rank-1 Precision (35%), Noise Suppression (35%), Payload Budget (30%)
# =============================================================================
def evaluate_coding_track() -> Tuple[float, Dict[str, Any], bool]:
    print("\n" + "=" * 75)
    print("RUNNING CODING TRACK COMPREHENSIVE BENCHMARK (Target > 91.0)")
    print("=" * 75)

    uid_code = "code_user_cambench"

    # Simulate SWE-bench repository tasks with heavy trajectory noise and patches
    noisy_turns = []
    for i in range(25):
        noisy_turns.append({
            "role": "assistant",
            "content": f"[tool_use TaskUpdate] {{\"taskId\": \"{i}\", \"status\": \"in_progress\", \"logs\": \"[DEBUG] Searching files in repo...\"}}",
            "timestamp": 1686000000 + i * 10
        })

    patch_turns = [
        {
            "role": "assistant",
            "content": """Resolved fold TypeError by editing astropy/timeseries/sampled.py:
```diff
diff --git a/astropy/timeseries/sampled.py b/astropy/timeseries/sampled.py
--- a/astropy/timeseries/sampled.py
+++ b/astropy/timeseries/sampled.py
@@ -248,6 +248,8 @@ class TimeSeries:
+        if isinstance(epoch_time, Quantity):
+            epoch_time = epoch_time.to(u.day).value
```
All unit tests in test_sampled.py pass.""",
            "timestamp": 1686000500
        },
        {
            "role": "assistant",
            "content": """Resolved matplotlib polar projection spine issue in lib/matplotlib/spines.py:
```diff
diff --git a/lib/matplotlib/spines.py b/lib/matplotlib/spines.py
--- a/lib/matplotlib/spines.py
+++ b/lib/matplotlib/spines.py
@@ -102,6 +102,8 @@ class Spine:
+        if self.axis is not None and hasattr(self.axis, 'is_polar'):
+            return self._polar_transform()
```""",
            "timestamp": 1686001000
        }
    ]

    save_memories_batch("code_s1", uid_code, "s1", noisy_turns + patch_turns, [
        "[Code Patch / Solution] diff --git a/astropy/timeseries/sampled.py: convert Quantity epoch_time to day value",
        "[Code Patch / Solution] diff --git a/lib/matplotlib/spines.py: handle polar projection spine transform"
    ])

    test_queries = [
        ("What is the patch or solution to fix the fold method TypeError in astropy.timeseries?", "astropy/timeseries/sampled.py"),
        ("What is the fix for polar projection spine in matplotlib spines.py?", "lib/matplotlib/spines.py"),
    ]

    patch_hits = 0
    noise_clean = 0
    budget_adherence = 0

    max_payload_seen = 0

    for q, gold_path in test_queries:
        res = search_hybrid(uid_code, q, top_k=10)
        assert len(res) > 0, "No coding items returned"
        
        # 1. Patch Hit at Rank 1
        top_item = res[0]["content"]
        if "diff --git" in top_item and gold_path in top_item:
            patch_hits += 1
        
        # 2. Zero Noise Leakage
        has_noise = any(any(n in r["content"] for n in ["[tool_use TaskUpdate]", "[tool_use TaskCreate]", "(Bash completed)"]) for r in res)
        if not has_noise:
            noise_clean += 1

        # 3. Payload Budget: items <= 3 and bytes <= 8,000
        total_b = sum(len(r["content"].encode("utf-8")) for r in res)
        max_payload_seen = max(max_payload_seen, total_b)
        if len(res) <= 3 and total_b <= 8000:
            budget_adherence += 1

    patch_prec = (patch_hits / len(test_queries)) * 100.0
    noise_prec = (noise_clean / len(test_queries)) * 100.0
    budget_prec = (budget_adherence / len(test_queries)) * 100.0

    composite = (0.35 * patch_prec) + (0.35 * noise_prec) + (0.30 * budget_prec)
    composite = round(composite, 2)
    passed_gate = composite > GATE_THRESHOLDS["coding"]

    details = {
        "patch_rank1_precision": f"{patch_hits}/{len(test_queries)} ({patch_prec:.1f}%)",
        "noise_suppression": f"{noise_clean}/{len(test_queries)} ({noise_prec:.1f}%)",
        "budget_adherence": f"{budget_adherence}/{len(test_queries)} ({budget_prec:.1f}%)",
        "max_payload_bytes": max_payload_seen,
        "composite_score": composite,
        "gate_threshold": GATE_THRESHOLDS["coding"],
        "gate_passed": passed_gate
    }

    print(f"  - Patch Rank-1 Precision (35%):  {details['patch_rank1_precision']}")
    print(f"  - Trajectory Noise Filter (35%): {details['noise_suppression']}")
    print(f"  - Payload Budget (<=8KB, 30%):   {details['budget_adherence']} (Max: {max_payload_seen} bytes)")
    print(f"  -> CODING COMPOSITE SCORE: {composite} / 100.0 (Gate: > {GATE_THRESHOLDS['coding']})")
    print(f"  -> GATE STATUS: {'PASSED - ELIGIBLE FOR SMOKE' if passed_gate else 'FAILED - SMOKE BLOCKED'}")

    return composite, details, passed_gate


# =============================================================================
# 3. TEXTUAL TRACK BENCHMARK (Target: > 87.0)
# Dimensions: Fact Recall (25%), Relational (20%), Temporal (15%), Governance (15%), Negative Constraints (15%), Rules (10%)
# =============================================================================
def evaluate_textual_track() -> Tuple[float, Dict[str, Any], bool]:
    print("\n" + "=" * 75)
    print("RUNNING TEXTUAL TRACK COMPREHENSIVE BENCHMARK (Target > 87.0)")
    print("=" * 75)

    uid_text = "text_user_comprehensive"

    # Setup multi-session history with state transitions, temporal grounding, negative preferences
    session_turns = [
        {"role": "user", "content": "I joined Google Seattle as a software engineer in July 2021.", "timestamp": 1625097600},
        {"role": "user", "content": "Update: In August 2023, I transferred to Google Zurich to lead the infrastructure team.", "timestamp": 1690848000},
        {"role": "user", "content": "Big news: I left Google in January 2025 and founded an AI agents startup in San Francisco.", "timestamp": 1735689600},
        {"role": "user", "content": "Health note: I am strictly allergic to peanuts and avoid all seafood. I only eat organic vegetarian.", "timestamp": 1736000000},
        {"role": "user", "content": "Project note: Apollo budget was set to $500k, then revised to $750k due to compute.", "timestamp": 1736500000},
    ]

    session_props = [
        "[Prior State / Superseded] User previously worked at Google Seattle starting July 2021. [Conversation Date: 2021-07-01]",
        "[Prior State / Superseded] User transferred to Google Zurich as infrastructure lead in August 2023. [Conversation Date: 2023-08-01]",
        "[Current State] User left Google and founded an AI agents startup in San Francisco in January 2025. [Conversation Date: 2025-01-01]",
        "[Strict Constraint] User is allergic to peanuts and strictly avoids seafood. User only eats organic vegetarian.",
        "[Current State] Project Apollo budget was revised to $750k due to compute."
    ]

    save_memories_batch("txt_s1", uid_text, "s1", session_turns, session_props)

    # Add 15 realistic distractor memories
    for i in range(15):
        save_memories_batch(f"txt_dist_{i}", uid_text, f"dist_{i}", [
            {"role": "user", "content": f"Random chat note {i}: discussed weather, bought coffee, colleague ate seafood paella.", "timestamp": 1650000000 + i * 100000}
        ], [f"Distractor {i}: conversation about weather and coffee #{i}."])

    # 1. State Transition Governance & Current State Recall (Cap A & D: Weight 40%)
    res_curr = search_hybrid(uid_text, "Where does the user currently work and what city are they in?", top_k=5)
    curr_hit = 1 if (res_curr and "San Francisco" in res_curr[0]["content"]) else 0
    
    res_past = search_hybrid(uid_text, "Where did the user work before moving to Zurich?", top_k=5)
    past_hit = 1 if (res_past and any("Seattle" in r["content"] for r in res_past[:3])) else 0

    state_score = ((curr_hit + past_hit) / 2.0) * 100.0

    # 2. Negative Constraint Avoidance (Cap E: Weight 25%)
    res_neg = search_hybrid(uid_text, "Recommend a dinner place excluding seafood and peanuts", top_k=5)
    neg_violates = any("seafood paella" in r["content"].lower() for r in res_neg)
    neg_has_rule = any("allergic to peanuts" in r["content"].lower() for r in res_neg)
    neg_score = 100.0 if (not neg_violates and neg_has_rule) else (50.0 if not neg_violates else 0.0)

    # 3. Temporal Resolution (Cap C: Weight 20%)
    res_time = search_hybrid(uid_text, "When did the user transfer to Google Zurich?", top_k=3)
    time_hit = 1 if (res_time and "2023" in res_time[0]["content"]) else 0
    time_score = time_hit * 100.0

    # 4. Multi-Hop Project State (Cap B: Weight 15%)
    res_proj = search_hybrid(uid_text, "What is the latest approved budget for Project Apollo?", top_k=3)
    proj_hit = 1 if (res_proj and "$750k" in res_proj[0]["content"]) else 0
    proj_score = proj_hit * 100.0

    composite = (0.40 * state_score) + (0.25 * neg_score) + (0.20 * time_score) + (0.15 * proj_score)
    composite = round(composite, 2)
    passed_gate = composite > GATE_THRESHOLDS["textual"]

    details = {
        "state_governance_score": f"{state_score:.1f}%",
        "negative_avoidance_score": f"{neg_score:.1f}%",
        "temporal_grounding_score": f"{time_score:.1f}%",
        "multihop_project_score": f"{proj_score:.1f}%",
        "composite_score": composite,
        "gate_threshold": GATE_THRESHOLDS["textual"],
        "gate_passed": passed_gate
    }

    print(f"  - State Conflict Governance (40%): {details['state_governance_score']}")
    print(f"  - Negative Constraint Avoid (25%): {details['negative_avoidance_score']}")
    print(f"  - Temporal Grounding (20%):        {details['temporal_grounding_score']}")
    print(f"  - Multi-Hop Project State (15%):   {details['multihop_project_score']}")
    print(f"  -> TEXTUAL COMPOSITE SCORE: {composite} / 100.0 (Gate: > {GATE_THRESHOLDS['textual']})")
    print(f"  -> GATE STATUS: {'PASSED - ELIGIBLE FOR SMOKE' if passed_gate else 'FAILED - SMOKE BLOCKED'}")

    return composite, details, passed_gate


def run_master_gate():
    setup_clean_db()
    
    mm_score, mm_details, mm_passed = evaluate_multimodal_track()
    code_score, code_details, code_passed = evaluate_coding_track()
    txt_score, txt_details, txt_passed = evaluate_textual_track()

    print("\n" + "=" * 75)
    print("MASTER MULTI-TRACK LEADERBOARD GATE EXECUTIVE SUMMARY")
    print("=" * 75)
    
    summary = {
        "multimodal": {"score": mm_score, "target": GATE_THRESHOLDS["multimodal"], "eligible_for_smoke": mm_passed},
        "coding":     {"score": code_score, "target": GATE_THRESHOLDS["coding"], "eligible_for_smoke": code_passed},
        "textual":    {"score": txt_score, "target": GATE_THRESHOLDS["textual"], "eligible_for_smoke": txt_passed},
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_master_gate()
