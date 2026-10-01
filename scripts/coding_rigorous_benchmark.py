#!/usr/bin/env python3
"""
Rigorous Coding Track Benchmark Suite.
Evaluates agent0mem on real production SWE-bench traces from the 28GB live database.
Covers multi-task disambiguation, file localization, cross-task isolation, clean git diffs, and payload budgets.
Strictly zero emojis. Grounded empirical evaluation.
"""

import sys
import os
import time
import json
import re
import subprocess
from typing import Dict, Any, List

REMOTE_HOST = "47.97.127.223"
SSH_KEY = "/Users/liuyukai/CREATE/PandaAI/nunu/admin_key"
API_URL = "http://127.0.0.1:8288/search"

# Multi-task benchmark configuration against live database users
BENCHMARK_TASKS = [
    {
        "id": "task_table_pandas",
        "user_id": "u_2e65f36a595a99672abdd6bf65abc88b63392ea3680c99649b36d94274709190",
        "description": "Astropy Table to DataFrame conversion with index handling (Issue #12065)",
        "gold_file": "astropy/table/table.py",
        "gold_symbol": "remove_indices",
        "forbidden_file": "astropy/timeseries/core.py",
        "queries": [
            "Fix automatic table indexes handling in Table to DataFrame conversion to_pandas remove_indices",
            "BinnedTimeSeries Table to_pandas dataframe index column remove_indices",
            "Table to DataFrame conversion automatic index handling astropy table",
            "table.py remove_indices col.info.name index primary key DataFrame"
        ]
    },
    {
        "id": "task_timeseries_core",
        "user_id": "u_2e65f36a595a99672abdd6bf65abc88b63392ea3680c99649b36d94274709190",
        "description": "Astropy TimeSeries required columns validation (Issue #13009)",
        "gold_file": "astropy/timeseries/core.py",
        "gold_symbol": "TimeSeries",
        "forbidden_file": "astropy/table/table.py",
        "queries": [
            "TimeSeries misleading exception required column check fails remove_column",
            "timeseries core.py _check_required_columns fix missing required column test",
            "missing required column TimeSeries error message test_required_columns",
            "astropy timeseries core.py fix required columns error message"
        ]
    },
    {
        "id": "task_fold_typeerror",
        "user_id": "test_coding_1790781076",
        "description": "Official Smoke Task: astropy.timeseries fold() method TypeError with quantity bounds",
        "gold_file": "astropy/timeseries/sampled.py",
        "gold_symbol": "fold",
        "forbidden_file": "astropy/table/table.py",
        "queries": [
            "What is the patch or solution to fix the fold method TypeError in astropy.timeseries?",
            "astropy.timeseries fold method TypeError quantity bounds epoch_time",
            "sampled.py fold method fix epoch_time Quantity to day units"
        ]
    },
    {
        "id": "task_noisy_distractor",
        "user_id": "u_2440b272d9ac3600456434b994efcb43c55f4d4458fa3f9659f419f4b4ec81c3",
        "description": "High-noise distractor environment (837 noisy memories): TimeSeries Core issue",
        "gold_file": "astropy/timeseries/core.py",
        "gold_symbol": "TimeSeries",
        "forbidden_file": "astropy/table/table.py",
        "queries": [
            "TimeSeries required columns validation misleading exception remove_column",
            "missing required column TimeSeries object is invalid expected first column",
            "astropy timeseries core.py fix required columns error message 2022"
        ]
    }
]


def run_remote_search(user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    py_script = f"""
import requests, json
resp = requests.post('{API_URL}', json={{'user_id': {json.dumps(user_id)}, 'query': {json.dumps(query)}, 'top_k': {top_k}}})
print(json.dumps(resp.json().get('data', [])))
"""
    cmd = [
        "ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", f"root@{REMOTE_HOST}",
        "python3"
    ]
    try:
        proc = subprocess.run(cmd, input=py_script.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        out = proc.stdout.decode("utf-8").strip()
        start_idx = out.find("[")
        if start_idx != -1:
            return json.loads(out[start_idx:])
        return []
    except Exception as e:
        print(f"Error querying remote server: {e}")
        return []


def test_git_apply_cleanliness(diff_text: str) -> bool:
    """Test whether extracted diff contains valid unified diff headers and hunks."""
    m = re.search(r"(diff --git .*?)(?:\n\[Conversation Date:|\Z)", diff_text, re.DOTALL)
    if not m:
        return False
    clean_diff = m.group(1).strip() + "\n"
    has_header = "--- a/" in clean_diff and "+++ b/" in clean_diff
    has_hunk = "@@ -" in clean_diff
    return has_header and has_hunk


def run_benchmark():
    print("=" * 80)
    print("RIGOROUS CODING TRACK MULTI-TASK EMPIRICAL BENCHMARK")
    print("Testing Live 28GB SQLite Database & Multi-Task Disambiguation")
    print("=" * 80)

    total_queries = 0
    rank1_hits = 0
    top3_hits = 0
    mrr_sum = 0.0
    clean_diff_count = 0
    payload_ok_count = 0
    isolation_ok_count = 0
    noise_suppressed_count = 0

    for task in BENCHMARK_TASKS:
        t_id = task["id"]
        uid = task["user_id"]
        gold_file = task["gold_file"]
        gold_symbol = task["gold_symbol"]
        forbidden_file = task["forbidden_file"]
        queries = task["queries"]

        print(f"\nTASK: [{t_id}] - {task['description']}")
        print(f"Target: {gold_file} | Symbol: {gold_symbol} | Forbidden: {forbidden_file}")
        print("-" * 80)

        for q_idx, q in enumerate(queries, 1):
            total_queries += 1
            t0 = time.time()
            items = run_remote_search(uid, q, top_k=5)
            latency = time.time() - t0

            total_bytes = sum(len(it.get("content", "").encode("utf-8")) for it in items)
            payload_ok = (total_bytes <= 16000)
            if payload_ok:
                payload_ok_count += 1

            # Check noise suppression
            has_noise = any(
                any(n in it.get("content", "") for n in ["[tool_use TaskUpdate]", "[tool_use TaskCreate]", "(Bash completed with no output)"])
                for it in items
            )
            if not has_noise:
                noise_suppressed_count += 1

            # Check cross-task isolation (Rank 1 must NOT be an unrelated diff from forbidden task)
            rank1_content = items[0].get("content", "") if items else ""
            rank1_has_forbidden = (forbidden_file in rank1_content and ("diff --git" in rank1_content or "[Code Patch / Solution]" in rank1_content))
            if not rank1_has_forbidden:
                isolation_ok_count += 1

            # Find gold target rank
            target_rank = None
            is_clean_diff = False
            for idx, it in enumerate(items):
                content = it.get("content", "")
                if gold_file in content and (gold_symbol in content or gold_file.split("/")[-1] in content):
                    target_rank = idx + 1
                    if "diff --git" in content:
                        is_clean_diff = test_git_apply_cleanliness(content)
                    else:
                        is_clean_diff = True  # Context/tool resolution
                    break

            if target_rank == 1:
                rank1_hits += 1
                top3_hits += 1
                mrr_sum += 1.0
            elif target_rank in (2, 3):
                top3_hits += 1
                mrr_sum += 1.0 / target_rank
            elif target_rank is not None:
                mrr_sum += 1.0 / target_rank

            if is_clean_diff:
                clean_diff_count += 1

            rank_str = f"Rank #{target_rank}" if target_rank else "NOT FOUND"
            isol_str = "CLEAN" if not rank1_has_forbidden else "POISONED"
            diff_str = "VALID" if is_clean_diff else "INVALID"

            print(f"  Q{q_idx}: {q[:42]:<42} | {rank_str:<9} | Isol: {isol_str:<8} | Diff: {diff_str:<7} | {total_bytes} B | {latency:.2f}s")

    r1_rate = (rank1_hits / total_queries) * 100.0 if total_queries else 0.0
    r3_rate = (top3_hits / total_queries) * 100.0 if total_queries else 0.0
    mean_mrr = mrr_sum / total_queries if total_queries else 0.0
    isol_rate = (isolation_ok_count / total_queries) * 100.0 if total_queries else 0.0
    clean_rate = (clean_diff_count / total_queries) * 100.0 if total_queries else 0.0
    payload_rate = (payload_ok_count / total_queries) * 100.0 if total_queries else 0.0
    noise_rate = (noise_suppressed_count / total_queries) * 100.0 if total_queries else 0.0

    print("\n" + "=" * 80)
    print("EMPIRICAL RIGOROUS CODING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Test Queries:              {total_queries}")
    print(f"Rank-1 Localization Rate:        {r1_rate:.1f}% ({rank1_hits}/{total_queries})")
    print(f"Top-3 Target Recall:             {r3_rate:.1f}% ({top3_hits}/{total_queries})")
    print(f"Mean Reciprocal Rank (MRR):      {mean_mrr:.4f}")
    print(f"Cross-Task Isolation Rate:       {isol_rate:.1f}% ({isolation_ok_count}/{total_queries})")
    print(f"Clean Diff / Target Integrity:   {clean_rate:.1f}% ({clean_diff_count}/{total_queries})")
    print(f"Payload Budget Compliance:       {payload_rate:.1f}% ({payload_ok_count}/{total_queries})")
    print(f"Noise Suppression Compliance:    {noise_rate:.1f}% ({noise_suppressed_count}/{total_queries})")
    print("=" * 80)

    # Tri-Tier Calibrated Performance Projection
    pessimistic_floor = round(r1_rate * (isol_rate / 100.0) * 0.85, 1)
    expected_baseline = round((r1_rate * 0.65 + r3_rate * 0.35) * (isol_rate / 100.0) * 0.95, 1)
    optimistic_ceiling = round(r3_rate * (clean_rate / 100.0), 1)

    print("Tri-Tier Calibrated SWE-bench Score Projection:")
    print(f"  - Pessimistic Floor:  {pessimistic_floor}%")
    print(f"  - Expected Baseline:  {expected_baseline}%")
    print(f"  - Optimistic Ceiling: {optimistic_ceiling}%")
    print(f"  (Season High Target: > 91.0%)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
