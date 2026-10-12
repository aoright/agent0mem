"""Multimodal High-Concurrency Stress Test Suite (16-thread benchmark).

Simulates official evaluation concurrency (search_concurrency=16) across realistic
multimodal queries (image attribute retrieval, cross-session temporal reasoning,
adversarial abstention traps, and narrative bridging).

Measures:
- Request success rate (target: 100.0%, strictly 0 timeouts / 0 504s)
- Latency percentiles: P50, P90, P95, P99, Max
- Throughput (QPS under 16 concurrent workers)
- Memory and payload safety

Usage:
  python3 scripts/stress_test_multimodal_concurrency.py --mode direct --n-queries 1000 --workers 16
  python3 scripts/stress_test_multimodal_concurrency.py --mode http --target http://127.0.0.1:8288 --n-queries 500 --workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Base queries representative of the official AML multimodal distribution
BENCHMARK_PROMPTS = [
    # Visual / Image groundings
    "What is the background color of the image showing the beverage bottle?",
    "What color is the background in the Starbucks Ethos Water photo?",
    "What is shown in the photo shared during the dinner discussion?",
    "What was the price displayed on the receipt in the image?",
    "Describe the visual objects in the image sent last weekend.",
    # Temporal & Chronological sequences
    "What happened first: the pottery workshop or the road trip?",
    "In what order did the two projects get completed?",
    "When did the user book the Airbnb in Seattle?",
    "Which event happened earliest in the sequence?",
    # Multi-session narrative & Location bridging
    "Where did Jolene and her partner find a cool diving spot?",
    "What video games does Nate play regularly?",
    "Where did we go for our anniversary retreat?",
    "What was the outcome of the tournament discussed in September?",
    # In-domain Epistemic Abstention Traps (adversarial speaker attribution)
    "What kind of sports car did Sarah purchase last month?",
    "How many marathons has David completed so far?",
    "When did the assistant attend the international conference in Berlin?",
    "What was the recipe for the vegan pasta dish prepared yesterday?",
    "Why was the project cancelled after the initial kickoff meeting?",
]


def load_heldout_users(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT user_id FROM memories WHERE user_id LIKE 'heldout_%'")
    users = [r[0] for r in cur.fetchall()]
    conn.close()
    return users


def direct_worker(job: Tuple[str, str, int]) -> Dict[str, Any]:
    from app.core.clean_search import clean_search

    uid, q, top_k = job
    t0 = time.perf_counter()
    try:
        items = clean_search(uid, q, top_k=top_k, max_items=100, byte_budget=48_000)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": True,
            "status": 200,
            "duration_ms": dur_ms,
            "count": len(items),
            "bytes": sum(len(it.get("content", "").encode("utf-8")) for it in items),
            "error": None,
        }
    except Exception as exc:
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "status": 500,
            "duration_ms": dur_ms,
            "count": 0,
            "bytes": 0,
            "error": str(exc),
        }


def http_worker(job: Tuple[str, str, str, int]) -> Dict[str, Any]:
    endpoint, uid, q, top_k = job
    url = f"{endpoint.rstrip('/')}/agent0mem/search"
    payload = json.dumps({
        "user_id": uid,
        "query": q,
        "top_k": top_k,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            data = json.loads(resp.read().decode("utf-8"))
            items = data.get("data", [])
            return {
                "success": True,
                "status": resp.status,
                "duration_ms": dur_ms,
                "count": len(items),
                "bytes": sum(len(it.get("content", "").encode("utf-8")) for it in items),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "status": exc.code,
            "duration_ms": dur_ms,
            "count": 0,
            "bytes": 0,
            "error": f"HTTP {exc.code}: {exc.reason}",
        }
    except Exception as exc:
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "status": 504 if "timed out" in str(exc).lower() else 500,
            "duration_ms": dur_ms,
            "count": 0,
            "bytes": 0,
            "error": str(exc),
        }


def run_benchmark(mode: str, target: str, n_queries: int, workers: int, db_path: str):
    print("=" * 70)
    print("Agent0Mem High-Concurrency Multimodal Stress Benchmark")
    print(f"Mode: {mode}")
    print(f"Concurrency Workers: {workers} (Official benchmark matches 16)")
    print(f"Total Queries: {n_queries}")
    if mode == "http":
        print(f"Target Endpoint: {target}")
    print("=" * 70)

    users = load_heldout_users(db_path)
    if not users:
        print(f"Error: No heldout users found in database {db_path}")
        return

    print(f"Loaded {len(users)} distinct heldout users from database.")
    random.seed(42)
    jobs = []
    for _ in range(n_queries):
        u = random.choice(users)
        q = random.choice(BENCHMARK_PROMPTS)
        if mode == "direct":
            jobs.append((u, q, 100))
        else:
            jobs.append((target, u, q, 100))

    t_start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    worker_fn = direct_worker if mode == "direct" else http_worker

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_fn, j) for j in jobs]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if i % 100 == 0 or i == n_queries:
                avg_ms = sum(r["duration_ms"] for r in results) / len(results)
                success_n = sum(1 for r in results if r["success"])
                print(f"Progress: {i}/{n_queries} ({i/n_queries*100:.1f}%) | "
                      f"Success: {success_n}/{i} | Current Mean Latency: {avg_ms:.1f}ms")

    total_time = time.perf_counter() - t_start
    qps = n_queries / total_time

    # Calculate statistics
    durations = sorted(r["duration_ms"] for r in results)
    p50 = durations[int(len(durations) * 0.50)]
    p90 = durations[int(len(durations) * 0.90)]
    p95 = durations[int(len(durations) * 0.95)]
    p99 = durations[int(len(durations) * 0.99)]
    max_lat = durations[-1]
    min_lat = durations[0]

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    success_rate = (len(successes) / len(results)) * 100.0

    timeouts_504 = sum(1 for r in failures if r.get("status") == 504)
    errors_500 = sum(1 for r in failures if r.get("status") == 500)

    print("\n" + "=" * 70)
    print("STRESS BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Completed Requests: {len(results)}")
    print(f"Success Rate:             {success_rate:.2f}% ({len(successes)}/{len(results)})")
    print(f"Failed Requests:          {len(failures)}")
    print(f"  HTTP 504 Timeouts:      {timeouts_504} (Target: strictly 0)")
    print(f"  HTTP 500 Errors:        {errors_500} (Target: strictly 0)")
    print(f"Total Wall Time:          {total_time:.2f} seconds")
    print(f"Effective Throughput:     {qps:.2f} QPS (under {workers} parallel threads)")
    print("\nLatency Distribution (per request):")
    print(f"  Min:                    {min_lat:.1f} ms")
    print(f"  P50 (Median):           {p50:.1f} ms")
    print(f"  P90:                    {p90:.1f} ms")
    print(f"  P95:                    {p95:.1f} ms")
    print(f"  P99:                    {p99:.1f} ms (Target: < 800 ms)")
    print(f"  Max:                    {max_lat:.1f} ms (Target: < 2000 ms)")
    print("=" * 70)

    if failures:
        print("\nSample Failure Details:")
        for f in failures[:5]:
            print(f"  Status: {f.get('status')} | Error: {f.get('error')}")


def main():
    parser = argparse.ArgumentParser(description="High-Concurrency Multimodal Stress Benchmark")
    parser.add_argument("--mode", choices=["direct", "http"], default="direct", help="Evaluation mode")
    parser.add_argument("--target", default="http://47.97.127.223", help="Target endpoint for HTTP mode")
    parser.add_argument("--n-queries", type=int, default=1000, help="Total queries to simulate")
    parser.add_argument("--workers", type=int, default=16, help="Concurrency worker threads")
    parser.add_argument("--db-path", default=os.path.expanduser("~/aml_heldout/db2/memories.sqlite3"),
                        help="Path to heldout SQLite database")
    args = parser.parse_args()

    os.environ["AGENT0MEM_DATA_DIR"] = os.path.dirname(os.path.abspath(args.db_path))
    run_benchmark(args.mode, args.target, args.n_queries, args.workers, args.db_path)


if __name__ == "__main__":
    main()
