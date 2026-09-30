#!/usr/bin/env python3
"""
Automated Smoke Test Iterative Optimization & Radar Benchmarker for Agent0Mem.
Triggers zero-cooldown smoke tests, tracks capability and subcapability score progressions,
and benchmarks directly against the 87+ target.
"""

import os
import sys
import json
import time
import uuid
import argparse
import urllib.request
import urllib.error
from datetime import datetime

AML_BASE_URL = "https://agentmemoryleaderboard.ai"
HISTORY_FILE = "benchmark_smoke_history.json"


def request_json(url: str, method: str = "GET", headers: dict = None, body: dict = None, timeout: float = 30.0):
    all_headers = {
        "User-Agent": "Mozilla/5.0 (Agent0Mem-AutoBenchmarker/2.0)",
        "Accept": "application/json",
    }
    if headers:
        all_headers.update(headers)

    data = None
    if body is not None:
        all_headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=all_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
        except Exception:
            err_json = {"detail": err_body}
        return e.code, err_json
    except Exception as e:
        return 0, {"detail": str(e)}


def run_smoke_iteration(api_key: str, add_url: str, search_url: str, health_url: str, label: str = "iteration"):
    client_id = str(uuid.uuid4())
    print("=" * 60)
    print(f"Launching Zero-Cooldown Smoke Test [{label}]")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Add URL:    {add_url}")
    print(f"Search URL: {search_url}")
    print(f"Health URL: {health_url}")
    print("=" * 60)

    # 1. Trigger smoke test
    status, res = request_json(
        f"{AML_BASE_URL}/integration-tests",
        method="POST",
        headers={
            "X-Integration-Client-Id": client_id,
            "X-Leaderboard-Api-Key": api_key,
        },
        body={
            "add_endpoint": add_url,
            "search_endpoint": search_url,
            "health_endpoint": health_url,
            "api_auth_scheme": "none",
            "memory_api_key": None
        }
    )

    if status not in (200, 201):
        print(f"Error triggering smoke test (HTTP {status}): {res}")
        return None

    test_id = res.get("test_id")
    test_token = res.get("test_token")
    print(f"Smoke test submitted: Test ID={test_id}")

    # 2. Poll progress
    poll_url = f"{AML_BASE_URL}/integration-tests/{test_id}"
    start_time = time.time()
    last_msg = ""

    while True:
        time.sleep(3)
        p_status, p_res = request_json(
            poll_url,
            method="GET",
            headers={"X-Integration-Test-Token": test_token}
        )

        if p_status != 200:
            print(f"Polling HTTP {p_status}: {p_res}")
            continue

        stage = p_res.get("status")
        progress = p_res.get("progress", "")
        elapsed = int(time.time() - start_time)

        msg = f"[{elapsed}s] Status: {stage} | {progress}"
        if msg != last_msg:
            print(f"  {msg}")
            last_msg = msg

        if stage in ("passed", "success", "completed", "succeeded"):
            print("\nSmoke test COMPLETED! Fetching detailed results...")
            break
        elif stage in ("failed", "error"):
            print(f"\nSmoke test FAILED: {p_res.get('error', p_res)}")
            return None

        if elapsed > 600:
            print("Smoke test timed out after 10 minutes.")
            return None

    # 3. Retrieve evaluation job detail for capability scores
    job_id = p_res.get("evaluation_id") or p_res.get("job_id") or test_id
    eval_status, eval_res = request_json(
        f"{AML_BASE_URL}/api/v1/evaluations/{job_id}",
        headers={"X-Leaderboard-Api-Key": api_key}
    )

    score_data = eval_res.get("result", {})
    if not score_data:
        score_data = p_res.get("result", {})

    record = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "test_id": test_id,
        "job_id": job_id,
        "score_data": score_data
    }

    # Save to history file
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(record)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print_radar_summary(record)
    return record


def print_radar_summary(record: dict):
    data = record.get("score_data", {})
    overall = data.get("score", 0.0) * 100.0
    caps = {c["id"]: c.get("score", 0.0) * 100.0 for c in data.get("capabilities", [])}
    subcaps = {s["id"]: s.get("score", 0.0) * 100.0 for s in data.get("subcapabilities", [])}

    print("\n" + "=" * 60)
    print(f"SMOKE TEST RADAR REPORT: [{record.get('label')}]")
    print(f"Overall Composite Score: {overall:.2f}% (Target: >87.0%)")
    print("=" * 60)
    print(f"  Capability A (Fact & Entity Recall):         {caps.get('A', 0.0):.1f}%")
    print(f"  Capability B (Temporal & Calendar):          {caps.get('B', 0.0):.1f}% (B2: {subcaps.get('B2', 0.0):.1f}%)")
    print(f"  Capability C (Preference Evolution):         {caps.get('C', 0.0):.1f}% (C1: {subcaps.get('C1', 0.0):.1f}%)")
    print(f"  Capability D (Negative Rules & Constraints): {caps.get('D', 0.0):.1f}% (D1: {subcaps.get('D1', 0.0):.1f}%)")
    print(f"  Capability E (State Change & Conflict):      {caps.get('E', 0.0):.1f}%")
    print(f"  Capability G (Dialogue Flow & Topic Shift):  {caps.get('G', 0.0):.1f}% (G4: {subcaps.get('G4', 0.0):.1f}%)")
    print(f"  Capability H (Long-Context & Multi-Hop):     {caps.get('H', 0.0):.1f}%")
    print(f"  Streaming Recall (Online & Transfer):        {caps.get('streaming', 0.0):.1f}%")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Agent0Mem Smoke Iterative Benchmarker")
    parser.add_argument("--key", default=os.environ.get("AML_API_KEY", "ldbd_sk_Wj20KAB5b3BMMf_qcpGHS2at-F7JlD7hqSX-qGTVma4"))
    parser.add_argument("--add-url", default="http://47.97.127.223/agent0mem/add")
    parser.add_argument("--search-url", default="http://47.97.127.223/agent0mem/search")
    parser.add_argument("--health-url", default="http://47.97.127.223/agent0mem/health")
    parser.add_argument("--label", default="v2_dev_run")
    args = parser.parse_args()

    run_smoke_iteration(args.key, args.add_url, args.search_url, args.health_url, args.label)


if __name__ == "__main__":
    main()
