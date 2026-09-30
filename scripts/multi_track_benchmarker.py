#!/usr/bin/env python3
"""
Agent0Mem Multi-Track Evaluation & Benchmark Orchestrator.
Supports Textual, Coding, and Multimodal tracks on the Agent Memory Leaderboard (AML).
Enforces manual approval gate for Full evaluations.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Dict, Any, Optional

AML_BASE_URL = "https://agentmemoryleaderboard.ai"
DEFAULT_KEY = "ldbd_sk_Wj20KAB5b3BMMf_qcpGHS2at-F7JlD7hqSX-qGTVma4"
DEFAULT_VERSION = "version_268b35e9cfd4"
HISTORY_FILE = "benchmark_runs_history.json"


def request_json(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, body: Optional[Dict[str, Any]] = None, timeout: float = 30.0):
    all_headers = {
        "User-Agent": "Mozilla/5.0 (Agent0Mem-MultiTrack-Benchmarker/2.0)",
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


def get_track_eligibility(api_key: str, version_id: str):
    url = f"{AML_BASE_URL}/api/v1/versions/{version_id}/track-eligibility"
    status, res = request_json(url, headers={"X-Leaderboard-Api-Key": api_key})
    if status == 200:
        return res
    print(f"Error fetching track eligibility (HTTP {status}): {res}")
    return None


def print_track_status(eligibility: Dict[str, Any]):
    print("=" * 70)
    print("AGENT0MEM MULTI-TRACK ELIGIBILITY DASHBOARD")
    print(f"Version ID: {eligibility.get('version_id')}")
    print("=" * 70)
    tracks = eligibility.get("tracks", {})
    for track_name in ["textual", "coding", "multimodal"]:
        t_data = tracks.get(track_name, {})
        adm = t_data.get("admission", {})
        smoke_stat = f"{adm.get('smoke_used', 0)}/{adm.get('smoke_limit', 30)} (Allowed: {t_data.get('smoke_allowed')})"
        full_stat = f"{adm.get('full_used', 0)}/{adm.get('full_limit', 2)} (Allowed: {adm.get('new_full_allowed')})"
        contract = t_data.get("contract_version", "unknown")
        
        print(f"Track: {track_name.upper():12} | Contract: {contract:14}")
        print(f"  - Smoke Status:  {t_data.get('smoke', 'not_started'):12} | Quota: {smoke_stat}")
        print(f"  - Full Status:   Used {adm.get('full_used', 0)}/2     | Quota: {full_stat}")
        if adm.get("full_block_reason"):
            print(f"  - Block Reason:  {adm.get('full_block_reason')} (Next: {adm.get('next_full_at')})")
        print("-" * 70)


def start_evaluation(
    api_key: str,
    version_id: str,
    track: str,
    phase: str = "smoke",
    run_label: Optional[str] = None,
    confirmed_approval: bool = False
):
    if phase == "full" and not confirmed_approval:
        print("ERROR: Full evaluations require explicit manual confirmation. Aborting.")
        return None

    # Track-specific options
    options: Dict[str, Any] = {
        "max_add_concurrency": 16,
        "search_concurrency": 16,
    }
    if track == "coding":
        options["coding_model"] = "flash"
    elif track == "multimodal":
        options["top_k"] = 100
        options["media_transport"] = "inline_data_uri"
    elif track == "textual":
        options["top_k"] = 100

    payload = {
        "version_id": version_id,
        "benchmark_type": track,
        "phase": phase,
        "options": options,
    }
    if run_label:
        payload["run_label"] = run_label

    while True:
        print(f"\nSubmitting [{phase.upper()}] Evaluation for Track: [{track.upper()}]...")
        status, res = request_json(
            f"{AML_BASE_URL}/api/v1/evaluations",
            method="POST",
            headers={"X-Leaderboard-Api-Key": api_key},
            body=payload
        )

        if status in (200, 201):
            eval_id = res.get("evaluation_id")
            print(f"Evaluation queued successfully! Evaluation ID: {eval_id}")
            return eval_id

        detail = res.get("detail", {})
        reason_str = str(detail.get("reason", "") if isinstance(detail, dict) else detail)
        if "smoke_cooldown" in reason_str:
            import re
            match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", reason_str)
            if match:
                target_dt = datetime.fromisoformat(match.group(0) + "+00:00")
                wait_sec = max(2, int((target_dt - datetime.now(timezone.utc)).total_seconds()) + 3)
                print(f"Smoke cooldown active. Auto-waiting {wait_sec}s until {match.group(0)}...")
                time.sleep(wait_sec)
                continue

        print(f"Failed to submit evaluation (HTTP {status}): {res}")
        return None


def monitor_evaluation(api_key: str, eval_id: str, track: str):
    print(f"\nMonitoring [{track.upper()}] evaluation: {eval_id}")
    url = f"{AML_BASE_URL}/api/v1/evaluations/{eval_id}"
    start_time = time.time()
    last_msg = ""
    poll_count = 0

    while True:
        poll_count += 1
        time.sleep(5)
        status, res = request_json(url, headers={"X-Leaderboard-Api-Key": api_key})
        if status != 200:
            print(f"Poll HTTP {status}: {res}")
            continue

        c_status = res.get("status")
        stage = res.get("stage", "")
        progress = res.get("progress", {})
        done = res.get("done") or (progress.get("done") if isinstance(progress, dict) else None)
        total = res.get("total") or (progress.get("total") if isinstance(progress, dict) else None)
        elapsed = int(time.time() - start_time)

        msg = f"[{elapsed}s][Poll {poll_count}] Status: {c_status} | Stage: {stage} | Progress: {done}/{total}"
        if msg != last_msg:
            print(f"  {msg}")
            last_msg = msg

        if c_status in ("completed", "finished", "success", "succeeded"):
            print("\n==========================================")
            print(f"EVALUATION SUCCEEDED: [{track.upper()}]")
            print("==========================================")
            score_data = res.get("result", res.get("scores", res.get("summary", {})))
            print_score_report(track, score_data)
            record_history(eval_id, track, score_data)
            return score_data
        elif c_status in ("failed", "error", "canceled"):
            print(f"\nEvaluation ended with status: {c_status}")
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return None


def print_score_report(track: str, score_data: Dict[str, Any]):
    score = score_data.get("score", 0.0) * 100.0
    print(f"Overall Composite Score: {score:.2f}% (Target: >87.0%)")
    
    # Textual capabilities
    if "capabilities" in score_data:
        caps = {c.get("id"): c.get("score", 0.0) * 100.0 for c in score_data.get("capabilities", [])}
        print("Capability Breakdown:")
        for cid, cscore in sorted(caps.items()):
            print(f"  Capability {cid}: {cscore:.1f}%")

    # Coding tasks
    if "tasks" in score_data:
        tasks_val = score_data.get("tasks")
        if isinstance(tasks_val, dict):
            print("Coding Task Results:")
            for t, val in tasks_val.items():
                print(f"  Task {t}: {val}")
        else:
            print(f"Coding Tasks Count: {tasks_val}")

    if "difficulty" in score_data:
        diff = score_data.get("difficulty", {})
        print("Difficulty & Noise Performance:")
        for mode in ["relevant", "noisy"]:
            m_data = diff.get(mode, {})
            if m_data:
                print(f"  Mode [{mode.upper()}]: Resolved: {m_data.get('resolved_count', 0)}/{m_data.get('total', 0)} (Rate: {m_data.get('resolved_rate', 0.0) * 100:.1f}%), File Localized: {m_data.get('file_localized_rate', 0.0) * 100:.1f}%")

    # Multimodal datasets
    if "datasets" in score_data:
        print("Multimodal Dataset Results:")
        for ds, val in score_data.get("datasets", {}).items():
            print(f"  Dataset {ds}: {val}")


def record_history(eval_id: str, track: str, score_data: Dict[str, Any]):
    record = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_id": eval_id,
        "track": track,
        "score_data": score_data
    }
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


def main():
    parser = argparse.ArgumentParser(description="Agent0Mem Multi-Track Evaluation Orchestrator")
    parser.add_argument("--key", default=os.environ.get("AML_API_KEY", DEFAULT_KEY))
    parser.add_argument("--version-id", default=DEFAULT_VERSION)
    parser.add_argument("--status", action="store_true", help="Display multi-track eligibility and quota")
    parser.add_argument("--track", choices=["textual", "coding", "multimodal"], help="Target track")
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--label", default=None)
    parser.add_argument("--job-id", default=None, help="Existing job ID to monitor")
    parser.add_argument("--confirm-manual-approval", action="store_true", help="Gate for Full evaluation")

    args = parser.parse_args()

    if args.status or not args.track:
        eligibility = get_track_eligibility(args.key, args.version_id)
        if eligibility:
            print_track_status(eligibility)
        if not args.track:
            return

    if args.phase == "full" and not args.confirm_manual_approval:
        print("FATAL: Full evaluations require explicit user approval via --confirm-manual-approval. Exiting.")
        sys.exit(1)

    if args.job_id:
        monitor_evaluation(args.key, args.job_id, args.track)
        return

    eval_id = start_evaluation(
        args.key,
        args.version_id,
        args.track,
        args.phase,
        args.label,
        confirmed_approval=args.confirm_manual_approval
    )
    if eval_id:
        monitor_evaluation(args.key, eval_id, args.track)


if __name__ == "__main__":
    main()
