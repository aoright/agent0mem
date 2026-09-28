#!/usr/bin/env python3
"""
Agent0Mem AML Platform Automation Client.
Provides automated verification, smoke testing, and evaluation tracking
for the Agent Memory Leaderboard (AML).
"""

import os
import sys
import json
import time
import uuid
import argparse
import urllib.request
import urllib.error

AML_BASE_URL = "https://agentmemoryleaderboard.ai"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"


def request_json(url: str, method: str = "GET", headers: dict = None, body: dict = None, timeout: float = 30.0):
    all_headers = {
        "User-Agent": USER_AGENT,
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


def validate_key(api_key: str):
    print(f"Validating AML API key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else ''}")
    status, res = request_json(
        f"{AML_BASE_URL}/access/validate",
        method="POST",
        headers={"X-Leaderboard-Api-Key": api_key},
        body={"api_key": api_key}
    )
    if status == 200:
        print("Key validation SUCCESSful!")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return res
    else:
        print(f"Key validation FAILED (HTTP {status}): {res}")
        return None


def run_smoke_test(api_key: str, add_url: str, search_url: str, health_url: str):
    client_id = str(uuid.uuid4())
    print("\nStarting public compatibility Smoke Test...")
    print(f"  Add URL:    {add_url}")
    print(f"  Search URL: {search_url}")
    print(f"  Health URL: {health_url}")

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
        print(f"Failed to initiate smoke test (HTTP {status}): {res}")
        return False

    test_id = res.get("test_id")
    test_token = res.get("test_token")
    print(f"Smoke test initiated. Test ID: {test_id} (Token: {test_token})")

    # Poll smoke test status using X-Integration-Test-Token
    poll_url = f"{AML_BASE_URL}/integration-tests/{test_id}"
    for attempt in range(200):
        time.sleep(3)
        p_status, p_res = request_json(
            poll_url,
            method="GET",
            headers={
                "X-Integration-Test-Token": test_token,
            }
        )
        if p_status != 200:
            print(f"Polling HTTP {p_status}: {p_res}")
            continue

        stage = p_res.get("status")
        progress = p_res.get("progress", "")
        print(f"  [Poll {attempt + 1}] Status: {stage} | Details: {progress}")

        if stage in ("passed", "success", "completed"):
            print("Smoke test PASSED! System is fully compatible.")
            return True
        elif stage in ("failed", "error"):
            print(f"Smoke test FAILED: {p_res.get('error', p_res)}")
            return False

    print("Smoke test timed out after 10 minutes.")
    return False


def get_active_versions(api_key: str):
    status, res = request_json(
        f"{AML_BASE_URL}/access/versions",
        method="GET",
        headers={"X-Leaderboard-Api-Key": api_key}
    )
    if status == 200:
        return res.get("versions", [])
    return []


def run_full_evaluation(api_key: str, version_id: str, run_label: str = "agent0mem_v1_run1"):
    print(f"\nSubmitting Full Leaderboard Evaluation for version {version_id}...")
    v2_payload = {
        "version_id": version_id,
        "benchmark_type": "textual",
        "phase": "full",
        "options": {
            "max_add_concurrency": 16,
            "search_concurrency": 16,
            "top_k": 100,
        },
        "run_label": run_label
    }

    # Primary: v2 track evaluations endpoint
    status, res = request_json(
        f"{AML_BASE_URL}/api/v1/evaluations",
        method="POST",
        headers={"X-Leaderboard-Api-Key": api_key},
        body=v2_payload
    )

    if status not in (200, 201):
        print(f"v2 submission returned HTTP {status}: {res}, trying legacy endpoint...")
        # Fallback to legacy eval-jobs
        status, res = request_json(
            f"{AML_BASE_URL}/eval-jobs",
            method="POST",
            headers={"X-Leaderboard-Api-Key": api_key},
            body={
                "version_id": version_id,
                "mode": "full",
                "max_add_concurrency": 16,
                "top_k": 100,
                "run_label": run_label
            }
        )

    if status not in (200, 201):
        print(f"Failed to submit evaluation job (HTTP {status}): {res}")
        return None

    job_id = res.get("evaluation_id") or res.get("job_id")
    print(f"Evaluation submitted successfully! Job ID: {job_id}")
    return job_id


def monitor_job(api_key: str, job_id: str):
    print(f"\nMonitoring evaluation job: {job_id}")
    v2_url = f"{AML_BASE_URL}/api/v1/evaluations/{job_id}"
    poll_url = f"{AML_BASE_URL}/eval-jobs/{job_id}"

    last_status = None
    poll_count = 0
    while True:
        poll_count += 1
        status, res = request_json(v2_url, headers={"X-Leaderboard-Api-Key": api_key})
        if status != 200:
            status, res = request_json(poll_url, headers={"X-Leaderboard-Api-Key": api_key})

        if status == 200:
            current_status = res.get("status")
            progress = res.get("progress", {})
            current_item = res.get("current", "")
            stage = res.get("stage", "")
            done = res.get("done") or (progress.get("done") if isinstance(progress, dict) else None)
            total = res.get("total") or (progress.get("total") if isinstance(progress, dict) else None)

            progress_str = f"[{current_status}] stage: {stage} | progress: {done}/{total} | {current_item}"
            print(f"[Poll {poll_count}] {progress_str}")

            if current_status in ("completed", "finished", "success"):
                print("\n==========================================")
                print("EVALUATION COMPLETED SUCCESSFULLY!")
                print("==========================================")
                scores = res.get("scores", res.get("summary", res.get("result", {})))
                print("Results:")
                print(json.dumps(scores, indent=2, ensure_ascii=False))
                print(json.dumps(res, indent=2, ensure_ascii=False))
                break
            elif current_status in ("failed", "error", "canceled"):
                print(f"\nEvaluation finished with status: {current_status}")
                print(json.dumps(res, indent=2, ensure_ascii=False))
                break
        else:
            print(f"[Poll {poll_count}] Polling HTTP {status}: {res}")

        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Agent0Mem AML Platform Runner")
    parser.add_argument("--key", help="AML Evaluation Key (ldbd_key)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate the key")
    parser.add_argument("--smoke", action="store_true", help="Run compatibility smoke test")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation")
    parser.add_argument("--add-url", default="http://47.97.127.223/agent0mem/add")
    parser.add_argument("--search-url", default="http://47.97.127.223/agent0mem/search")
    parser.add_argument("--health-url", default="http://47.97.127.223/agent0mem/health")
    parser.add_argument("--version-id", default=None, help="Bound version ID")
    parser.add_argument("--job-id", default=None, help="Existing job ID to monitor")

    args = parser.parse_args()

    api_key = args.key or os.environ.get("AML_API_KEY")
    if not api_key:
        print("Error: No AML API Key provided. Provide via --key or set AML_API_KEY environment variable.")
        sys.exit(1)

    if args.job_id:
        monitor_job(api_key, args.job_id)
        return

    session = validate_key(api_key)
    if not session or args.validate_only:
        return

    if args.smoke:
        smoke_passed = run_smoke_test(api_key, args.add_url, args.search_url, args.health_url)
        if not smoke_passed:
            print("Aborting: Smoke test must pass before running full evaluation.")
            sys.exit(1)

    if args.evaluate:
        version_id = args.version_id or session.get("bound_version_id") or session.get("version_id")
        if not version_id:
            versions = get_active_versions(api_key)
            if versions:
                version_id = versions[0].get("version_id")
                print(f"Auto-selected bound version: {version_id}")
            else:
                print("Error: No bound version found. Please specify --version-id.")
                sys.exit(1)

        job_id = run_full_evaluation(api_key, version_id)
        if job_id:
            monitor_job(api_key, job_id)


if __name__ == "__main__":
    main()
