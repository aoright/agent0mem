"""Replay the exact /search calls made by official textual smoke teval_bf5963d171bbe6d6.

Input: a file of lines "<user_id>: query=<repr>, returned N items" extracted from
journalctl. Only lines whose query is fully captured (not truncated at 120 chars)
are replayed verbatim; truncated ones are replayed with the prefix (marked).
Read-only: issues POST /search only, never /add.
"""
import ast
import json
import re
import sys
import urllib.request

BASE = "http://47.97.127.223/agent0mem"
LINE_RE = re.compile(r"^(u_[0-9a-f]+): query=(.+), returned (\d+) items\s*$")


def search(user_id, query, top_k=100):
    body = json.dumps({"user_id": user_id, "query": query, "top_k": top_k}).encode()
    req = urllib.request.Request(f"{BASE}/search", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main(path):
    seen = set()
    for raw in open(path, encoding="utf-8"):
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        uid, qrepr, n = m.group(1), m.group(2), int(m.group(3))
        try:
            q = ast.literal_eval(qrepr)
        except Exception:
            continue
        key = (uid, q)
        if key in seen:
            continue
        seen.add(key)
        truncated = len(q) >= 120
        res = search(uid, q)
        items = res.get("data", res.get("results", [])) or []
        print("=" * 100)
        print(f"USER {uid[:14]}  logged_items={n}  replay_items={len(items)}  truncated_query={truncated}")
        print(f"Q: {q[:200]!r}")
        for i, it in enumerate(items[:5]):
            c = (it.get("content") or it.get("text") or "").replace("\n", " ")
            print(f"  [{i+1}] score={it.get('score')} id={str(it.get('id') or it.get('memory_id'))[:40]} :: {c[:260]}")


if __name__ == "__main__":
    main(sys.argv[1])
