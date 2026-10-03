import urllib.request, json

BASE = "http://47.97.127.223/agent0mem"

test_cases = [
    {
        "name": "Item 8: Direct Recall (Coffee spend)",
        "user_id": "u_34419a896aa604b2cb7763a4f5a433f176ade7e39851f80b2c409e6ed5943084",
        "query": "How much total have I spent on coffee makers?\nAnswer with the exact amount (e.g., \"$45.00\") only."
    },
    {
        "name": "Item 11: Atomic Retrieval (Ethos Water color)",
        "user_id": "u_6836a2f75bd366a4c395247cb13334a34a6cd60646094b5f6406038d4f378bf3",
        "query": "What is the background color in the original image of the Starbucks Ethos Water 'heart-shaped pour' ad?"
    },
    {
        "name": "Item 12: Atomic Retrieval (Ethos Water MCQ)",
        "user_id": "u_f72cb4973ac52a7a48a607bbb0f14c75f2e37f48424de3c983afb970972f1265",
        "query": "For the Starbucks Ethos Water 'heart-shaped pour' ad, which background description matches the original image?\n\nA. White"
    },
    {
        "name": "Item 13: Relational Reasoning (Engagement steps)",
        "user_id": "u_7fcbf9e38b0426ddfca13ee6a137277ac3660879ce0281187f989676968b4735",
        "query": "订婚纪念夜那晚，我先后做了哪两步，才慢慢把职业焦虑想清楚？"
    }
]

for tc in test_cases:
    print("=" * 60)
    print(tc["name"])
    req = urllib.request.Request(f"{BASE}/search", data=json.dumps({"user_id": tc["user_id"], "query": tc["query"], "top_k": 5}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
    items = res.get("data", [])
    print(f"Returned {len(items)} items:")
    for i, it in enumerate(items):
        cnt = it.get("content", it.get("text", ""))
        print(f"  [{i+1}] (score={it.get('score', 0.0):.4f}) {cnt}")
