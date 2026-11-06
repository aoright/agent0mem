import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import init_db, save_memories_batch, search_hybrid
from app.core.extractor import extract_propositions


def test_full_pipeline():
    print("Testing Agent0Mem Upgraded Hybrid Engine...")
    init_db()

    user_id = "test_user_eval_002"
    session_id = "sess_002"
    request_id = "req_002"

    # Multi-turn conversation with explicit timestamps and state update
    messages = [
        {"role": "user", "content": "Hi! My name is Jordan and I am a software engineer based in Seattle.", "timestamp": 1685000000},
        {"role": "assistant", "content": "Nice to meet you Jordan! How can I help you today?", "timestamp": 1685000010},
        {"role": "user", "content": "I usually drink black coffee every morning, but recently I stopped drinking coffee and switched completely to matcha latte.", "timestamp": 1685000020},
        {"role": "assistant", "content": "Got it, matcha latte is a great healthy alternative!", "timestamp": 1685000030},
        {"role": "user", "content": "Next month on October 15, I will be traveling to Kyoto for an AI conference.", "timestamp": 1685000040}
    ]

    print("Step 1: Extracting propositions with temporal context...")
    propositions = extract_propositions(messages)
    print(f"Extracted {len(propositions)} propositions:")
    for p in propositions:
        print(f"  - {p}")

    print("\nStep 2: Persisting memories with hybrid embeddings...")
    count = save_memories_batch(request_id, user_id, session_id, messages, propositions)
    print(f"Saved {count} total memory items.")

    print("\nStep 3A: Present-tense query: 'What does Jordan drink in the morning?'...")
    results_present = search_hybrid(user_id, "What does Jordan drink in the morning?", top_k=3)
    for idx, r in enumerate(results_present):
        print(f"  [{idx + 1}] Score: {r['score']} | Content: {r['content']}")

    print("\nStep 3B: Past-tense query: 'What did Jordan previously drink before switching?'...")
    results_past = search_hybrid(user_id, "What did Jordan previously drink before switching?", top_k=3)
    for idx, r in enumerate(results_past):
        print(f"  [{idx + 1}] Score: {r['score']} | Content: {r['content']}")

    print("\nStep 4: MCQ Search with options...")
    query = "Where is Jordan traveling next month?"
    options = ["A. Tokyo", "B. Kyoto", "C. Osaka", "D. Seoul"]
    results_mcq = search_hybrid(user_id, query, options=options, top_k=3)
    for idx, r in enumerate(results_mcq):
        print(f"  [{idx + 1}] Score: {r['score']} | Content: {r['content']}")

    print("\nVerification complete.")


if __name__ == "__main__":
    test_full_pipeline()
