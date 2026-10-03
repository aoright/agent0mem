#!/usr/bin/env python3
"""
Comprehensive offline test suite verifying Textual, Multimodal, and Coding retrieval.
Validates:
1. Sarah Garcia biographical and epistemic abstention queries.
2. Portal CEO 4-hop knowledge graph update chain.
3. Multimodal coffee maker spend $150.00 leading token.
4. Multimodal engagement night two steps clean propositions.
5. In-domain epistemic abstentions.
6. Coding query patch isolation and noise elimination.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import search_hybrid, unwrap_proposition
from app.core.temporal import strip_temporal_markers

def run_tests():
    passed = 0
    total = 0

    print("=" * 70)
    print("COMPREHENSIVE OFFLINE RETRIEVAL & REASONING VERIFICATION SUITE")
    print("=" * 70)

    # 1. Sarah Garcia Biographical & Epistemic Tests
    uid_sarah = "u_d51798d575e1b63d760abdb1bb82f9aa92270c2281668f9cd05c9b5940f2d1e6"
    
    # Test 1.1: Middle Name (Must abstain / unmentioned)
    total += 1
    res = search_hybrid(user_id=uid_sarah, query_text="What is Sarah Garcia's middle name?", top_k=5)
    t1_pass = len(res) == 1 and "未提及" in res[0]["content"]
    if t1_pass:
        passed += 1
        print("[PASS] Test 1.1: Sarah Garcia Middle Name correctly abstains")
    else:
        print(f"[FAIL] Test 1.1: Sarah Garcia Middle Name: {res[0]['content'][:100] if res else 'EMPTY'}")

    # Test 1.2: Maiden Name (Must abstain / unmentioned)
    total += 1
    res = search_hybrid(user_id=uid_sarah, query_text="What was Sarah Garcia's maiden name before marriage?", top_k=5)
    t2_pass = len(res) == 1 and "未提及" in res[0]["content"]
    if t2_pass:
        passed += 1
        print("[PASS] Test 1.2: Sarah Garcia Maiden Name correctly abstains")
    else:
        print(f"[FAIL] Test 1.2: Sarah Garcia Maiden Name: {res[0]['content'][:100] if res else 'EMPTY'}")

    # Test 1.3: Birth year contradiction (born in 1985 vs actual 1986-03-26)
    total += 1
    res = search_hybrid(user_id=uid_sarah, query_text="What is Sarah Garcia's birth date in 1985?", top_k=5)
    t3_pass = any("1986-03-26" in r["content"] and "not in 1985" in r["content"] for r in res)
    if t3_pass:
        passed += 1
        print("[PASS] Test 1.3: Sarah Garcia Birth Year Contradiction caught")
    else:
        print(f"[FAIL] Test 1.3: Sarah Garcia Birth Year: {res[0]['content'][:100] if res else 'EMPTY'}")

    # Test 1.4: Highest Level of Education (PhD in Political Science)
    total += 1
    res = search_hybrid(user_id=uid_sarah, query_text="What is Sarah Garcia's highest level of education?", top_k=5)
    t4_pass = any("PhD" in r["content"] and "Political Science" in r["content"] for r in res[:2])
    if t4_pass:
        passed += 1
        print("[PASS] Test 1.4: Sarah Garcia Highest Education is PhD in Political Science")
    else:
        print(f"[FAIL] Test 1.4: Sarah Garcia Education: {res[0]['content'][:100] if res else 'EMPTY'}")

    # 2. Portal CEO 4-Hop Knowledge Graph Chain
    uid_portal = "u_6d83a63d06759a274dd0e684856d5d5993267e99e47035d3fd6d04ea946e81a7"
    total += 1
    q_portal = "What is the official language of the country where the chief executive officer of the creator of Portal is a citizen?"
    res = search_hybrid(user_id=uid_portal, query_text=q_portal, top_k=5)
    t5_pass = any("Latvian" in r["content"] or "Latvia" in r["content"] for r in res[:2])
    if t5_pass:
        passed += 1
        print("[PASS] Test 2.1: Portal CEO 4-hop chain resolves to Latvian / Latvia in Top 2")
    else:
        print(f"[FAIL] Test 2.1: Portal CEO chain: {res[0]['content'][:100] if res else 'EMPTY'}")

    # 3. Multimodal Memory Track Tests
    uid_mm = "u_2f0c15696ded85e604a91707890d45f61eadddbc7bb30f60e0d77e3bd4f5e942"
    uid_coffee = "u_840a735ecf94f93803e431729578fca6183b38912cd092fc7abd09717a783ca9"

    # Test 3.1: Coffee Maker Spend
    total += 1
    q_coffee = 'How much total have I spent on coffee makers?\nAnswer with the exact amount (e.g., "$45.00") only.'
    res = search_hybrid(user_id=uid_coffee, query_text=q_coffee, top_k=5)
    t6_pass = res and "$150.00" in res[0]["content"] and res[0]["content"].startswith("$150.00.")
    if t6_pass:
        passed += 1
        print("[PASS] Test 3.1: Coffee Maker spend returns leading $150.00.")
    else:
        print(f"[FAIL] Test 3.1: Coffee Maker spend: {res[0]['content'][:100] if res else 'EMPTY'}")

    # Test 3.2: Engagement Night Two Steps
    total += 1
    q_eng = "订婚纪念夜那晚，我先后做了哪两步，才慢慢把职业焦虑想清楚？"
    res = search_hybrid(user_id=uid_mm, query_text=q_eng, top_k=5)
    t7_pass = len(res) == 2 and "第一步" in res[0]["content"] and "第二步" in res[1]["content"] and "职业焦虑" in res[0]["content"] and "棘手项目" in res[1]["content"]
    if t7_pass:
        passed += 1
        print("[PASS] Test 3.2: Engagement Night returns strictly the 2 clean steps")
    else:
        print(f"[FAIL] Test 3.2: Engagement Night: {[r['content'][:50] for r in res]}")

    # Test 3.3: In-domain Epistemic Abstention (Chinese)
    total += 1
    q_abs = "2024年春节后，我和舅妈在西湖边散步时讨论了什么？"
    res = search_hybrid(user_id=uid_mm, query_text=q_abs, top_k=5)
    t8_pass = len(res) == 1 and "未提及" in res[0]["content"]
    if t8_pass:
        passed += 1
        print("[PASS] Test 3.3: Multimodal abstention query returns unmentioned notice")
    else:
        print(f"[FAIL] Test 3.3: Multimodal abstention: {res[0]['content'][:100] if res else 'EMPTY'}")

    # 4. Coding Memory Track Protection Test
    uid_code = "u_4baa948d7012613933a71ff3098b7e4c964233bc312ed6939fa9e2ceb3d3800e"
    total += 1
    q_code = "rels_to_update rel_db_params rel_collation old_rel_db_params _alter_column_collation_sql schema.py fix"
    res = search_hybrid(user_id=uid_code, query_text=q_code, top_k=5)
    t9_pass = len(res) <= 4 and any("schema.py" in r["content"] or "diff --git" in r["content"] for r in res)
    if t9_pass:
        passed += 1
        print("[PASS] Test 4.1: Coding search returns capped clean patch context without regression")
    else:
        print(f"[FAIL] Test 4.1: Coding search: len={len(res)}, contents={[r['content'][:60] for r in res]}")

    # 5. Persona Constraints Cross-Contamination Test
    total += 1
    q_lunch = "What dietary requirements or restrictions should we keep in mind for planning the team lunch?"
    res = search_hybrid(user_id=uid_sarah, query_text=q_lunch, top_k=5)
    t10_pass = not any("python" in r["content"].lower() or "programming" in r["content"].lower() for r in res[:3])
    if t10_pass:
        passed += 1
        print("[PASS] Test 5.1: Persona food/diet query strictly excludes programming language distractors")
    else:
        print(f"[FAIL] Test 5.1: Persona query polluted: {[r['content'][:60] for r in res]}")

    print("=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
