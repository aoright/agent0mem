import json
import re
import requests
import sys

URL = "http://47.97.127.223/agent0mem/search"
HEADERS = {"X-Leaderboard-Api-Key": "key_live_2026_agent0mem_sprint_v1"}

CRITERIA = {
    # 1. LongMemEval
    "When Jon has lost his job as a banker?": {
        "required": ["January 19, 2023"],
        "forbidden": [],
        "capability": "A"
    },
    "Which year did Audrey adopt the first three of her dogs?": {
        "required": ["2020"],
        "forbidden": ["3 years"],
        "capability": "A"
    },
    # 2. LoCoMo
    "When did Caroline go to the LGBTQ support group?": {
        "required": ["yesterday"],
        "forbidden": ["2024", "January"],
        "capability": "C"
    },
    "What fields would Caroline be likely to pursue in her education?": {
        "required": ["counseling", "mental health"],
        "forbidden": [],
        "capability": "C"
    },
    "Which political party did Joaquim Molins belong to in Jan, 1976?": {
        "required": ["Catalan Centre"],
        "forbidden": [],
        "capability": "B"
    },
    # 3. Clinical
    "2024-01-12记录的社区医院给患者开具的具体药物名称是什么？": {
        "required": ["塞来昔布", "扶他林凝胶"],
        "forbidden": ["200mg", "一天一颗"],
        "capability": "C"
    },
    "患者这次INR复查是在什么时候进行的？": {
        "required": ["半个月前"],
        "forbidden": ["2024", "1月中旬"],
        "capability": "C"
    },
    "患者右膝晨僵时长从最初的10分钟，变化成当前是多少分钟了？": {
        "required": ["15分钟"],
        "forbidden": [],
        "capability": "C"
    },
    "医生，最近膝盖又有点刺痛，外用什么比较好？\n\nA. 使用双氯芬酸凝胶减轻炎症\nB. 改用辣椒素乳膏作局部止痛\nC. 试试酮洛芬凝胶加强抗炎效果\nD. 以冷敷与轻度伸展替代药膏处理": {
        "required": ["D", "冷敷"],
        "forbidden": [],
        "capability": "C"
    },
    # 4. PersonaMem (Sarah Garcia)
    "What is Sarah Garcia's highest education level as of September 4, 2025?": {
        "required": ["PhD in Political Science"],
        "forbidden": [],
        "capability": "D"
    },
    "What is Sarah Garcia's birth date in 1985?": {
        "required": ["1986", "not in 1985"],
        "forbidden": [],
        "capability": "D"
    },
    "What is Sarah Garcia's job title as of September 5, 2025?": {
        "required": ["Manager at Oracle"],
        "forbidden": [],
        "capability": "D"
    },
    "What is Sarah Garcia's middle name?": {
        "required": ["not mentioned", "cannot be determined"],
        "forbidden": [],
        "capability": "D"
    },
    "On Sep 05, 2025, what was Sarah Garcia's job title at Microsoft?": {
        "required": ["not work at Microsoft", "Oracle"],
        "forbidden": [],
        "capability": "D"
    },
    "What is Sarah Garcia's middle name as of Sep 06, 2025?": {
        "required": ["not mentioned", "cannot be determined"],
        "forbidden": [],
        "capability": "D"
    },
    # 5. Multi-Hop
    "What continent is the country located in, where Tarsem Singh is a citizen?": {
        "required": ["Africa"],
        "forbidden": [],
        "capability": "B"
    },
    "What continent contains the country where Tarsem Singh is a citizen?": {
        "required": ["Africa"],
        "forbidden": [],
        "capability": "B"
    },
    "On which continent is the country located, whose citizen is Tarsem Singh?": {
        "required": ["Africa"],
        "forbidden": [],
        "capability": "B"
    },
    "What is the official language of the country where the chief executive officer of the creator of Portal is a citizen?": {
        "required": ["Latvian"],
        "forbidden": [],
        "capability": "B"
    },
    "Which language is officially spoken in the country of citizenship of the CEO of the company that created Portal?": {
        "required": ["Latvian"],
        "forbidden": [],
        "capability": "B"
    },
    "What is the official language of the country that the CEO of the company that created Portal is a citizen of?": {
        "required": ["Latvian"],
        "forbidden": [],
        "capability": "B"
    },
    # 6. ScriptMem
    "A courtroom demonstration includes someone miming the act of unfastening a chain and opening a door. Given the described": {
        "required": ["Option B", "chain"],
        "forbidden": [],
        "capability": "G"
    },
    "Taken together, Billing’s repeated oath-heavy outbursts most strongly suggest what underlying shift in his stance toward": {
        "required": ["Option B", "moral condemnation"],
        "forbidden": [],
        "capability": "G"
    },
    "What does the camera’s movement and the crowding into the elevator most strongly imply about how the scene is framing th": {
        "required": ["elevator", "claustrophobic"],
        "forbidden": [],
        "capability": "G"
    },
    "What does the Foreman’s behavior most strongly indicate about his role in the group’s decision-making at that moment?\n\nA": {
        "required": ["facilitator", "moderator"],
        "forbidden": [],
        "capability": "G"
    },
    # 7. Science / BEAM
    "What type of music does the author of Dune Messiah play?": {
        "required": ["funk"],
        "forbidden": [],
        "capability": "A"
    },
    "Which genre of music is played by the author of Dune Messiah?": {
        "required": ["funk"],
        "forbidden": [],
        "capability": "A"
    },
    "How would you describe the overall OTC uptake rate exhibited on MIL-53(Al)@RH as? the ave": {
        "required": ["chemisorption", "uptake"],
        "forbidden": [],
        "capability": "H"
    },
    "Reach 24 using the numbers [5, 5, 5, 8].": {
        "required": ["impossible", "cannot"],
        "forbidden": [],
        "capability": "H"
    },
    "Can you list the order in which I brought up different aspects of implementing": {
        "required": ["1. Initial project scope", "2. Weather data service", "3. API key"],
        "forbidden": [],
        "capability": "H"
    },
    # 8. Refusals & Streaming
    "NTSB preliminary excerpt:\nAt 0816 local, an experimental amateur-built, two-seat, low-wing airplane departed RWY 27 on": {
        "required": ["no record", "CFD aerodynamic simulations"],
        "forbidden": [],
        "capability": "D"
    },
    "DB task index 289": {
        "required": ["DB", "reward"],
        "forbidden": [],
        "capability": "streaming"
    },
    "DB task index 214": {
        "required": ["DB", "reward"],
        "forbidden": [],
        "capability": "streaming"
    },
    "ALF task index 86": {
        "required": ["ALF", "desklamp"],
        "forbidden": [],
        "capability": "streaming"
    },
    "ALF task index 96": {
        "required": ["ALF", "desklamp"],
        "forbidden": [],
        "capability": "streaming"
    }
}

def main():
    with open('/tmp/smoke_v10_queries.json') as f:
        records = json.load(f)

    seen = {}
    for r in records:
        q = r['query']
        uid = r['user_id']
        if q not in seen:
            seen[q] = uid

    passed = 0
    total = 0
    failures = []

    print(f"Executing Rigorous Pessimistic Evaluation across {len(seen)} queries...\n")

    for q, uid in seen.items():
        matched_crit = None
        for key, crit in CRITERIA.items():
            if key in q or q in key or key[:40] in q:
                matched_crit = crit
                break
        
        if not matched_crit:
            # General query, check return status 200
            total += 1
            payload = {"user_id": uid, "query": q, "top_k": 5}
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=10)
            if r.status_code == 200 and len(r.json().get("data", [])) > 0:
                passed += 1
            else:
                failures.append((q[:60], "HTTP error or empty data"))
            continue

        total += 1
        payload = {"user_id": uid, "query": q, "top_k": 5}
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=10)
        if r.status_code != 200:
            failures.append((q[:60], f"HTTP {r.status_code}"))
            continue
        data = r.json().get("data", [])
        if not data:
            failures.append((q[:60], "Empty data returned"))
            continue

        rank1 = data[0].get("content") or data[0].get("text") or ""
        
        # Check required tokens
        req_missing = [req for req in matched_crit["required"] if req.lower() not in rank1.lower()]
        forb_present = [forb for forb in matched_crit["forbidden"] if forb.lower() in rank1.lower()]

        if req_missing or forb_present:
            failures.append((q[:60], f"Missing: {req_missing}, Forbidden: {forb_present}, Rank 1: {rank1[:80]}"))
        else:
            passed += 1

    pessimistic_score = (passed / total) * 100.0
    print("=" * 70)
    print(f"OFFLINE PESSIMISTIC EVALUATION RESULT: {passed}/{total} ({pessimistic_score:.2f}%)")
    print("=" * 70)
    
    if failures:
        print(f"\n{len(failures)} FAILURES DETECTED:")
        for q_sub, reason in failures:
            print(f"  - Query: {q_sub}")
            print(f"    Reason: {reason}")
        sys.exit(1)
    else:
        print("\nALL TEST CASES PASSED WITH 100.0% PESSIMISTIC COMPLIANCE!")
        sys.exit(0)

if __name__ == "__main__":
    main()
