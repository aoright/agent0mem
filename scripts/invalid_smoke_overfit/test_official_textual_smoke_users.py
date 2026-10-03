import requests
import json

BASE_URL = "http://47.97.127.223/agent0mem/search"

TEST_CASES = [
    # Sarah Garcia (Capability D / Persona)
    ("u_383782f7dd98e075c8315c5f70e936777ac18df9437670aebd930e1aad86f75f", "What is Sarah Garcia's highest education level as of September 4, 2025?"),
    ("u_383782f7dd98e075c8315c5f70e936777ac18df9437670aebd930e1aad86f75f", "What is Sarah Garcia's birth date in 1985?"),
    ("u_383782f7dd98e075c8315c5f70e936777ac18df9437670aebd930e1aad86f75f", "What is Sarah Garcia's job title as of September 5, 2025?"),
    ("u_383782f7dd98e075c8315c5f70e936777ac18df9437670aebd930e1aad86f75f", "What is Sarah Garcia's middle name?"),
    ("u_383782f7dd98e075c8315c5f70e936777ac18df9437670aebd930e1aad86f75f", "On Sep 05, 2025, what was Sarah Garcia's job title at Microsoft?"),
    ("u_383782f7dd98e075c8315c5f70e936777ac18df9437670aebd930e1aad86f75f", "What is Sarah Garcia's middle name as of Sep 06, 2025?"),
    
    # Caroline (LoCoMo / Capability C)
    ("u_91c4444ee636575e8be2148d565f6759d31515c97bfdd66e84904c68e63c038f", "When did Caroline go to the LGBTQ support group?"),
    ("u_91c4444ee636575e8be2148d565f6759d31515c97bfdd66e84904c68e63c038f", "What fields would Caroline be likely to pursue in her education?"),
    
    # Patient (Capability C & D)
    ("u_78940a7b736991fe3fcf67c6a8372726171322979494c96edc814a24fedc55da", "2024-01-12记录的社区医院给患者开具的具体药物名称是什么？"),
    ("u_78940a7b736991fe3fcf67c6a8372726171322979494c96edc814a24fedc55da", "患者这次INR复查是在什么时候进行的？"),
    ("u_78940a7b736991fe3fcf67c6a8372726171322979494c96edc814a24fedc55da", "患者右膝晨僵时长从最初的10分钟，变化成当前是多少分钟了？"),
    ("u_78940a7b736991fe3fcf67c6a8372726171322979494c96edc814a24fedc55da", "医生，最近膝盖又有点刺痛，外用什么比较好？\n\nA. 使用双氯芬酸凝胶减轻炎症\nB. 改用辣椒素乳膏作局部止痛\nC. 试试酮洛芬凝胶加强抗炎效果\nD. 以冷敷与轻度伸展替代药膏处理"),
    
    # Joaquim Molins (Capability C)
    ("u_d74866faf1598df5170ea2abbf58a47e61ee8b31312669b59b06d154750b2262", "Which political party did Joaquim Molins belong to in Jan, 1976?"),
    
    # Tarsem Singh / MQuAKE (Capability D Multi-hop)
    ("u_be98860efa73822dd4b54c4f65d356ee35176fe8843b6444deedbe3144a6ed2c", "What continent is the country located in, where Tarsem Singh is a citizen?"),
    ("u_be98860efa73822dd4b54c4f65d356ee35176fe8843b6444deedbe3144a6ed2c", "What continent contains the country where Tarsem Singh is a citizen?"),
    
    # Portal CEO (Capability D Multi-hop)
    ("u_e89a5216c6851bb347b58bafd1858e7092603d3ceff566ed74c2161f416e4fb6", "What is the official language of the country where the chief executive officer of the creator of Portal is a citizen?"),
    
    # MemTrap / Reach 24
    ("u_ac4d15134306065f61e7864e29eeda95035975d0803281eb50b28a627028aa8c", "Reach 24 using the numbers [5, 5, 5, 8]."),
    
    # Streaming Tasks
    ("u_17c9d0653761fc44ebd13e51bd9fad343cdd7e9dab9c7f82c07fc8a1bd8d68d6", "DB task index 289"),
    ("u_17c9d0653761fc44ebd13e51bd9fad343cdd7e9dab9c7f82c07fc8a1bd8d68d6", "DB task index 214"),
    ("u_eeff3623ccd7111fd537bbb00ef5882fabbb56ec7d3c9e3f9c53ad8442380b8f", "ALF task index 86"),
    ("u_eeff3623ccd7111fd537bbb00ef5882fabbb56ec7d3c9e3f9c53ad8442380b8f", "ALF task index 96"),
]

def main():
    print("Testing Official Textual Smoke Cases against Live Endpoint...")
    for idx, (uid, query) in enumerate(TEST_CASES, 1):
        try:
            r = requests.post(BASE_URL, json={"user_id": uid, "query": query}, timeout=10)
            if r.status_code != 200:
                print(f"[{idx:02d}] HTTP {r.status_code} for user {uid[:12]}: {query[:60]}")
                continue
            data = r.json().get("data", [])
            print(f"\n[{idx:02d}] User: {uid[:12]} | Items: {len(data)}")
            print(f"Query: {query[:100]}")
            if not data:
                print("  NO ITEMS RETURNED (POSSIBLE UNMENTIONED NOTICE OR SUPPRESSION)")
            else:
                for i, it in enumerate(data[:2], 1):
                    content = it.get("content", "")
                    score = it.get("score", 0.0)
                    print(f"  Rank {i} (score: {score:.4f}): {content[:140]}")
        except Exception as e:
            print(f"[{idx:02d}] ERROR: {str(e)}")

if __name__ == "__main__":
    main()
