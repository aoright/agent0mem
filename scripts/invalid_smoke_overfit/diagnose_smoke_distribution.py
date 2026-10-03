import requests
import json

BASE_URL = "http://47.97.127.223/agent0mem/search"

# We have the exact pairs from the smoke runs:
TEST_GROUPS = {
    "Cap_A_and_B_LongMemEval": [
        ("u_e6f0101b3a4c7e37ee13ee9235f9db9999cd49b4335713b0987575987e55d49d", "When Jon has lost his job as a banker?"),
        ("u_c55af9d6fc1380f6218a615156595a000f731632e96afb44075c162c93d73168", "Which year did Audrey adopt the first three of her dogs?"),
    ],
    "Cap_C_LoCoMo": [
        ("u_7c88171e11679d83dd58c3db4d26b7ca20150d3ed8d44953409604a7388f745b", "When did Caroline go to the LGBTQ support group?"),
        ("u_7c88171e11679d83dd58c3db4d26b7ca20150d3ed8d44953409604a7388f745b", "What fields would Caroline be likely to pursue in her education?"),
        ("u_22f7de59d5858623e8827ffd690466ae5709099c691439a4ef44d756b9c7b63b", "Which political party did Joaquim Molins belong to in Jan, 1976?"),
        ("u_19cd1212bdea03013a1587acb41197dec28b33f56d3a0217ba5ff289753aea77", "2024-01-12记录的社区医院给患者开具的具体药物名称是什么？"),
        ("u_19cd1212bdea03013a1587acb41197dec28b33f56d3a0217ba5ff289753aea77", "患者这次INR复查是在什么时候进行的？"),
        ("u_19cd1212bdea03013a1587acb41197dec28b33f56d3a0217ba5ff289753aea77", "患者右膝晨僵时长从最初的10分钟，变化成当前是多少分钟了？"),
        ("u_19cd1212bdea03013a1587acb41197dec28b33f56d3a0217ba5ff289753aea77", "医生，最近膝盖又有点刺痛，外用什么比较好？\n\nA. 使用双氯芬酸凝胶减轻炎症\nB. 改用辣椒素乳膏作局部止痛\nC. 试试酮洛芬凝胶加强抗炎效果\nD. 以冷敷与轻度伸展替代药膏处理"),
    ],
    "Cap_D_Persona_and_MQuAKE": [
        ("u_73da4f26181550cb408b3b7761078bc872b3a97862754a9831ad8dec165e1221", "What is Sarah Garcia's highest education level as of September 4, 2025?"),
        ("u_73da4f26181550cb408b3b7761078bc872b3a97862754a9831ad8dec165e1221", "What is Sarah Garcia's birth date in 1985?"),
        ("u_73da4f26181550cb408b3b7761078bc872b3a97862754a9831ad8dec165e1221", "What is Sarah Garcia's job title as of September 5, 2025?"),
        ("u_73da4f26181550cb408b3b7761078bc872b3a97862754a9831ad8dec165e1221", "What is Sarah Garcia's middle name?"),
        ("u_73da4f26181550cb408b3b7761078bc872b3a97862754a9831ad8dec165e1221", "On Sep 05, 2025, what was Sarah Garcia's job title at Microsoft?"),
        ("u_73da4f26181550cb408b3b7761078bc872b3a97862754a9831ad8dec165e1221", "What is Sarah Garcia's middle name as of Sep 06, 2025?"),
        ("u_7e49b8d493f84d8723c3ac8784385b85a6b81ad459f0372e3115e100f1ebb148", "What continent is the country located in, where Tarsem Singh is a citizen?"),
        ("u_7e49b8d493f84d8723c3ac8784385b85a6b81ad459f0372e3115e100f1ebb148", "What continent contains the country where Tarsem Singh is a citizen?"),
        ("u_9ed26eb8a4131d7445ef9cace25e7f859bb2284cc3d2eee6aba798a1f5ac8c83", "What is the official language of the country where the chief executive officer of the creator of Portal is a citizen?"),
    ],
    "Cap_E_PersonaMem_Care": [
        ("u_f540f14c587c25c646499ea5550ac7c5c07cbe99fcae1e3d4454f8d03bc36c50", "What kind of warm-up should I do before a casual game of pickup basketball?"),
        ("u_f540f14c587c25c646499ea5550ac7c5c07cbe99fcae1e3d4454f8d03bc36c50", "What are some ways to make my weekend reading time feel more relaxing and immersive?"),
    ],
    "Cap_G_ScriptMem": [
        ("u_fd7be1f34bbd01200a3e7eeb99c7daa0cd87009205fcefa46004d464d40014de", "What does the camera’s movement and the crowding into the elevator most strongly imply about how the scene is framing th"),
        ("u_fd7be1f34bbd01200a3e7eeb99c7daa0cd87009205fcefa46004d464d40014de", "A courtroom demonstration includes someone miming the act of unfastening a chain and opening a door. Given the described"),
        ("u_fd7be1f34bbd01200a3e7eeb99c7daa0cd87009205fcefa46004d464d40014de", "What does the Foreman’s behavior most strongly indicate about his role in the group’s decision-making at that moment?\n\nA"),
        ("u_c934d27904b7b7c3aa1b5685b44a3d24e3695e0214da2ccee2795ee6789f0afe", "Taken together, Billing’s repeated oath-heavy outbursts most strongly suggest what underlying shift in his stance toward"),
    ],
    "Cap_H_BEAM": [
        ("u_5f9e5cb41bc8a837b0bdb317f6231e9316c3d9f88e3084178d0b792bf54ea645", "What type of music does the author of Dune Messiah play?"),
        ("u_5c031e02f0e421cc6389df2c0df3fd6bc51851c1122641a5b285de83675897b6", "How would you describe the overall OTC uptake rate exhibited on MIL-53(Al)@RH as? the ave"),
    ],
    "MemTrap": [
        ("u_700d6d0c98ceae07d7d71780a0c588ec013036c81f2d8bb568e44d37fcdc610e", "Reach 24 using the numbers [5, 5, 5, 8]."),
    ],
    "CLBench": [
        ("u_b9582c89d1142bc58ca5d7d9f6effd8ac8b8bbbcbc23765d8cf5bbe27772df65", "Can you list the order in which I brought up different aspects of implementing"),
        ("u_b9582c89d1142bc58ca5d7d9f6effd8ac8b8bbbcbc23765d8cf5bbe27772df65", "Have I obtained an API key for this project?"),
    ],
    "Streaming": [
        ("u_4efcda4240c3444fddb136b116a2ac40b1764965980f234aa0e33cb06f1afd47", "DB task index 289"),
        ("u_4efcda4240c3444fddb136b116a2ac40b1764965980f234aa0e33cb06f1afd47", "DB task index 214"),
        ("u_c8c92d3776b981f7c58f9de3d4de02f044d1cf53db4d7c2001244016d39e7423", "ALF task index 86"),
        ("u_c8c92d3776b981f7c58f9de3d4de02f044d1cf53db4d7c2001244016d39e7423", "ALF task index 96"),
    ]
}

def main():
    for group, cases in TEST_GROUPS.items():
        print(f"\n{'='*70}\nGROUP: {group} ({len(cases)} items)\n{'='*70}")
        for uid, q in cases:
            try:
                r = requests.post(BASE_URL, json={"user_id": uid, "query": q}, timeout=10)
                if r.status_code != 200:
                    print(f"HTTP {r.status_code} for user {uid[:12]}: {q[:50]}")
                    continue
                data = r.json().get("data", [])
                print(f"\nUser {uid[:12]} | Items: {len(data)}")
                print(f"Query: {q[:90]}")
                if not data:
                    print("  [NO ITEMS RETURNED!]")
                else:
                    for i, it in enumerate(data[:2], 1):
                        content = it.get("content", "").replace("\n", " ")
                        score = it.get("score", 0.0)
                        print(f"  Rank {i} ({score:.4f}): {content[:130]}")
            except Exception as e:
                print(f"ERROR on {uid[:12]}: {e}")

if __name__ == "__main__":
    main()
