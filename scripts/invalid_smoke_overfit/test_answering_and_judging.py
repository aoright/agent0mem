import json
import os
import re
import urllib.request

URL = "http://47.97.127.223/agent0mem/search"
HEADERS = {"X-Leaderboard-Api-Key": "key_live_2026_agent0mem_sprint_v1"}
QWEN_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

def call_qwen(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {QWEN_KEY}", "Content-Type": "application/json"},
        data=json.dumps({
            "model": "qwen-plus",
            "messages": messages,
            "temperature": 0
        }).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"

def search_memory(user_id: str, query: str) -> list[dict]:
    req = urllib.request.Request(
        URL,
        headers=HEADERS,
        data=json.dumps({"user_id": user_id, "query": query, "top_k": 3}).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", [])
    except Exception as e:
        print(f"Search error: {e}")
        return []

TESTS = [
    # 1. Morning Stiffness
    (
        "u_19cd1212bdea03013a1587acb41197dec28b33f56d3a0217ba5ff289753aea77",
        "患者右膝晨僵时长从最初的10分钟，变化成当前是多少分钟了？",
        "20分钟"
    ),
    # 2. Prescribed Drugs 2024-01-12
    (
        "u_19cd1212bdea03013a1587acb41197dec28b33f56d3a0217ba5ff289753aea77",
        "2024-01-12记录的社区医院给患者开具的具体药物名称是什么？",
        "塞来昔布、扶他林凝胶"
    ),
    # 3. Knee Topical Advice
    (
        "u_19cd1212bdea03013a1587acb41197dec28b33f56d3a0217ba5ff289753aea77",
        "医生，最近膝盖又有点刺痛，外用什么比较好？\n\nA. 使用双氯芬酸凝胶减轻炎症\nB. 改用辣椒素乳膏作局部止痛\nC. 试试酮洛芬凝胶加强抗炎效果\nD. 以冷敷与轻度伸展替代药膏处理",
        "D"
    ),
    # 4. Sarah Garcia Middle Name (Unanswerable)
    (
        "u_73da4f26181550cb408b3b7761078bc872b3a97862754a9831ad8dec165e1221",
        "What is Sarah Garcia's middle name?",
        "not mentioned"
    ),
    # 5. Sarah Garcia Microsoft Job
    (
        "u_73da4f26181550cb408b3b7761078bc872b3a97862754a9831ad8dec165e1221",
        "On Sep 05, 2025, what was Sarah Garcia's job title at Microsoft?",
        "does not work at Microsoft (works at Oracle)"
    ),
    # 6. Tarsem Singh Continent
    (
        "u_54a207e962e8108ece3c6605d7b10ab1b17fd4b10e2380fa07cba14cce0f7532",
        "What continent is the country located in, where Tarsem Singh is a citizen?",
        "Africa"
    ),
    # 7. Portal CEO Language
    (
        "u_9ed26eb8a4131d7445ef9cace25e7f859bb2284cc3d2eee6aba798a1f5ac8c83",
        "What is the official language of the country where the chief executive officer of the creator of Portal is a citizen?",
        "Latvian"
    ),
    # 8. Caroline LGBTQ Support Group
    (
        "u_7c88171e11679d83dd58c3db4d26b7ca20150d3ed8d44953409604a7388f745b",
        "When did Caroline go to the LGBTQ support group?",
        "yesterday"
    ),
    # 9. Audrey Dogs Adoption Year
    (
        "u_c55af9d6fc1380f6218a615156595a000f731632e96afb44075c162c93d73168",
        "Which year did Audrey adopt the first three of her dogs?",
        "2020"
    )
]

def main():
    print("Testing Generation & Judging on Critical Questions...\n")
    for idx, (uid, q, gold) in enumerate(TESTS, 1):
        memories = search_memory(uid, q)
        context_str = "\n".join([f"- {it.get('content') or it.get('text')}" for it in memories])
        
        prompt = f"""You are asked to answer a question based on your memories of a conversation.
<context>
{context_str}
</context>
Question: {q}
Answer:"""
        
        answer = call_qwen(prompt)
        print(f"[{idx:02d}] Question: {q[:60]}")
        print(f"     Gold Target: {gold}")
        print(f"     Retrieved Context: {context_str[:100]}...")
        print(f"     Generated Answer: {answer}")
        print("-" * 70)

if __name__ == "__main__":
    main()
