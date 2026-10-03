import sys
sys.path.insert(0, ".")
from app.database import search_hybrid

def test_queries():
    cases = [
        ("u_383782f7dd98e075c8315c5f70e936777ac18df9437670aebd930e1aad86f75f", "What is Sarah Garcia's highest education level as of September 4, 2025?"),
        ("u_383782f7dd98e075c8315c5f70e936777ac18df9437670aebd930e1aad86f75f", "What is Sarah Garcia's job title as of September 5, 2025?"),
        ("u_78940a7b736991fe3fcf67c6a8372726171322979494c96edc814a24fedc55da", "2024-01-12记录的社区医院给患者开具的具体药物名称是什么？"),
        ("u_78940a7b736991fe3fcf67c6a8372726171322979494c96edc814a24fedc55da", "患者右膝晨僵时长从最初的10分钟，变化成当前是多少分钟了？"),
        ("u_78940a7b736991fe3fcf67c6a8372726171322979494c96edc814a24fedc55da", "医生，最近膝盖又有点刺痛，外用什么比较好？\n\nA. 使用双氯芬酸凝胶减轻炎症\nB. 改用辣椒素乳膏作局部止痛\nC. 试试酮洛芬凝胶加强抗炎效果\nD. 以冷敷与轻度伸展替代药膏处理"),
    ]
    for uid, q in cases:
        print("=" * 60)
        print("Query:", q[:80])
        res = search_hybrid(user_id=uid, query_text=q, top_k=3)
        print("Items returned:", len(res))
        for i, it in enumerate(res):
            score = it.get("score", 0.0)
            content = it.get("content", "")
            print(f"  [{i+1}] (score={score:.4f}) {content[:140]}")

if __name__ == "__main__":
    test_queries()
