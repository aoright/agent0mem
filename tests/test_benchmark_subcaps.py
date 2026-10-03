import os
import json
import urllib.request

QWEN_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

def call_qwen(prompt):
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization": f"Bearer {QWEN_KEY}", "Content-Type": "application/json"},
        data=json.dumps({"model": "qwen-plus", "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode()
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip()

# Test Audrey
prompt_audrey = """You are asked to answer a question based on your memories of a conversation.

<instructions>
10. Keep the final answer minimal. Do not add explanation, background, or extra dates unless needed for correctness.
</instructions>

<memories>
Memories for user speaker 1:

- Audrey adopted the first three of her dogs in the year 2020.
</memories>

Question: Which year did Audrey adopt the first three of her dogs?
Answer with the shortest correct phrase or sentence. No preamble, no fluff:"""
ans_audrey = call_qwen(prompt_audrey)
print("Audrey ans:", ans_audrey)

judge_audrey = f"""Your task is to label an answer as 'CORRECT' or 'WRONG' given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR<->HOUR, DAY<->DAY, MONTH<->MONTH, YEAR<->YEAR.

Question: Which year did Audrey adopt the first three of her dogs?
Gold answer: 2020
Generated answer: {ans_audrey}

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```"""
print("Audrey judge:", call_qwen(judge_audrey))

# Test Dune Messiah
prompt_dune = """You are an assistant that MUST answer questions using ONLY the information provided in the context below. 

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Do NOT use your internal knowledge

CONTEXT:
<context>
The author of Dune Messiah is Frank Herbert. Frank Herbert plays funk music (the genre of music played by the author of Dune Messiah is funk).
</context>

QUESTION:
<question>
What type of music does the author of Dune Messiah play?
</question>

ANSWER REQUIREMENTS:
- Be direct and concise
- Only output the answer to the question without any explanation 

RESPONSE:
"""
ans_dune = call_qwen(prompt_dune)
print("Dune ans:", ans_dune)

judge_dune = f"""You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): What type of music does the author of Dune Messiah play?
- RUBRIC CRITERION (what to check): The response must state that Frank Herbert (or the author of Dune Messiah) plays funk music.
- RESPONSE TO EVALUATE: {ans_dune}

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:
{{
   "score": 1.0,
   "reason": "..."
}}
"""
print("Dune judge:", call_qwen(judge_dune))
