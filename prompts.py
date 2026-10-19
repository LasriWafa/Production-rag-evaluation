RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question directly and concisely in 1-2 sentences based ONLY on the provided context. If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{query}

Answer:"""

JUDGE_FAITHFULNESS_TEMPLATE = """You are an unbiased AI evaluation judge.
Evaluate the FAITHFULNESS of the answer against the given context.
Faithfulness measures whether all claims in the answer are strictly supported by the context without any hallucinations.

Context:
{context}

Answer:
{answer}

Respond strictly with a JSON object in this format:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means 100% faithful and 0.0 means completely hallucinated>,
  "reason": "<short 1-sentence reason>"
}}"""

JUDGE_RELEVANCY_TEMPLATE = """You are an unbiased AI evaluation judge.
Evaluate the RELEVANCY of the answer to the user's question.
Relevancy measures whether the answer directly and specifically addresses what was asked.

Question:
{question}

Answer:
{answer}

Respond strictly with a JSON object in this format:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means directly answers the question and 0.0 means completely irrelevant>,
  "reason": "<short 1-sentence reason>"
}}"""
