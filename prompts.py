RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{query}

Answer:"""
