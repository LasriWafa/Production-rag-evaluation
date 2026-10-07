# RAG from Scratch (No Frameworks)

This project is a complete Retrieval-Augmented Generation (RAG) pipeline built from absolute first principles. It intentionally avoids frameworks like LangChain or LlamaIndex to demonstrate a deep, ground-up understanding of how AI search and retrieval systems actually work under the hood.

## 🚀 Features

- **Custom PDF Parsing:** Extracts text directly from a test document.
- **Manual Sliding-Window Chunking:** Splits text into overlapping semantic blocks to preserve context.
- **Local Embeddings:** Converts text chunks into vector embeddings using `nomic-embed-text` via Ollama.
- **Bare-Metal Vector Search:** Calculates cosine similarity by hand using `numpy` to rank and retrieve the most relevant chunks.
- **Strict Grounding:** Uses custom prompt engineering to force the LLM to answer *only* from the retrieved context, effectively preventing hallucination.
- **Observability:** Prints the exact chunks used for generation so the user can verify the retrieval step.

## 📂 Project Structure

This project follows a strict separation of concerns:

- `main.py` — The interactive chat loop and entry point.
- `rag_core.py` — The engine (loading, chunking, embedding, cosine similarity, and API calls).
- `prompts.py` — Isolated prompt templates for easy iteration.
- `techflow_report.pdf` — A fictional, fact-dense company report used to test retrieval accuracy.

## 🛠️ Setup and Installation

1. **Prerequisites:** 
   - Python 3.9+
   - [Ollama](https://ollama.com/) installed and running locally.

2. **Pull Required Local Models:**
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen3:8b
   ```
   *(Note: You can swap `qwen3:8b` for `llama3` or any other model in `rag_core.py`)*

3. **Create a Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the Project:**
   ```bash
   python main.py
   ```

## 🧠 Why Build This?

Many engineers glue together LangChain components without understanding the underlying mechanics. By building the entire pipeline from scratch, this project demonstrates:
- How chunk size and overlap actually affect retrieval.
- How vector math (cosine similarity) acts as a search engine.
- How to separate the "Retrieval" failure modes from the "Generation" failure modes.
- How to enforce strict LLM grounding.
