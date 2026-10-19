# Production RAG from Scratch (No Frameworks)

A production-grade Retrieval-Augmented Generation (RAG) system built from absolute first principles. It intentionally avoids third-party orchestrators like LangChain or LlamaIndex to implement the foundational algorithms behind modern AI search, hybrid retrieval, and automated evaluation.

---

## 🚀 Key Features

### 1. Hybrid Search (Dense Vectors + Sparse BM25)
* **Bare-Metal Vector Search:** Computes cosine similarity across dense embeddings generated locally by `nomic-embed-text` via Ollama.
* **BM25 Keyword Search:** Uses the `rank_bm25` algorithm to match exact keywords, technical terms (e.g. `AES-256`), and product codes that vector embeddings can overlook.
* **Reciprocal Rank Fusion (RRF):** Fuses ranking positions from both algorithms into a single balanced score using the industry standard formula:
  $$\text{RRF Score} = \sum \frac{1}{60 + \text{rank}}$$

### 2. Automated LLM-as-a-Judge Evaluation
* **Golden Dataset (`eval_dataset.json`):** A curated test suite of reference questions, ground-truth answers, and source context excerpts.
* **Faithfulness Metric (0.0 to 1.0):** Evaluates whether the generated answer contains any claims not supported by the retrieved context (hallucination detection).
* **Answer Relevancy Metric (0.0 to 1.0):** Evaluates whether the answer directly and specifically answers the user's prompt without evasion or filler.
* **Quality Gate:** Fails with exit code `1` if either metric drops below `0.80`.

### 3. CI/CD Quality Gate (GitHub Actions)
* `.github/workflows/rag-eval.yml` runs automated regression evaluations on every `git push` or pull request to prevent regressions before merging.

---

## 📂 Project Architecture

```text
├── .github/workflows/
│   └── rag-eval.yml         # GitHub Actions automated CI evaluation gate
├── techflow_report.pdf      # Sample fact-dense enterprise report
├── rag_core.py              # Core engine: loading, chunking, embeddings, BM25 & RRF
├── prompts.py               # Grounded answer templates & Judge evaluation prompts
├── main.py                  # Interactive CLI assistant with hybrid search
├── evaluate.py              # Automated LLM-as-a-judge evaluation harness
├── eval_dataset.json        # Golden benchmark dataset
└── requirements.txt         # Dependencies (numpy, PyPDF2, requests, rank_bm25)
```

---

## 🛠️ Setup & Usage

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) running locally.

### 2. Pull Required Models
```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

### 3. Install Dependencies
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Interactive Chat
Run the interactive assistant:
```bash
python main.py
```

### 5. Run Automated Evaluation
Run the LLM-as-a-Judge benchmark:
```bash
# Quick benchmark (first 3 questions)
python -u evaluate.py --limit 3

# Full evaluation suite (all questions)
python -u evaluate.py
```

Sample output:
```text
============================================================
EVALUATION SCOREBOARD
============================================================
#   | Faithfulness | Relevancy  | Question
------------------------------------------------------------
1   | 1.00         | 1.00       | Who founded TechFlow Solutions and ...
2   | 1.00         | 1.00       | What was TechFlow's annual recurrin...
3   | 1.00         | 1.00       | How much does the Starter plan for ...
------------------------------------------------------------
Average Faithfulness: 1.00 / 1.00
Average Relevancy:    1.00 / 1.00
============================================================

[PASS] System meets production threshold (>= 0.80).
```

---

## 🧠 What This Demonstrates
1. **Understanding Under-the-Hood Mechanics:** No black-box frameworks; vector math, chunking strategies, and BM25 tokenization are transparently implemented.
2. **Solving the Vector Blindspot:** Implementing Hybrid Search with Reciprocal Rank Fusion to combine semantic nuance with exact keyword precision.
3. **Continuous Quality & Hallucination Prevention:** Measuring pipeline health mathematically with an automated LLM judge and CI gates.
