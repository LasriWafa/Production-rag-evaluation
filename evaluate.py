import json
import sys
import argparse
import requests

# Ensure UTF-8 output encoding on Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rag_core import (
    load_pdf, 
    chunk_text, 
    get_embedding, 
    create_bm25_index, 
    hybrid_search, 
    generate_answer
)
from prompts import JUDGE_FAITHFULNESS_TEMPLATE, JUDGE_RELEVANCY_TEMPLATE

def judge_metric(prompt_template, model="qwen3:8b", **kwargs):
    """Asks the LLM to act as an evaluator and return a numerical score (0.0 to 1.0) with an explanation."""
    prompt = prompt_template.format(**kwargs)
    try:
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 120,
                    "num_thread": 8
                }
            }, 
            timeout=180
        )
        data = json.loads(response.json()["response"])
        score = float(data.get("score", 0.0))
        reason = data.get("reason", "")
        return score, reason
    except Exception as e:
        return 0.0, f"Judge error: {e}"

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation Harness")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test questions to evaluate (e.g. --limit 3)")
    parser.add_argument("--model", type=str, default="qwen3:8b", help="Model to use for generation and evaluation (default: qwen3:8b)")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print(f"RAG Evaluation Harness -- LLM as a Judge (Model: {args.model})", flush=True)
    print("=" * 60, flush=True)

    # 1. Load the Golden Dataset
    print("\n[1/4] Loading Golden Dataset (eval_dataset.json)...", flush=True)
    with open("eval_dataset.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if args.limit:
        test_cases = test_cases[:args.limit]
        print(f"Loaded first {len(test_cases)} test case(s) (--limit {args.limit}).", flush=True)
    else:
        print(f"Loaded all {len(test_cases)} test cases.", flush=True)

    # 2. Prepare the RAG system index
    print("\n[2/4] Initializing RAG Pipeline (Embedding & BM25)...", flush=True)
    pdf_text = load_pdf("techflow_report.pdf")
    chunks = chunk_text(pdf_text, chunk_size=500, overlap=100)
    
    print(f"Embedding {len(chunks)} chunks with Ollama...", flush=True)
    embeddings = [get_embedding(c) for c in chunks]
    bm25_index = create_bm25_index(chunks)
    print("Pipeline ready.", flush=True)

    # 3. Evaluate each test question
    print("\n[3/4] Running Questions through RAG & Evaluating with Judge...", flush=True)
    results = []

    for i, test in enumerate(test_cases, start=1):
        question = test["question"]
        print(f"\n--- Test #{i}/{len(test_cases)}: \"{question}\" ---", flush=True)
        
        # Retrieve context using Hybrid Search
        retrieved_chunks = hybrid_search(question, chunks, embeddings, bm25_index, top_k=3)
        context_text = "\n\n".join(retrieved_chunks)
        
        # Generate the answer
        print("  Generating answer...", flush=True)
        answer = generate_answer(question, retrieved_chunks, model=args.model)
        print(f"  Answer: {answer.strip()}", flush=True)

        # Judge Faithfulness (No hallucinations)
        print("  Evaluating Faithfulness...", flush=True)
        faith_score, faith_reason = judge_metric(
            JUDGE_FAITHFULNESS_TEMPLATE, 
            model=args.model,
            context=context_text, 
            answer=answer
        )
        print(f"  - Faithfulness: {faith_score:.2f} ({faith_reason})", flush=True)

        # Judge Relevancy (Directly answers the question)
        print("  Evaluating Relevancy...", flush=True)
        rel_score, rel_reason = judge_metric(
            JUDGE_RELEVANCY_TEMPLATE, 
            model=args.model,
            question=question, 
            answer=answer
        )
        print(f"  - Relevancy:    {rel_score:.2f} ({rel_reason})", flush=True)

        results.append({
            "question": question,
            "faithfulness": faith_score,
            "relevancy": rel_score
        })

    # 4. Generate Summary Report
    print("\n" + "=" * 60, flush=True)
    print("EVALUATION SCOREBOARD", flush=True)
    print("=" * 60, flush=True)
    print(f"{'#':<3} | {'Faithfulness':<12} | {'Relevancy':<10} | Question", flush=True)
    print("-" * 60, flush=True)
    for i, r in enumerate(results, start=1):
        q_short = (r['question'][:35] + '...') if len(r['question']) > 35 else r['question']
        print(f"{i:<3} | {r['faithfulness']:<12.2f} | {r['relevancy']:<10.2f} | {q_short}", flush=True)
    print("-" * 60, flush=True)

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["relevancy"] for r in results) / len(results)

    print(f"Average Faithfulness: {avg_faith:.2f} / 1.00", flush=True)
    print(f"Average Relevancy:    {avg_rel:.2f} / 1.00", flush=True)
    print("=" * 60, flush=True)

    # Threshold Gate (Step 5 requirement: Fail if < 0.80)
    THRESHOLD = 0.80
    if avg_faith >= THRESHOLD and avg_rel >= THRESHOLD:
        print(f"\n[PASS] System meets production threshold (>= {THRESHOLD:.2f}).", flush=True)
        sys.exit(0)
    else:
        print(f"\n[FAIL] Scores are below quality threshold (>= {THRESHOLD:.2f}).", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
