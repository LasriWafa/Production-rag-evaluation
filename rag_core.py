import PyPDF2
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from prompts import RAG_PROMPT_TEMPLATE

# --- Step 1: Loading ---
def load_pdf(file_path):
    """Reads a PDF file and extracts all its text into a single string."""
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- Step 2: Chunking ---
def chunk_text(text, chunk_size=500, overlap=100):
    """Splits a large text into smaller overlapping chunks so the AI can process them."""
    start = 0
    chunks = []
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Step 3 & 4: Embeddings and Search ---
def get_embedding(text):
    """Converts a piece of text into a vector (a list of numbers) using Ollama."""
    url = "http://localhost:11434/api/embed"
    response = requests.post(url, json={"model": "nomic-embed-text", "input": text})
    return response.json()["embeddings"][0]

def cosine_similarity(vec1, vec2):
    """Calculates how mathematically similar two vectors are (1.0 = identical, 0.0 = unrelated)."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def vector_search(query, chunks, embeddings, top_k=3):
    """Finds the top chunks using semantic vector similarity."""
    query_emb = get_embedding(query)
    scores = []
    
    # Compare the query against every chunk
    for chunk_emb in embeddings:
        score = cosine_similarity(query_emb, chunk_emb)
        scores.append(score)
        
    # Get the indices of the top scoring chunks
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Backwards compatibility
search = vector_search

# --- BM25 Keyword Search ---
def create_bm25_index(chunks):
    """Tokenizes chunks into lowercase words and builds a BM25 index."""
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)

def bm25_search(query, chunks, bm25_index, top_k=3):
    """Finds top chunks based on exact keyword matches."""
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# --- Hybrid Search (RRF: Reciprocal Rank Fusion) ---
def hybrid_search(query, chunks, embeddings, bm25_index, top_k=3, k=60):
    """Combines semantic vector search and BM25 keyword search using Reciprocal Rank Fusion."""
    # 1. Rank chunks with Vector Search (semantic similarity)
    query_emb = get_embedding(query)
    vec_scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
    vec_ranking = np.argsort(vec_scores)[::-1]
    
    # 2. Rank chunks with BM25 (exact keyword match)
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranking = np.argsort(bm25_scores)[::-1]
    
    # 3. Calculate RRF score for each chunk: 1 / (k + rank)
    rrf_scores = {}
    
    for rank, idx in enumerate(vec_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
        
    for rank, idx in enumerate(bm25_ranking):
        # Only reward chunks that actually matched at least one keyword
        if bm25_scores[idx] > 0:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
            
    # 4. Sort chunks by highest RRF score and return the top_k
    sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
    top_indices = sorted_indices[:top_k]
    return [chunks[i] for i in top_indices]

# --- Step 5: LLM Generation ---
def generate_answer(query, context_chunks):
    """Generates an answer using the LLM, restricting it to ONLY use the provided context chunks."""
    url = "http://localhost:11434/api/generate"
    context_text = "\n\n".join(context_chunks)
    
    # Inject the context and query into our prompt template
    prompt = RAG_PROMPT_TEMPLATE.format(context=context_text, query=query)

    response = requests.post(url, json={
        "model": "qwen3:8b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 250,
            "num_thread": 8
        }
    }, timeout=180)
    
    return response.json()["response"]
