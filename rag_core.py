import PyPDF2
import numpy as np
import requests
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

def search(query, chunks, embeddings, top_k=3):
    """Finds the top chunks that are most relevant to the user's query."""
    query_emb = get_embedding(query)
    scores = []
    
    # Compare the query against every chunk
    for chunk_emb in embeddings:
        score = cosine_similarity(query_emb, chunk_emb)
        scores.append(score)
        
    # Get the indices of the top scoring chunks
    top_indices = np.argsort(scores)[-top_k:][::-1]
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
        "stream": False
    })
    
    return response.json()["response"]
