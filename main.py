from rag_core import load_pdf, chunk_text, get_embedding, search, generate_answer

def main():
    # 1. Setup Phase: Prepare the document database
    print("1. Loading PDF...")
    text = load_pdf("techflow_report.pdf")
    
    print("2. Chunking text...")
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    
    print(f"3. Embedding {len(chunks)} chunks (this takes a few seconds)...")
    embeddings = []
    for chunk in chunks:
        embeddings.append(get_embedding(chunk))
        
    # 2. Interactive Phase: Chat with the user
    print("\nSystem Ready! Ask a question about TechFlow Solutions.")
    print("(Type 'quit' to exit)\n")
    
    while True:
        query = input("You: ")
        if query.lower() in ["quit", "exit"]:
            break
            
        print("Searching...")
        # Find the most relevant pieces of text for this specific question
        relevant_chunks = search(query, chunks, embeddings)
        
        print("Thinking...")
        # Generate a final answer using ONLY those relevant pieces
        answer = generate_answer(query, relevant_chunks)
        
        print(f"\nAI: {answer}\n")
        
        # Display the chunks used so the user can verify the AI isn't hallucinating
        print("-" * 50)
        print("Chunks used for this answer:")
        for i, c in enumerate(relevant_chunks):
            print(f"[{i+1}] {c[:75]}...")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
