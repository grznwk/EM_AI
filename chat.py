import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./db_index")
collection = chroma_client.get_collection(name="my_docs")

print("\n--- System RAG (v2026) gotowy ---")

while True:
    query = input("\nTwoje pytanie: ")
    if query.lower() in ['exit', 'quit']: break

    # 1. Zamień pytanie na wektor
    query_resp = client.models.embed_content(
        #model="text-embedding-004", # do tego nie mamy prawa
        model="models/gemini-embedding-001",
        contents=query
    )
    query_embedding = query_resp.embeddings[0].values

    # 2. Znajdź 3 najlepsze fragmenty w lokalnej bazie
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    context = "\n\n".join(results['documents'][0])

    # 3. Wyślij tylko fragmenty do Gemini
    prompt = f"Odpowiedz na pytanie na podstawie tych fragmentów:\n{context}\n\nPytanie: {query}"
    print(f"\nAI: {prompt}")
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    print(f"\nAI: {response.text}")
