import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. Podłączamy się do bazy, którą właśnie stworzyłeś
chroma_client = chromadb.PersistentClient(path="./db_index")
collection = chroma_client.get_collection(name="my_docs")

print("\n--- SYSTEM RAG GOTOWY ---")
print("Zadaj pytanie dotyczące swojej książki (wpisz 'exit' aby zakończyć):")

while True:
    query = input("\nTy: ")
    if query.lower() in ['exit', 'quit']: break

    try:
        # 2. Zamieniamy pytanie na wektor tym samym modelem co w ingest.py
        query_resp = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=query
        )
        query_embedding = query_resp.embeddings[0].values

        # 3. Szukamy 3 najbardziej pasujących fragmentów w Twojej bazie na dysku
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # Łączymy znalezione fragmenty w jeden kontekst
        context = "\n\n---\n\n".join(results['documents'][0])

        # 4. Wysyłamy do Gemini TYLKO te fragmenty (oszczędność danych!)
        prompt = f"""Jesteś pomocnym asystentem. Odpowiedz na pytanie użytkownika,
        korzystając WYŁĄCZNIE z poniższych fragmentów dokumentu.
        Jeśli odpowiedzi nie ma w tekście, powiedz szczerze, że nie wiesz.

        KONTEKST:
        {context}

        PYTANIE: {query}"""

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )

        print(f"\nAI: {response.text}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
