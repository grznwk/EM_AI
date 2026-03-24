import os
import time
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def create_vector_db():
    print("1. Wczytywanie i dzielenie PDF...")
    reader = PdfReader("C:\\SD\\Python\\MyAI\\dokumenty\\KsJN_book.pdf")
    full_text = "".join([page.extract_text() for page in reader.pages])

    chunk_size = 1000
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    print(f"2. Generowanie wektorów dla {len(chunks)} fragmentów...")

    chroma_client = chromadb.PersistentClient(path="./db_index")
    collection = chroma_client.get_or_create_collection(name="my_docs")

    for i, chunk in enumerate(chunks):
        try:
            # UŻYWAMY DOKŁADNIE TEJ NAZWY, KTÓRĄ ZWRÓCIŁA DIAGNOSTYKA
            response = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=chunk
            )
            embedding = response.embeddings[0].values

            collection.add(
                ids=[f"{file_name}_{i}"], # Lepsze ID: nazwa_pliku_numer
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": file_name}] # TU DODAJEMY "ETYKIETĘ"
            )

            if i % 20 == 0:
                print(f"Przetworzono {i}/{len(chunks)}...")

            # Mała przerwa, żeby nie przekroczyć limitów darmowego API
            time.sleep(0.2)

        except Exception as e:
            print(f"Błąd przy fragmencie {i}: {e}")
            continue

    print("\nSUKCES! Baza 'db_index' jest gotowa i pełna danych.")

if __name__ == "__main__":
    create_vector_db()
