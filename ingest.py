import os
import time
import hashlib
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from docx import Document
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Konfiguracja
DOCS_FOLDER = "dokumenty"
DB_PATH = "./db_index"
COLLECTION_NAME = "my_docs"
MODEL_NAME = "models/gemini-embedding-001"

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def extract_text(file_path):
    """Rozpoznaje format pliku i wyciąga z niego tekst."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                content = page.extract_text()
                if content: text += content + "\n"

        elif ext == '.docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

    except Exception as e:
        print(f"Błąd podczas czytania pliku {file_path}: {e}")

    return text

def run_ingestion():
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"Utworzono folder '{DOCS_FOLDER}'.")

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # Obsługiwane rozszerzenia
    supported_ext = ('.pdf', '.docx', '.txt')
    all_files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(supported_ext)]

    if not all_files:
        print("Brak obsługiwanych plików w folderze 'dokumenty'.")
        return

    for file_name in all_files:
        file_path = os.path.join(DOCS_FOLDER, file_name)
        current_hash = get_file_hash(file_path)

        existing_data = collection.get(where={"source": file_name}, include=['metadatas'])

        if existing_data['metadatas']:
            stored_hash = existing_data['metadatas'][0].get('hash')
            if stored_hash == current_hash:
                print(f"--- Aktualny: {file_name} ---")
                continue
            else:
                print(f"--- Aktualizacja: {file_name} ---")
                collection.delete(where={"source": file_name})

        print(f"Przetwarzanie: {file_name}...")
        full_text = extract_text(file_path)

        if not full_text.strip():
            print(f"  Pominięto: {file_name} (brak tekstu).")
            continue

        # Dzielenie na fragmenty
        chunk_size = 1000
        overlap = 200
        chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size - overlap)]

        for i, chunk in enumerate(chunks):
            try:
                response = client.models.embed_content(
                    model=MODEL_NAME,
                    contents=chunk
                )
                embedding = response.embeddings[0].values

                collection.add(
                    ids=[f"{file_name}_{i}_{current_hash[:8]}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": file_name, "hash": current_hash}]
                )

                if i % 10 == 0:
                    print(f"  Postęp: {i}/{len(chunks)}...")

                time.sleep(0.2)

            except Exception as e:
                print(f"  Błąd w {file_name} (fragment {i}): {e}")

    print("\nSynchronizacja zakończona pomyślnie.")

if __name__ == "__main__":
    run_ingestion()
