import chromadb
import os

# 1. Podłącz się do bazy
chroma_client = chromadb.PersistentClient(path="./db_index")
collection = chroma_client.get_collection(name="my_docs")

# 2. Podaj nazwę pliku, który chcesz usunąć (musi być taka sama jak przy dodawaniu)
#file_to_remove = "stara_ksiazka.pdf"
file_to_remove = "KsJN_book.pdf"

# 3. Usuń wszystkie fragmenty, które w metadanych mają tę nazwę
try:
    collection.delete(where={"source": file_to_remove})
    print(f"Sukces! Dokument '{file_to_remove}' został usunięty z bazy.")
except Exception as e:
    print(f"Błąd podczas usuwania: {e}")
