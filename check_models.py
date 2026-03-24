import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("--- DIAGNOSTYKA MODELI (SDK 2.0) ---")
try:
    # Pobieramy listę modeli
    models = client.models.list()

    print("Dostępne modele i ich akcje:")
    for m in models:
        # Sprawdzamy czy model wspiera tworzenie wektorów (embeddings)
        if 'embedContent' in m.supported_actions:
            print(f"MOŻESZ UŻYĆ -> {m.name}")

except Exception as e:
    print(f"Błąd: {e}")
    # Jeśli powyższe zawiedzie, wypiszmy wszystko co ma model, żeby zobaczyć strukturę
    print("\nPróba awaryjna (lista wszystkich nazw):")
    for m in client.models.list():
        print(f"- {m.name}")
