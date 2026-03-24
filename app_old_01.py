# Uruchomienie: streamlit run app.py

import streamlit as st
import chromadb
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Konfiguracja bazy
chroma_client = chromadb.PersistentClient(path="./db_index")
collection = chroma_client.get_collection(name="my_docs")

st.set_page_config(page_title="Mój AI Asystent",
    page_icon="🤖",
    layout="wide"  # Tutaj ustawiasz szerokie okno
    )
st.title("🤖 Mój Prywatny Asystent RAG")

# Inicjalizacja historii czatu w pamięci sesji
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetlanie historii
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Pole wpisywania
if prompt := st.chat_input("Zadaj pytanie dotyczące dokumentów..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Logika RAG
    with st.spinner("Szukam w dokumentach..."):
        query_resp = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=prompt
        )
        query_emb = query_resp.embeddings[0].values

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=5 # Zwiększ do 5 lub 10, jeśli chcesz, by AI czytało więcej stron naraz
        )
        context = "\n---\n".join(results['documents'][0])
        sources = set(m['source'] for m in results['metadatas'][0])

        # full_prompt = f"Kontekst:\n{context}\n\nPytanie: {prompt}"
        # response = client.models.generate_content(
        #     model="gemini-2.5-flash-lite",
        #     contents=full_prompt
        # )

        # 1. Budujemy aktualny prompt na podstawie kontekstu z bazy
        full_prompt = f"Kontekst z dokumentów:\n{context}\n\nUżyj powyższego kontekstu, aby odpowiedzieć na pytanie: {prompt}"

        # 2. WINDOW MEMORY: Pobieramy tylko X ostatnich wiadomości z sesji
        # -10 oznacza 5 ostatnich pytań i 5 ostatnich odpowiedzi
        memory_window = st.session_state.messages[-10:]

        # 3. Dodajemy aktualny "full_prompt" jako ostatni element do wysłania
        # Ale nie dodajemy go do st.session_state, żeby nie zaśmiecać widoku użytkownika
        messages_to_send = []
        for msg in memory_window:
            messages_to_send.append({"role": msg["role"], "parts": [msg["content"]]})

        # Na końcu dodajemy naszą aktualną wiadomość z kontekstem RAG
        messages_to_send.append({"role": "user", "parts": [full_prompt]})

        # 4. Wysyłamy do modelu
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite", # (lub gemini-1.5-flash)
            contents=messages_to_send
        )

        answer = response.text + f"\n\n**Źródła:** {', '.join(sources)}"

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
