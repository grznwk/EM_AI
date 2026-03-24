# Uruchomienie: streamlit run app.py
# Local LAN: streamlit run app.py --server.address 0.0.0.0
# START_CZAT.bat
# @echo off
# streamlit run app.py --server.address 0.0.0.0
# pause
# 192.168.0.84

import streamlit as st
import chromadb
from google import genai
from dotenv import load_dotenv
import os
import time
from google.genai import types

# Importujemy funkcję z Twojego ingest.py
from ingest import run_ingestion

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Konfiguracja ścieżek
DOCS_DIR = "dokumenty" # Upewnij się, że to nazwa Twojego folderu z PDFami
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# Konfiguracja bazy
chroma_client = chromadb.PersistentClient(path="./db_index")
collection = chroma_client.get_or_create_collection(name="my_docs")

st.set_page_config(page_title="AI Ekspert Dokumentów", layout="wide")

# --- SIDEBAR: PANEL STEROWANIA ---
with st.sidebar:
    st.title("⚙️ Zarządzanie")

    # 1. UPLOAD PLIKÓW
    st.subheader("📤 Wyślij nowy dokument")
    uploaded_file = st.file_uploader("Wybierz plik z dysku", type=["pdf", "docx", "txt", "xlsx"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("💾 Zapisz i indeksuj", use_container_width=True):
            with st.spinner("Zapisywanie pliku..."):
                # Zapis fizyczny pliku w folderze dokumenty
                file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"Plik {uploaded_file.name} zapisany!")

                # Automatyczne wywołanie ingestion po uploadzie
                with st.spinner("Aktualizacja bazy wiedzy..."):
                    run_ingestion()
                    st.success("Baza zaktualizowana!")
                    time.sleep(1)
                    st.rerun()

    st.divider()

    # 2. Przycisk Ingestion (Ręczny)
    if st.button("🔄 Synchronizuj folder", use_container_width=True):
        with st.spinner("Aktualizacja bazy..."):
            run_ingestion()
            st.success("Zsynchronizowano!")
            st.rerun()

    st.divider()

    # 3. Zarządzanie pojedynczymi plikami
    st.subheader("📚 Pliki w bazie")
    try:
        results = collection.get(include=['metadatas'])
        if results['metadatas']:
            unique_sources = sorted(list(set(m['source'] for m in results['metadatas'])))

            for src in unique_sources:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.markdown(f"""
                        <div style="
                            white-space: nowrap;
                            overflow: hidden;
                            text-overflow: ellipsis;
                            width: 100%;
                            padding-top: 5px;
                            font-size: 0.85rem;
                        " title="{src}">
                            📄 {src}
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️", key=f"del_{src}", help=f"Usuń {src} z bazy"):
                        collection.delete(where={"source": src})
                        # Opcjonalnie: odkomentuj poniższą linię, jeśli chcesz usuwać też plik z dysku
                        # if os.path.exists(os.path.join(DOCS_DIR, src)): os.remove(os.path.join(DOCS_DIR, src))
                        st.warning(f"Usunięto {src}")
                        time.sleep(1)
                        st.rerun()
        else:
            st.info("Baza jest pusta.")
    except Exception as e:
        st.error(f"Błąd bazy: {e}")

    st.divider()

    msg_count = len(st.session_state.get("messages", []))
    st.info(f"Wiadomości w historii: {msg_count}")

    if st.button("🧹 Wyczyść czat"):
        st.session_state.messages = []
        st.rerun()

# --- GŁÓWNE OKNO CZATU ---
st.title("🤖 Czat z Wiedzą o ETIM-Mapper")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Zadaj pytanie..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analizuję dokumenty..."):
        query_resp = client.models.embed_content(model="models/gemini-embedding-001", contents=prompt)
        query_emb = query_resp.embeddings[0].values
        results = collection.query(query_embeddings=[query_emb], n_results=5)

        context = "\n---\n".join(results['documents'][0])
        sources = set(m['source'] for m in results['metadatas'][0])

        enriched_prompt = f"POTRZEBNA WIEDZA Z PLIKÓW:\n{context}\n\nPYTANIE UŻYTKOWNIKA: {prompt}"

        try:
            history_for_api = []
            for m in st.session_state.messages[:-1]:
                history_for_api.append({
                    "role": "user" if m["role"] == "user" else "model",
                    "parts": [{"text": m["content"]}]
                })

            chat = client.chats.create(model="gemini-2.5-flash-lite", history=history_for_api)
            response = chat.send_message(enriched_prompt)

            answer = response.text + f"\n\n**Źródła:** {', '.join(sources)}"

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Wystąpił błąd podczas generowania odpowiedzi: {e}")
            print(f"DEBUG ERROR: {e}")
