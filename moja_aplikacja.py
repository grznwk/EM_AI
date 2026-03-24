# streamlit run moja_aplikacja.py

import streamlit as st
import pandas as pd
import numpy as np

# Nagłówek aplikacji
st.title("Witaj w świecie Streamlit! 🚀")

# Prosty tekst
st.write("To jest moja pierwsza aplikacja działająca w przeglądarce.")

# Suwak interaktywny
liczba_punktow = st.slider("Ile punktów na mapie wygenerować?", 10, 500, 100)

# Generowanie losowych danych (symulacja współrzędnych)
dane_mapy = pd.DataFrame(
    np.random.randn(liczba_punktow, 2) / [50, 50] + [52.23, 21.01], # Warszawa
    columns=['lat', 'lon']
)

# Wyświetlenie mapy
st.map(dane_mapy)

st.success("Mapa zaktualizowana!")
