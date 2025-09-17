import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

st.set_page_config(page_title="📈 Momentum-Satelliten", layout="wide")
st.title("📊 Analyse – Satellitenwerte")

# --- Eingabe-Möglichkeiten ---
uploaded_file = st.file_uploader("📂 Lade eine CSV-Datei mit Tickers hoch", type="csv")
tickers_input = st.text_input(
    "Oder gib Ticker (Yahoo Finance) ein, getrennt durch Komma:",
    value="APP, LEU, XMTR, RHM.DE"
)

# --- Ticker-Liste vorbereiten ---
if uploaded_file:
    df_tickers = pd.read_csv(uploaded_file)
    if "Ticker" not in df_tickers.columns:
        st.error("❌ CSV muss eine Spalte 'Ticker' enthalten!")
        ticker_list = []
    else:
        ticker_list = df_tickers["Ticker"].dropna().tolist()
else:
    ticker_list = [t.strip() for t in tickers_input.split(",") if t.strip()]

# --- Button zum Start ---
if st.button("🔄 Aktualisieren") and ticker_list:
    end = datetime.today()
    start = end - timedelta(days=400)

    results = []

    for ticker in ticker_list:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                st.warning(f"⚠️ Keine Daten für {ticker} geladen.")
                results.append([ticker, None, None, None, None, None, None])
                continue

            # Fallback: Adj Close oder Close
            if "Adj Close" in data.columns:
                data["Kurs
