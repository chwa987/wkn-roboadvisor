import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Warnungen (FutureWarning etc.) ausblenden
warnings.simplefilter(action="ignore", category=FutureWarning)

st.set_page_config(page_title="📈 Momentum-Satelliten", layout="wide")
st.title("📊 Analyse – Satellitenwerte")

# Feste Ticker-Liste
ticker_list = {
    "AppLovin": "APP",
    "Centrus Energy": "LEU",
    "Xometry": "XMTR",
    "Rheinmetall": "RHM.DE"
}

if st.button("🔄 Aktualisieren"):
    end = datetime.today()
    start = end - timedelta(days=400)

    results = []

    for name, ticker in ticker_list.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                st.warning(f"⚠️ Keine Daten für {name} ({ticker}) geladen.")
                results.append([name, None, None, None, None, None])
                continue

            # Fallback: nutze Adj Close wenn vorhanden, sonst Close
            if "Adj Close" in data.columns:
                data["Kurs"] = data["Adj Close"]
            else:
                data["Kurs"] = data["Close"]

            # Gleitende Durchschnitte
            data["GD200"] =
