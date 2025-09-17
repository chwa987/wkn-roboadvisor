import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="WKN RoboAdvisor", layout="wide")

st.title("📈 WKN RoboAdvisor – Trendfolgebewertung")

# Eingabe der WKNs
wkn_input = st.text_area("WKNs (eine pro Zeile):", value="A2QA4S\nA3D8PW\nA0Q4DC\nA0YJQ2")
wkn_list = [w.strip().upper() for w in wkn_input.splitlines() if w.strip()]

# Button zum Aktualisieren
if st.button("🔄 Aktualisieren"):

    def get_ticker_from_wkn(wkn):
        try:
            # Schnelle Yahoo-Finance-Suche über das WKN-Symbol
            ticker = yf.Ticker(wkn)
            info = ticker.info
            symbol = info.get("symbol", None)
            name = info.get("shortName", f"{wkn} (Kein Name gefunden)")
            return symbol, name
        except:
            return None, f"{wkn} (Keine Daten)"

    def calculate_indicators(ticker):
        try:
            df = yf.download(ticker, period="300d", interval="1d", progress=False)

            if df.empty or len(df) < 260:
                return None, None, None, None

            close = df["Close"]

            gd200 = ((close[-1] - close.rolling(200).mean().iloc[-1]) / close.rolling(200).mean().iloc[-1]) * 100
            gd130 = ((close[-1] - close.rolling(130).mean().iloc[-1]) / close.rolling(130).mean().iloc[-1]) * 100
            mom260 = ((close[-1] - close[-260]) / close[-260]) * 100
            momjt = close.pct_change(20).mean() * 100  # Durchschnittliches Monatsmomentum

            return round(gd200, 2), round(gd130, 2), round(mom260, 2), round(momjt, 2)

        except:
            return None, None, None, None

    results = []

    for wkn in wkn_list:
        symbol, name = get_ticker_from_wkn(wkn)
        if not symbol:
            results.append({"Wert": name, "WKN": wkn, "GD200": None, "GD130": None, "MOM260": None, "MOMJT": None, "Score": 0})
            continue

        gd200, gd130, mom260, momjt = calculate_indicators(symbol)

        # Scoring-Schema (kann angepasst werden)
        score = sum(v is not None for v in [gd200, gd130, mom260, momjt])

        results.append({
            "Wert": name,
            "WKN": wkn,
            "GD200": gd200,
            "GD130": gd130,
            "MOM260": mom260,
            "MOMJT": momjt,
            "Score": score
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Score", ascending=False).reset_index(drop=True)

    st.dataframe(df_results, use_container_width=True)
    st.success("Daten aktualisiert.")

else:
    st.info("Gib WKNs ein und klicke auf 'Aktualisieren', um die Daten zu laden.")
