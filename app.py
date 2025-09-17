import streamlit as st
import investpy
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="WKN RoboAdvisor", layout="wide")
st.title("📊 WKN RoboAdvisor – Trendfolge-Ranking")

# Eingabe-Feld für WKNs
wkn_input = st.text_area("WKNs (eine pro Zeile):", value="A0YJQ2\n870747\nA2JG9Z")
wkn_list = [wkn.strip().upper() for wkn in wkn_input.splitlines() if wkn.strip()]

def wkn_to_ticker(wkn):
    try:
        result = investpy.search_quotes(by='wkn', value=wkn, products=['stocks'], countries=['germany'])
        return result[0].symbol, result[0].name
    except:
        return None, None

def calculate_indicators(ticker):
    try:
        df = yf.download(ticker, period="300d", interval="1d", progress=False)
        if df.empty or len(df) < 260:
            return None, None, None, None
        close = df["Close"]

        gd200 = ((close[-1] - close.rolling(200).mean().iloc[-1]) / close.rolling(200).mean().iloc[-1]) * 100
        gd130 = ((close[-1] - close.rolling(130).mean().iloc[-1]) / close.rolling(130).mean().iloc[-1]) * 100
        mom260 = ((close[-1] - close[-260]) / close[-260]) * 100
        momjt = close.pct_change(20).mean() * 100

        return round(gd200, 2), round(gd130, 2), round(mom260, 2), round(momjt, 2)
    except:
        return None, None, None, None

# Button zum Starten
if st.button("🔄 Aktualisieren"):
    results = []

    for wkn in wkn_list:
        ticker, name = wkn_to_ticker(wkn)
        if not ticker:
            results.append({
                "WKN": wkn,
                "Name": "Ticker nicht gefunden",
                "GD200": None,
                "GD130": None,
                "MOM260": None,
                "MOMJT": None,
                "Score": 0
            })
            continue

        gd200, gd130, mom260, momjt = calculate_indicators(ticker)
        valid_indicators = [v for v in [gd200, gd130, mom260, momjt] if v is not None]
        score = round(sum(valid_indicators) / len(valid_indicators), 2) if valid_indicators else 0

        results.append({
            "WKN": wkn,
            "Name": name,
            "GD200": gd200,
            "GD130": gd130,
            "MOM260": mom260,
            "MOMJT": momjt,
            "Score": score
        })

    df = pd.DataFrame(results)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

else:
    st.info("Gib WKNs ein und klicke auf 🔄 Aktualisieren.")
