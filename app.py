import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 WKN RoboAdvisor", layout="wide")
st.title("📊 WKN RoboAdvisor – Trendfolge Ranking")

st.markdown("""
Gib unten eine Liste von WKNs ein (eine pro Zeile). Die App berechnet automatisch:
- Abstand zur 200-Tage-Linie (GD200)
- Abstand zur 130-Tage-Linie (GD130 / Relative Stärke)
- Momentum über 260 Tage (MOM260)
- Momentum-Indikator nach Jegadeesh/Titman (MOMJT)
""")

# Beispiel-WKNs (können geändert werden)
default_wkns = """A1CX3T
A2QK1G
881103
A3DRAS
A0HGFA
A3D40L
A2QA4S
A3D8PW
A0Q4DC
A0YJQ2"""

# Session State, damit die WKNs gespeichert bleiben
if 'wkn_input' not in st.session_state:
    st.session_state.wkn_input = default_wkns

# Eingabefeld
wkn_input = st.text_area("✍️ WKN-Liste (eine pro Zeile)", st.session_state.wkn_input, height=200)
wkn_list = [w.strip() for w in wkn_input.split("\n") if w.strip()]
st.session_state.wkn_input = wkn_input

def calculate_indicators(ticker):
    try:
        data = yf.download(ticker, period="300d")
        if len(data) < 260:
            return None  # Nicht genug Daten

        data['Close'] = data['Adj Close']

        # GD130, GD200
        gd130 = data['Close'].rolling(window=130).mean()
        gd200 = data['Close'].rolling(window=200).mean()

        latest_close = data['Close'].iloc[-1]
        last_gd130 = gd130.iloc[-1]
        last_gd200 = gd200.iloc[-1]

        dist_gd130 = (latest_close - last_gd130) / last_gd130 * 100
        dist_gd200 = (latest_close - last_gd200) / last_gd200 * 100

        # MOM260 (Time Series Momentum)
        mom260 = (latest_close / data['Close'].iloc[-260] - 1) * 100

        # MOMJT: Durchschnittliches Momentum der letzten 6 Monate
        six_months = 6
        monthly_returns = []
        for i in range(1, six_months + 1):
            past_day = -21 * i
            ret = (latest_close / data['Close'].iloc[past_day] - 1) * 100
            monthly_returns.append(ret)
        momjt = np.mean(monthly_returns)

        return round(dist_gd200, 2), round(dist_gd130, 2), round(mom260, 2), round(momjt, 2)
    except Exception as e:
        return None

# Button zur Auswertung
if st.button("🔄 Aktualisieren"):
    result = []
    for wkn in wkn_list:
        stock = yf.Ticker(wkn)
        name = stock.info.get("shortName", wkn)
        indicators = calculate_indicators(wkn)
        if indicators:
            gd200, gd130, mom260, momjt = indicators
            score = gd200 + gd130 + mom260 + momjt  # Summe als einfaches Ranking-Kriterium
            result.append({
                "Wert": name,
                "WKN": wkn,
                "GD200": gd200,
                "GD130": gd130,
                "MOM260": mom260,
                "MOMJT": momjt,
                "Gesamt": round(score, 2)
            })
        else:
            result.append({
                "Wert": f"{wkn} (Keine Daten)",
                "WKN": wkn,
                "GD200": None,
                "GD130": None,
                "MOM260": None,
                "MOMJT": None,
                "Gesamt": None
            })

    df = pd.DataFrame(result)
    df = df.sort_values(by="Gesamt", ascending=False)
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
