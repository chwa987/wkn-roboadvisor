import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="📈 WKN RoboAdvisor", layout="wide")
st.title("📊 WKN RoboAdvisor – Trendfolge Ranking")

st.markdown("""
Gib unten eine Liste von Ticker-Symbolen (eine pro Zeile) ein. Die App berechnet automatisch:
- Abstand zur 200-Tage-Linie (GD200)
- Abstand zur 130-Tage-Linie (GD130 / Relative Stärke)
- Momentum über 260 Tage (MOM260)
- Momentum-Indikator nach Jegadeesh/Titman (MOMJT)
""")

# Beispiel-Ticker (können geändert werden)
default_ticker_list = """AAPL
MSFT
TSLA
NVDA
GOOG"""

# Session State für persistente Eingabe
if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = default_ticker_list

# Eingabefeld
ticker_input = st.text_area("✍️ Ticker-Liste (eine pro Zeile)", st.session_state.ticker_input, height=200)
ticker_list = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
st.session_state.ticker_input = ticker_input

def calculate_indicators(ticker):
    try:
        data = yf.download(ticker, period="300d", progress=False)

        if data.empty or len(data) < 260:
            return None

        # Absichern der Preis-Spalte
        if 'Adj Close' in data.columns and not data['Adj Close'].isnull().all():
            data['Close'] = data['Adj Close']
        elif 'Close' in data.columns:
            data['Close'] = data['Close']
        else:
            return None

        latest_close = data['Close'].iloc[-1]
        gd130 = data['Close'].rolling(window=130).mean().iloc[-1]
        gd200 = data['Close'].rolling(window=200).mean().iloc[-1]

        dist_gd130 = (latest_close - gd130) / gd130 * 100
        dist_gd200 = (latest_close - gd200) / gd200 * 100
        mom260 = (latest_close / data['Close'].iloc[-260] - 1) * 100

        # MOMJT: 6-Monats-Momentum
        monthly_returns = []
        for i in range(1, 7):
            past_day = -21 * i
            if abs(past_day) >= len(data):
                return None
            ret = (latest_close / data['Close'].iloc[past_day] - 1) * 100
            monthly_returns.append(ret)
        momjt = np.mean(monthly_returns)

        return round(dist_gd200, 2), round(dist_gd130, 2), round(mom260, 2), round(momjt, 2)

    except Exception as e:
        return None

# Button zur Auswertung
if st.button("🔄 Aktualisieren"):
    st.markdown("✅ **Datenüberprüfung der Ticker**")
    result = []
    for ticker in ticker_list:
        indicators = calculate_indicators(ticker)
        if indicators:
            gd200, gd130, mom260, momjt = indicators
            score = gd200 + gd130 + mom260 + momjt
            result.append({
                "Wert": ticker,
                "Ticker": ticker,
                "GD200": gd200,
                "GD130": gd130,
                "MOM260": mom260,
                "MOMJT": momjt,
                "Gesamt": round(score, 2)
            })
        else:
            result.append({
                "Wert": f"{ticker} (Fehler bei Analyse)",
                "Ticker": ticker,
                "GD200": None,
                "GD130": None,
                "MOM260": None,
                "MOMJT": None,
                "Gesamt": None
            })

    df = pd.DataFrame(result)
    df = df.sort_values(by="Gesamt", ascending=False, na_position="last")
    st.markdown("📉 **Trendfolge Ranking**")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
