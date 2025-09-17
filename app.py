import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="📈 WKN RoboAdvisor", layout="wide")
st.title("📊 WKN RoboAdvisor – Trendfolge Ranking")

st.markdown("""
Gib unten eine Liste von WKNs oder Ticker-Symbolen ein (eine pro Zeile). Die App berechnet automatisch:

- Abstand zur 200-Tage-Linie (GD200)
- Abstand zur 130-Tage-Linie (GD130 / Relative Stärke)
- Momentum über 260 Tage (MOM260)
- Momentum-Indikator nach Jegadeesh/Titman (MOMJT)
""")

# Beispiel-Werte
default_input = """AAPL
BNTX
NVDA
TSLA
MSFT"""

# Session State für Eingabe
if 'wkn_input' not in st.session_state:
    st.session_state.wkn_input = default_input

# Eingabe
user_input = st.text_area("✍️ WKNs oder Ticker (eine pro Zeile)", st.session_state.wkn_input, height=200)
tickers = [line.strip() for line in user_input.splitlines() if line.strip()]
st.session_state.wkn_input = user_input

# Funktion zur Berechnung der Indikatoren
def calculate_indicators(ticker):
    try:
        data = yf.download(ticker, period="300d")
        if len(data) < 260 or data.empty:
            return None

        data['Close'] = data['Adj Close']
        latest_close = data['Close'].iloc[-1]
        gd130 = data['Close'].rolling(window=130).mean().iloc[-1]
        gd200 = data['Close'].rolling(window=200).mean().iloc[-1]

        dist_gd130 = (latest_close - gd130) / gd130 * 100
        dist_gd200 = (latest_close - gd200) / gd200 * 100
        mom260 = (latest_close / data['Close'].iloc[-260] - 1) * 100

        # MOMJT (6 Monats-Renditen durchschnittlich)
        monthly_returns = []
        for i in range(1, 7):
            past_day = -21 * i
            ret = (latest_close / data['Close'].iloc[past_day] - 1) * 100
            monthly_returns.append(ret)
        momjt = np.mean(monthly_returns)

        return round(dist_gd200, 2), round(dist_gd130, 2), round(mom260, 2), round(momjt, 2)

    except Exception as e:
        return None

# Button zur Auswertung
if st.button("🔄 Aktualisieren"):
    st.subheader("✅ Datenüberprüfung der Ticker")
    result_list = []

    for ticker in tickers:
        data = yf.download(ticker, period="300d")
        if data.empty:
            st.warning(f"❌ {ticker}: Keine Daten gefunden")
            result_list.append({
                "Wert": f"{ticker} (Keine Daten)",
                "Ticker": ticker,
                "GD200": None,
                "GD130": None,
                "MOM260": None,
                "MOMJT": None,
                "Gesamt": None
            })
            continue

        stock = yf.Ticker(ticker)
        name = stock.info.get("shortName", ticker)

        indicators = calculate_indicators(ticker)
        if indicators:
            gd200, gd130, mom260, momjt = indicators
            score = gd200 + gd130 + mom260 + momjt
            result_list.append({
                "Wert": name,
                "Ticker": ticker,
                "GD200": gd200,
                "GD130": gd130,
                "MOM260": mom260,
                "MOMJT": momjt,
                "Gesamt": round(score, 2)
            })
        else:
            result_list.append({
                "Wert": f"{ticker} (Fehler bei Analyse)",
                "Ticker": ticker,
                "GD200": None,
                "GD130": None,
                "MOM260": None,
                "MOMJT": None,
                "Gesamt": None
            })

    df = pd.DataFrame(result_list)
    df = df.sort_values(by="Gesamt", ascending=False)
    st.subheader("📈 Trendfolge Ranking")
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
