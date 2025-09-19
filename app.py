import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="📊 Momentum-Strategie", layout="wide")
st.title("📈 Momentum Screener mit Exit-Logik (Top 20 → Top 10 + Reserve)")

# CSV Upload
uploaded_file = st.file_uploader("CSV mit 'Ticker' und optional 'Name' hochladen", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    tickers = df["Ticker"].dropna().unique().tolist()
    names = dict(zip(df["Ticker"], df["Name"])) if "Name" in df.columns else {t: t for t in tickers}
else:
    st.stop()

# Kursdaten laden
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        raise KeyError("Weder Adj Close noch Close in Yahoo Finance Daten!")
    return prices.dropna(how="all")

start_date = "2018-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
prices = load_data(tickers, start_date, end_date)

# GD-Berechnungen
gd50 = prices.rolling(50).mean()
gd130 = prices.rolling(130).mean()
gd200 = prices.rolling(200).mean()

# Momentum-Indikatoren
mom260 = prices.pct_change(260) * 100
momjt = (prices / prices.shift(130) - 1) * 100

# Score-Berechnung
latest = prices.index[-1]
scores = []
for t in tickers:
    try:
        price = prices.loc[latest, t]
        val = {
            "Ticker": t,
            "Name": names.get(t, t),
            "Kurs": round(price, 2),
            "Abstand GD200 (%)": round((price / gd200.loc[latest, t] - 1) * 100, 2),
            "Abstand GD130 (%)": round((price / gd130.loc[latest, t] - 1) * 100, 2),
            "MOM260 (%)": round(mom260.loc[latest, t], 2),
            "MOMJT (%)": round(momjt.loc[latest, t], 2),
            "GD50": price > gd50.loc[latest, t] if not np.isnan(gd50.loc[latest, t]) else False
        }
        # Momentum-Score: gewichtete Summe
        val["Momentum-Score"] = (
            0.3 * val["Abstand GD200 (%)"] +
            0.3 * val["Abstand GD130 (%)"] +
            0.2 * val["MOM260 (%)"] +
            0.2 * val["MOMJT (%)"]
        )
        scores.append(val)
    except Exception as e:
        st.warning(f"Fehler bei {t}: {e}")

df_scores = pd.DataFrame(scores).sort_values("Momentum-Score", ascending=False).reset_index(drop=True)

# Exit-Logik: Top 20 → Top 10 aktiv + Reserve
top20 = df_scores.head(20).copy()
active, reserve = [], []
for _, row in top20.iterrows():
    if len(active) < 10 and row["GD50"]:
        active.append(row["Ticker"])
    elif len(reserve) < 3 and row["GD50"]:
        reserve.append(row["Ticker"])

# Spalte "Kaufkandidat?"
def status(row):
    if row["Ticker"] in active:
        return "✅ Aktiv"
    elif row["Ticker"] in reserve:
        return "🟨 Reserve"
    else:
        return "❌ Nein"

top20["Kaufkandidat?"] = top20.apply(status, axis=1)

# Darstellung
st.subheader("📊 Top 20 Ranking mit Exit-Logik")
st.dataframe(top20[[
    "Ticker", "Name", "Momentum-Score", "Abstand GD200 (%)", 
    "Abstand GD130 (%)", "MOM260 (%)", "MOMJT (%)", "Kaufkandidat?"
]])
