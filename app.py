import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Momentum-Satelliten", layout="wide")
st.title("📊 Momentum-Analyse – Satellitenwerte")

# Feste Ticker-Liste (kannst du anpassen/erweitern)
ticker_list = {
    "AppLovin": "APP",
    "Centrus Energy": "LEU",
    "Xometry": "XMTR",
    "Rheinmetall": "RHM.DE"
}

# Aktualisieren-Button
if st.button("🔄 Aktualisieren"):
    end = datetime.today()
    start = end - timedelta(days=400)

    results = []

    for name, ticker in ticker_list.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            data["Close"] = data["Adj Close"]

            # Gleitende Durchschnitte
            data["GD200"] = data["Close"].rolling(window=200).mean()
            data["GD130"] = data["Close"].rolling(window=130).mean()

            last_close = data["Close"].iloc[-1]
            gd200 = data["GD200"].iloc[-1]
            gd130 = data["GD130"].iloc[-1]

            # Abstände in %
            abstand_gd200 = (last_close - gd200) / gd200 * 100 if not np.isnan(gd200) else np.nan
            abstand_gd130 = (last_close - gd130) / gd130 * 100 if not np.isnan(gd130) else np.nan

            # MOM260
            if len(data) > 260:
                mom260 = (last_close / data["Close"].iloc[-260] - 1) * 100
            else:
                mom260 = np.nan

            # MOMJT (12M minus 1M)
            if len(data) > 260:
                ret_12m = (last_close / data["Close"].iloc[-260] - 1)
                ret_1m = (last_close / data["Close"].iloc[-21] - 1)
                momjt = (ret_12m - ret_1m) * 100
            else:
                momjt = np.nan

            results.append([
                name, round(last_close, 2), 
                round(abstand_gd200, 2), 
                round(abstand_gd130, 2), 
                round(mom260, 2), 
                round(momjt, 2)
            ])

        except Exception as e:
            results.append([name, None, None, None, None, None])

    # DataFrame bauen
    df = pd.DataFrame(results, columns=[
        "Aktie", "Kurs aktuell", "Abstand GD200 (%)", 
        "Abstand GD130 (%)", "MOM260 (%)", "MOMJT (%)"
    ])

    # Ranking (Summe der Ränge)
    rank_df = df.copy()
    for col in ["Abstand GD200 (%)", "Abstand GD130 (%)", "MOM260 (%)", "MOMJT (%)"]:
        rank_df[col + " Rank"] = rank_df[col].rank(ascending=False)

    df["Momentum-Score"] = rank_df[
        ["Abstand GD200 (%) Rank", "Abstand GD130 (%) Rank", "MOM260 (%) Rank", "MOMJT (%) Rank"]
    ].sum(axis=1)

    df = df.sort_values("Momentum-Score").reset_index(drop=True)

    st.dataframe(df)
