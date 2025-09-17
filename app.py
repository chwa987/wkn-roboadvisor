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
            data["GD200"] = data["Kurs"].rolling(window=200).mean()
            data["GD130"] = data["Kurs"].rolling(window=130).mean()

            last_close = data["Kurs"].iloc[-1]
            gd200 = data["GD200"].iloc[-1]
            gd130 = data["GD130"].iloc[-1]

            abstand_gd200 = (last_close - gd200) / gd200 * 100 if not np.isnan(gd200) else np.nan
            abstand_gd130 = (last_close - gd130) / gd130 * 100 if not np.isnan(gd130) else np.nan

            if len(data) > 260:
                mom260 = (last_close / data["Kurs"].iloc[-260] - 1) * 100
                ret_12m = (last_close / data["Kurs"].iloc[-260] - 1)
                ret_1m = (last_close / data["Kurs"].iloc[-21] - 1)
                momjt = (ret_12m - ret_1m) * 100
            else:
                mom260 = np.nan
                momjt = np.nan

            results.append([
                name, round(last_close, 2),
                round(abstand_gd200, 2), round(abstand_gd130, 2),
                round(mom260, 2), round(momjt, 2)
            ])

        except Exception as e:
            st.error(f"❌ Fehler bei {name} ({ticker}): {e}")
            results.append([name, None, None, None, None, None])

    # DataFrame bauen
    df = pd.DataFrame(results, columns=[
        "Aktie", "Kurs aktuell", "Abstand GD200 (%)",
        "Abstand GD130 (%)", "MOM260 (%)", "MOMJT (%)"
    ])

    # Ranking (Summe der Ränge)
    rank_df = df.copy()
    for col in ["Abstand GD200 (%)", "Abstand GD130 (%)", "MOM260 (%)", "MOMJT (%)"]:
        if df[col].notna().any():
            rank_df[col + " Rank"] = rank_df[col].rank(ascending=False)
        else:
            rank_df[col + " Rank"] = np.nan

    df["Momentum-Score"] = rank_df[
        [c for c in rank_df.columns if "Rank" in c]
    ].sum(axis=1, skipna=True)

    df = df.sort_values("Momentum-Score").reset_index(drop=True)

    st.dataframe(df)
