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
                ticker, round(last_close, 2),
                round(abstand_gd200, 2), round(abstand_gd130, 2),
                round(mom260, 2), round(momjt, 2), None
            ])

        except Exception as e:
            st.error(f"❌ Fehler bei {ticker}: {e}")
            results.append([ticker, None, None, None, None, None, None])

    # --- DataFrame bauen ---
    df = pd.DataFrame(results, columns=[
        "Ticker", "Kurs aktuell", "Abstand GD200 (%)",
        "Abstand GD130 (%)", "MOM260 (%)", "MOMJT (%)", "Signal"
    ])

    # --- Ranking ---
    rank_df = df.copy()
    for col in ["Abstand GD200 (%)", "Abstand GD130 (%)", "MOM260 (%)", "MOMJT (%)"]:
        if df[col].notna().any():
            rank_df[col + " Rank"] = rank_df[col].rank(ascending=False)
        else:
            rank_df[col + " Rank"] = np.nan

    df["Momentum-Score"] = rank_df[
        [c for c in rank_df.columns if "Rank" in c]
    ].sum(axis=1, skipna=True)

    # --- Signal (Ampel) ---
    if not df["Momentum-Score"].isna().all():
        quantiles = df["Momentum-Score"].quantile([0.33, 0.66]).to_dict()
        for i, row in df.iterrows():
            if row["Momentum-Score"] <= quantiles[0.33]:
                df.at[i, "Signal"] = "🟢 Stark"
            elif row["Momentum-Score"] <= quantiles[0.66]:
                df.at[i, "Signal"] = "🟡 Neutral"
            else:
                df.at[i, "Signal"] = "🔴 Schwach"

    # --- Sortieren nach Momentum ---
    df = df.sort_values("Momentum-Score").reset_index(drop=True)

    # --- Ergebnis anzeigen ---
    st.dataframe(df)

    # --- Export-Button ---
    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Ergebnisse als CSV speichern",
        data=csv_export,
        file_name="momentum_ergebnisse.csv",
        mime="text/csv"
                )
