import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="📊 Momentum Aktien App", layout="wide")
st.title("📈 Momentum-Analyse mit gewichteter Formel")

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def get_series(data, column="Close"):
    """Sichert, dass immer eine 1D-Serie zurückkommt."""
    try:
        if column not in data:
            return pd.Series(dtype=float)
        s = data[column]
        if isinstance(s, pd.DataFrame):  # mehrere Spalten -> nimm die erste
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)

def compute_indicators(price, idx_price, volume):
    """Berechne die 6 Kennzahlen für eine Aktie."""
    if price.dropna().empty:
        return [np.nan]*6

    last_close = price.iloc[-1]
    gd200 = price.rolling(200).mean().iloc[-1] if len(price) >= 200 else np.nan
    gd130 = price.rolling(130).mean().iloc[-1] if len(price) >= 130 else np.nan

    abw200 = (last_close - gd200) / gd200 * 100 if np.isfinite(gd200) else np.nan
    abw130 = (last_close - gd130) / gd130 * 100 if np.isfinite(gd130) else np.nan

    if len(price) > 260 and np.isfinite(price.iloc[-260]):
        mom260 = (last_close / price.iloc[-260] - 1) * 100
        ret_12m = last_close / price.iloc[-260] - 1
    else:
        mom260, ret_12m = np.nan, np.nan

    if len(price) > 21 and np.isfinite(price.iloc[-21]) and np.isfinite(ret_12m):
        ret_1m = last_close / price.iloc[-21] - 1
        momjt = (ret_12m - ret_1m) * 100
    else:
        momjt = np.nan

    if not idx_price.dropna().empty and len(idx_price) > 260 and np.isfinite(ret_12m):
        idx_ret12m = idx_price.iloc[-1] / idx_price.iloc[-260] - 1
        rel_str = ((1 + ret_12m) / (1 + idx_ret12m) - 1) * 100 if np.isfinite(idx_ret12m) else np.nan
    else:
        rel_str = np.nan

    if not volume.dropna().empty and len(volume) > 50:
        vol50 = volume.rolling(50).mean().iloc[-1]
        vol_score = (volume.iloc[-1] / vol50) if np.isfinite(vol50) and vol50 != 0 else np.nan
    else:
        vol_score = np.nan

    return [abw200, abw130, mom260, momjt, rel_str, vol_score]

# ----------------------------
# Streamlit UI
# ----------------------------
uploaded_file = st.file_uploader("📂 Lade eine CSV mit Tickers + optional Name", type="csv")
tickers_input = st.text_input("Oder gib Ticker ein (kommasepariert):", "APP, LEU, XMTR, RHM.DE")

if uploaded_file:
    df_tickers = pd.read_csv(uploaded_file)
    if "Ticker" in df_tickers.columns:
        ticker_list = df_tickers["Ticker"].dropna().tolist()
        if "Name" in df_tickers.columns:
            name_map = dict(zip(df_tickers["Ticker"], df_tickers["Name"]))
        else:
            name_map = {t: t for t in ticker_list}
    else:
        st.error("❌ CSV muss eine Spalte 'Ticker' enthalten.")
        ticker_list, name_map = [], {}
else:
    ticker_list = [t.strip() for t in tickers_input.split(",") if t.strip()]
    name_map = {t: t for t in ticker_list}

if st.button("🔄 Analyse starten") and ticker_list:
    end = datetime.today()
    start = end - timedelta(days=400)

    # Benchmark: S&P 500
    idx = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    idx_price = get_series(idx, "Close")

    results = []

    for ticker in ticker_list:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            price = get_series(data, "Close")
            volume = get_series(data, "Volume")

            if price.dropna().empty:
                continue

            indicators = compute_indicators(price, idx_price, volume)
            labels = ["GD200", "GD130", "MOM260", "MOMJT", "RelStr", "VolScore"]

            ind_df = pd.DataFrame([indicators], columns=labels)

            # z-Standardisierung
            z_scores = (ind_df - ind_df.mean()) / ind_df.std(ddof=0)

            # Gewichte
            weights = {
                "GD200": 0.20,
                "GD130": 0.15,
                "MOM260": 0.25,
                "MOMJT": 0.15,
                "RelStr": 0.15,
                "VolScore": 0.10
            }

            # gewichteter Score
            score = 0
            for col in labels:
                if pd.notna(z_scores.at[0, col]):
                    score += z_scores.at[0, col] * weights[col]

            results.append([
                None,  # Ampel-Signal
                ticker,
                name_map.get(ticker, ticker),
                round(price.iloc[-1], 2),
                *[round(x, 2) if pd.notna(x) else None for x in indicators],
                round(score, 2)
            ])
        except Exception as e:
            st.warning(f"⚠️ Fehler bei {ticker}: {e}")

    # DataFrame bauen
    df = pd.DataFrame(results, columns=[
        "Signal", "Ticker", "Name", "Kurs aktuell",
        "Abstand GD200 (%)", "Abstand GD130 (%)",
        "MOM260 (%)", "MOMJT (%)", "Relative Stärke (%)", "Volumen-Score",
        "Momentum-Score"
    ])

    # Ampel basierend auf Score
    if not df["Momentum-Score"].isna().all():
        quantiles = df["Momentum-Score"].quantile([0.33, 0.66]).to_dict()
        for i, row in df.iterrows():
            if row["Momentum-Score"] <= quantiles[0.33]:
                df.at[i, "Signal"] = "🟢 Stark"
            elif row["Momentum-Score"] <= quantiles[0.66]:
                df.at[i, "Signal"] = "🟡 Neutral"
            else:
                df.at[i, "Signal"] = "🔴 Schwach"

    df = df.sort_values("Momentum-Score", ascending=True).reset_index(drop=True)

    st.dataframe(df)

    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Ergebnisse als CSV speichern", data=csv_export, file_name="momentum_scores.csv", mime="text/csv")
