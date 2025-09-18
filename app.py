import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="📊 Momentum Aktien App", layout="wide")
st.title("📈 Momentum-Analyse mit gewichteter, normalisierter Formel")

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def get_series(data, column="Close"):
    """Gibt immer eine 1D-Series zurück (robust gegen Multi-Columns)."""
    try:
        if column not in data:
            return pd.Series(dtype=float)
        s = data[column]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)

def compute_indicators(price, idx_price, volume):
    """Berechnet die 6 Kriterien. Höher = besser für alle."""
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

    # Relative Stärke vs. S&P 500 (12M)
    if not idx_price.dropna().empty and len(idx_price) > 260 and np.isfinite(ret_12m):
        idx_ret12m = idx_price.iloc[-1] / idx_price.iloc[-260] - 1
        rel_str = ((1 + ret_12m) / (1 + idx_ret12m) - 1) * 100 if np.isfinite(idx_ret12m) else np.nan
    else:
        rel_str = np.nan

    # Volumen-Score (heute / 50d-Durchschnitt)
    if not volume.dropna().empty and len(volume) > 50:
        vol50 = volume.rolling(50).mean().iloc[-1]
        vol_score = (volume.iloc[-1] / vol50) if np.isfinite(vol50) and vol50 != 0 else np.nan
    else:
        vol_score = np.nan

    return [abw200, abw130, mom260, momjt, rel_str, vol_score]

# Gewichte der Formel (Mentor-Empfehlung)
WEIGHTS = {
    "Abstand GD200 (%)": 0.20,
    "Abstand GD130 (%)": 0.15,
    "MOM260 (%)":        0.25,
    "MOMJT (%)":         0.15,
    "Relative Stärke (%)": 0.15,
    "Volumen-Score":     0.10
}
IND_COLS = list(WEIGHTS.keys())

# ----------------------------
# Inputs
# ----------------------------
uploaded_file = st.file_uploader("📂 CSV mit 'Ticker' und optional 'Name' hochladen", type=["csv"])
tickers_input = st.text_input("Oder Ticker eingeben (kommasepariert):", "APP, LEU, XMTR, RHM.DE")

if uploaded_file:
    df_tickers = pd.read_csv(uploaded_file)
    if "Ticker" in df_tickers.columns:
        ticker_list = df_tickers["Ticker"].dropna().astype(str).str.strip().tolist()
        name_map = dict(zip(df_tickers["Ticker"].astype(str).str.strip(), 
                            df_tickers.get("Name", df_tickers["Ticker"]).astype(str)))
    else:
        st.error("❌ CSV muss eine Spalte 'Ticker' enthalten.")
        ticker_list, name_map = [], {}
else:
    ticker_list = [t.strip() for t in tickers_input.split(",") if t.strip()]
    name_map = {t: t for t in ticker_list}

# ----------------------------
# Analyse
# ----------------------------
if st.button("🔄 Analyse starten") and ticker_list:
    end = datetime.today()
    start = end - timedelta(days=400)

    # Benchmark: S&P 500 (für Relative Stärke)
    idx = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    idx_price = get_series(idx, "Close")

    rows = []
    for ticker in ticker_list:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            price = get_series(data, "Close")
            volume = get_series(data, "Volume")
            if price.dropna().empty:
                continue

            indicators = compute_indicators(price, idx_price, volume)  # 6 Werte
            rows.append({
                "Signal": None,
                "Ticker": ticker,
                "Name": name_map.get(ticker, ticker),
                "Kurs aktuell": round(price.iloc[-1], 2),
                "Abstand GD200 (%)": indicators[0],
                "Abstand GD130 (%)": indicators[1],
                "MOM260 (%)": indicators[2],
                "MOMJT (%)": indicators[3],
                "Relative Stärke (%)": indicators[4],
                "Volumen-Score": indicators[5]
            })
        except Exception as e:
            st.warning(f"⚠️ Fehler bei {ticker}: {e}")

    if not rows:
        st.error("Keine verwertbaren Daten gefunden.")
        st.stop()

    df = pd.DataFrame(rows)

    # --- Z-Standardisierung über das gesamte Universum ---
    z_df = pd.DataFrame(index=df.index)
    for col in IND_COLS:
        col_series = pd.to_numeric(df[col], errors="coerce")
        mu = col_series.mean(skipna=True)
        sigma = col_series.std(ddof=0, skipna=True)
        if pd.isna(sigma) or sigma == 0:
            # keine Streuung -> neutraler Beitrag
            z_df[col] = 0.0
        else:
            z_df[col] = (col_series - mu) / sigma

    # --- Gewichteter Score ---
    weight_vec = pd.Series(WEIGHTS)
    df["Momentum-Score"] = (z_df[IND_COLS] * weight_vec).sum(axis=1, skipna=True)

    # --- Ampel (Quantile) ---
    if df["Momentum-Score"].notna().sum() >= 3:
        q33, q66 = df["Momentum-Score"].quantile([0.33, 0.66])
    else:
        # Fallback: Median-Schwelle
        med = df["Momentum-Score"].median()
        q33 = q66 = med

    for i, s in df["Momentum-Score"].items():
        if pd.isna(s):
            df.at[i, "Signal"] = "⚪ n/a"
        elif s >= q66:
            df.at[i, "Signal"] = "🟢 Stark"
        elif s >= q33:
            df.at[i, "Signal"] = "🟡 Neutral"
        else:
            df.at[i, "Signal"] = "🔴 Schwach"

    # --- Sortierung: höherer Score = besser ---
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)

    # Ampel vorne
    cols = ["Signal", "Ticker", "Name", "Kurs aktuell"] + IND_COLS + ["Momentum-Score"]
    df = df[cols]

    # Runden für Anzeige
    df_display = df.copy()
    for c in ["Kurs aktuell", *IND_COLS, "Momentum-Score"]:
        df_display[c] = pd.to_numeric(df_display[c], errors="coerce").round(2)

    st.dataframe(df_display, use_container_width=True)

    # Export
    st.download_button(
        "📥 Ergebnisse als CSV speichern",
        data=df_display.to_csv(index=False).encode("utf-8"),
        file_name="momentum_scores_weighted.csv",
        mime="text/csv"
    )
