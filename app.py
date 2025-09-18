import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="📊 Momentum Aktien App", layout="wide")
st.title("📈 Momentum-Analyse (gewichtete Formel) + GD20-Signal")

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
    """
    Berechnet die 6 Score-Kriterien + GD20 Infos.
    Rückgabe:
      [abw200, abw130, mom260, momjt, rel_str, vol_score, abw20, gd20_flag]
    """
    if price.dropna().empty:
        return [np.nan]*8

    last_close = price.iloc[-1]

    # Gleitende Durchschnitte
    gd200 = price.rolling(200).mean().iloc[-1] if len(price) >= 200 else np.nan
    gd130 = price.rolling(130).mean().iloc[-1] if len(price) >= 130 else np.nan
    gd20  = price.rolling(20 ).mean().iloc[-1] if len(price) >= 20  else np.nan

    # Abstände
    abw200 = (last_close - gd200) / gd200 * 100 if np.isfinite(gd200) else np.nan
    abw130 = (last_close - gd130) / gd130 * 100 if np.isfinite(gd130) else np.nan
    abw20  = (last_close - gd20 ) / gd20  * 100 if np.isfinite(gd20 ) else np.nan

    # Momentum 12M & Jitney (12M - 1M)
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
    if volume.dropna().empty or len(volume) <= 50:
        vol_score = np.nan
    else:
        vol50 = volume.rolling(50).mean().iloc[-1]
        vol_score = (volume.iloc[-1] / vol50) if (np.isfinite(vol50) and vol50 != 0) else np.nan

    # GD20 Flag
    gd20_flag = None
    if np.isfinite(gd20):
        gd20_flag = "🟢 über GD20" if last_close >= gd20 else "🔴 unter GD20"

    return [abw200, abw130, mom260, momjt, rel_str, vol_score, abw20, gd20_flag]

# Gewichte (Mentor-Empfehlung) – nur für die 6 Score-Kriterien
WEIGHTS = {
    "Abstand GD200 (%)":     0.20,
    "Abstand GD130 (%)":     0.15,
    "MOM260 (%)":            0.25,
    "MOMJT (%)":             0.15,
    "Relative Stärke (%)":   0.15,
    "Volumen-Score":         0.10
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
        name_map = dict(zip(
            df_tickers["Ticker"].astype(str).str.strip(),
            df_tickers.get("Name", df_tickers["Ticker"]).astype(str)
        ))
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
                st.warning(f"⚠️ Keine gültigen Schlusskurse: {ticker}")
                continue

            abw200, abw130, mom260, momjt, rel_str, vol_score, abw20, gd20_flag = compute_indicators(price, idx_price, volume)

            rows.append({
                "Signal": None,
                "Ticker": ticker,
                "Name": name_map.get(ticker, ticker),
                "Kurs aktuell": round(price.iloc[-1], 2),
                "Abstand GD200 (%)": abw200,
                "Abstand GD130 (%)": abw130,
                "MOM260 (%)": mom260,
                "MOMJT (%)": momjt,
                "Relative Stärke (%)": rel_str,
                "Volumen-Score": vol_score,
                "Abstand GD20 (%)": abw20,
                "GD20-Signal": gd20_flag
            })
        except Exception as e:
            st.warning(f"⚠️ Fehler bei {ticker}: {e}")

    if not rows:
        st.error("Keine verwertbaren Daten gefunden.")
        st.stop()

    df = pd.DataFrame(rows)

    # --- Z-Standardisierung nur über die 6 Score-Kriterien ---
    z_df = pd.DataFrame(index=df.index)
    for col in IND_COLS:
        s = pd.to_numeric(df[col], errors="coerce")
        mu = s.mean(skipna=True)
        sigma = s.std(ddof=0, skipna=True)
        z_df[col] = 0.0 if (pd.isna(sigma) or sigma == 0) else (s - mu) / sigma

    # --- Gewichteter Score ---
    weight_vec = pd.Series(WEIGHTS)
    df["Momentum-Score"] = (z_df[IND_COLS] * weight_vec).sum(axis=1, skipna=True)

    # --- Ampel (Quantile) ---
    if df["Momentum-Score"].notna().sum() >= 3:
        q33, q66 = df["Momentum-Score"].quantile([0.33, 0.66])
    else:
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

    # Ampel vorne, GD20-Spalten sichtbar
    display_cols = ["Signal", "Ticker", "Name", "Kurs aktuell"] + IND_COLS + ["Abstand GD20 (%)", "GD20-Signal", "Momentum-Score"]
    df_display = df[display_cols].copy()

    # Runden für Anzeige
    for c in ["Kurs aktuell", *IND_COLS, "Abstand GD20 (%)", "Momentum-Score"]:
        df_display[c] = pd.to_numeric(df_display[c], errors="coerce").round(2)

    st.dataframe(df_display, use_container_width=True)

    # Export
    st.download_button(
        "📥 Ergebnisse als CSV speichern",
        data=df_display.to_csv(index=False).encode("utf-8"),
        file_name="momentum_scores_weighted_gd20.csv",
        mime="text/csv"
    )
