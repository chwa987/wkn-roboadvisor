import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Momentum Aktien App", layout="wide")
st.title("📈 Momentum-Analyse (gewichtete Formel) + Marktmonitor")

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

    # GDs
    gd200 = price.rolling(200).mean().iloc[-1] if len(price) >= 200 else np.nan
    gd130 = price.rolling(130).mean().iloc[-1] if len(price) >= 130 else np.nan
    gd20  = price.rolling(20 ).mean().iloc[-1] if len(price) >= 20  else np.nan

    # Abstände
    abw200 = (last_close - gd200) / gd200 * 100 if np.isfinite(gd200) else np.nan
    abw130 = (last_close - gd130) / gd130 * 100 if np.isfinite(gd130) else np.nan
    abw20  = (last_close - gd20 ) / gd20  * 100 if np.isfinite(gd20 ) else np.nan

    # Momentum (12M) & Jitney (12M - 1M)
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

# Gewichte – NUR für die 6 Score-Kriterien
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
# Marktmonitor-Daten (einmal laden & in beiden Tabs nutzen)
# ----------------------------
def load_market_monitor():
    """Lädt S&P500 (2Y) + berechnet GD50/GD200 + VIX (6M)."""
    sp500 = yf.download("^GSPC", period="2y", interval="1d", auto_adjust=True, progress=False)
    sp500["GD50"]  = sp500["Close"].rolling(50).mean()
    sp500["GD200"] = sp500["Close"].rolling(200).mean()

    aktueller_kurs = float(sp500["Close"].iloc[-1])
    gd50  = float(sp500["GD50"].iloc[-1])  if not np.isnan(sp500["GD50"].iloc[-1])  else np.nan
    gd200 = float(sp500["GD200"].iloc[-1]) if not np.isnan(sp500["GD200"].iloc[-1]) else np.nan

    abw50  = (aktueller_kurs - gd50)  / gd50  * 100 if np.isfinite(gd50)  else np.nan
    abw200 = (aktueller_kurs - gd200) / gd200 * 100 if np.isfinite(gd200) else np.nan

    # Marktphase
    market_phase = "🟢 Über GD200 (Aufwärtstrend)" if (np.isfinite(gd200) and aktueller_kurs >= gd200) else "🔴 Unter GD200 (Risiko-Phase)"

    # VIX (Proxy Angst/Gier)
    try:
        vix = yf.download("^VIX", period="6mo", interval="1d", auto_adjust=True, progress=False)["Close"]
        vix_now = float(vix.iloc[-1]) if not vix.dropna().empty else np.nan
    except Exception:
        vix_now = np.nan

    return sp500, aktueller_kurs, gd50, gd200, abw50, abw200, market_phase, vix_now

sp500, spx, gd50, gd200, abw50, abw200, market_phase, vix_now = load_market_monitor()

# ----------------------------
# Tabs: Marktmonitor | Screener
# ----------------------------
tab_market, tab_screener = st.tabs(["🧭 Marktmonitor (S&P 500)", "📊 Momentum-Screener"])

with tab_market:
    st.subheader("S&P 500 – GD50 / GD200 Übersicht")

    try:
        col1, col2, col3 = st.columns(3)
        col1.metric("S&P 500 (Close)", f"{spx:,.0f}")
        col2.metric("Abstand GD50",  f"{abw50:,.2f} %" if np.isfinite(abw50) else "n/a")
        col3.metric("Abstand GD200", f"{abw200:,.2f} %" if np.isfinite(abw200) else "n/a")

        st.caption(f"Marktphase: **{market_phase}**")
        if np.isfinite(vix_now):
            st.info(f"🧪 VIX (Volatilitätsindex – Angst/Gier-Proxy): **{vix_now:.1f}**")

        # Chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sp500.index, sp500["Close"], label="S&P 500", linewidth=1.4, color="black")
        ax.plot(sp500.index, sp500["GD50"],  label="GD50", linewidth=1.0, color="tab:blue")
        ax.plot(sp500.index, sp500["GD200"], label="GD200", linewidth=1.0, color="tab:red")
        ax.legend(loc="upper left")
        ax.set_ylabel("Indexstand")
        ax.grid(alpha=0.2)
        st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.error(f"❌ Marktmonitor konnte nicht geladen werden: {e}")

with tab_screener:
    st.subheader("Momentum-Screener (gewichtete Formel)")

    # Markt-Badge oben im Screener
    badge = "🟢 Markt: über GD200 – Normalgewicht fahren" if "🟢" in market_phase else "🔴 Markt: unter GD200 – Cash-Quote/Reduktion erwägen"
    st.markdown(f"**{badge}**")

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

        # Z-Standardisierung über das Universum (nur Score-Kriterien)
        z_df = pd.DataFrame(index=df.index)
        for col in IND_COLS:
            s = pd.to_numeric(df[col], errors="coerce")
            mu = s.mean(skipna=True)
            sigma = s.std(ddof=0, skipna=True)
            z_df[col] = 0.0 if (pd.isna(sigma) or sigma == 0) else (s - mu) / sigma

        # Gewichteter Score
        weight_vec = pd.Series(WEIGHTS)
        df["Momentum-Score"] = (z_df[IND_COLS] * weight_vec).sum(axis=1, skipna=True)

        # Ampel (Quantile)
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

        # Sortierung & Anzeige
        df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
        display_cols = ["Signal", "Ticker", "Name", "Kurs aktuell"] + IND_COLS + ["Abstand GD20 (%)", "GD20-Signal", "Momentum-Score"]
        df_display = df[display_cols].copy()
        for c in ["Kurs aktuell", *IND_COLS, "Abstand GD20 (%)", "Momentum-Score"]:
            df_display[c] = pd.to_numeric(df_display[c], errors="coerce").round(2)

        st.dataframe(df_display, use_container_width=True)

        st.download_button(
            "📥 Ergebnisse als CSV speichern",
            data=df_display.to_csv(index=False).encode("utf-8"),
            file_name="momentum_scores_weighted_gd20.csv",
            mime="text/csv"
        )
