import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

st.set_page_config(page_title="📈 Momentum-Satelliten", layout="wide")
st.title("📊 Momentum-Analyse – Satellitenwerte")

# ---------- Hilfsfunktionen ----------
def safe_round(x, nd=2):
    """Rundet sicher auf nd Stellen. Gibt None zurück bei NaN/inf/Fehler/Series."""
    try:
        # Falls Series/array: letzten Wert als float versuchen
        if isinstance(x, (pd.Series, np.ndarray)):
            x = x[-1]
        val = float(x)
        if np.isfinite(val):
            return round(val, nd)
    except Exception:
        pass
    return None

def rank_numeric(series, ascending=False):
    """Robustes Ranking: cast zu numeric, NaNs erlaubt."""
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(ascending=ascending, method="min")

# ---------- Eingaben ----------
uploaded_file = st.file_uploader("📂 CSV mit 'Ticker' und optional 'Name' hochladen", type="csv")
tickers_input = st.text_input(
    "Oder Ticker (Yahoo Finance) eingeben, Komma-getrennt:",
    value="APP, LEU, XMTR, RHM.DE"
)

# Ticker-Liste + Namensmapping
if uploaded_file:
    df_tickers = pd.read_csv(uploaded_file)
    if "Ticker" not in df_tickers.columns:
        st.error("❌ CSV braucht mindestens die Spalte 'Ticker'.")
        ticker_list, name_map = [], {}
    else:
        ticker_list = df_tickers["Ticker"].dropna().astype(str).str.strip().tolist()
        if "Name" in df_tickers.columns:
            name_map = dict(zip(df_tickers["Ticker"].astype(str).str.strip(), df_tickers["Name"].astype(str)))
        else:
            name_map = {t: t for t in ticker_list}
else:
    ticker_list = [t.strip() for t in tickers_input.split(",") if t.strip()]
    name_map = {t: t for t in ticker_list}

# ---------- Start ----------
if st.button("🔄 Aktualisieren") and ticker_list:
    end = datetime.today()
    start = end - timedelta(days=400)

    # Referenzindex für Relative Stärke (S&P 500)
    try:
        idx = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True, actions=False, threads=False)
        idx_price = pd.to_numeric(idx.get("Close"), errors="coerce")
    except Exception as e:
        st.error(f"❌ Indexdaten (^GSPC) nicht geladen: {e}")
        idx_price = None

    rows = []

    for ticker in ticker_list:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, actions=False, threads=False, group_by="column")
            if data is None or len(data) == 0:
                st.warning(f"⚠️ Keine Daten für {ticker}.")
                rows.append([None, ticker, name_map.get(ticker, ticker)] + [None]*7)
                continue

            # Preis/Volumen als Series erzwingen
            price = pd.to_numeric(data["Close"], errors="coerce")
            volume = pd.to_numeric(data.get("Volume", pd.Series(index=price.index, dtype=float)), errors="coerce")

            if price.dropna().empty:
                st.warning(f"⚠️ Keine gültigen Schlusskurse für {ticker}.")
                rows.append([None, ticker, name_map.get(ticker, ticker)] + [None]*7)
                continue

            last_close = price.iloc[-1]

            # GDs
            gd200 = price.rolling(200).mean().iloc[-1] if len(price) >= 200 else np.nan
            gd130 = price.rolling(130).mean().iloc[-1] if len(price) >= 130 else np.nan

            abw200 = (last_close - gd200) / gd200 * 100 if np.isfinite(gd200) else np.nan
            abw130 = (last_close - gd130) / gd130 * 100 if np.isfinite(gd130) else np.nan

            # Momentum
            if len(price) > 260 and np.isfinite(price.iloc[-260]):
                mom260 = (last_close / price.iloc[-260] - 1) * 100
                ret_12m = last_close / price.iloc[-260] - 1
            else:
                mom260 = np.nan
                ret_12m = np.nan

            if len(price) > 21 and np.isfinite(price.iloc[-21]) and np.isfinite(ret_12m):
                ret_1m = last_close / price.iloc[-21] - 1
                momjt = (ret_12m - ret_1m) * 100
            else:
                momjt = np.nan

            # Relative Stärke vs. Index (12M)
            if idx_price is not None and len(idx_price) > 260 and np.isfinite(ret_12m):
                idx_ret12m = idx_price.iloc[-1] / idx_price.iloc[-260] - 1
                if np.isfinite(idx_ret12m):
                    rel_str = ((1 + ret_12m) / (1 + idx_ret12m) - 1) * 100
                else:
                    rel_str = np.nan
            else:
                rel_str = np.nan

            # Volumen-Score (aktuelles Volumen / 50-Tage-Schnitt)
            if not volume.dropna().empty and len(volume) > 50:
                vol50 = volume.rolling(50).mean().iloc[-1]
                vol_score = (volume.iloc[-1] / vol50) if (vol50 and np.isfinite(vol50) and vol50 != 0) else np.nan
            else:
                vol_score = np.nan

            rows.append([
                None,
                ticker,
                name_map.get(ticker, ticker),
                safe_round(last_close),
                safe_round(abw200),
                safe_round(abw130),
                safe_round(mom260),
                safe_round(momjt),
                safe_round(rel_str),
                safe_round(vol_score, nd=2)
            ])

        except Exception as e:
            st.error(f"❌ Fehler bei {ticker}: {e}")
            rows.append([None, ticker, name_map.get(ticker, ticker)] + [None]*7)

    # DataFrame
    df = pd.DataFrame(rows, columns=[
        "Signal", "Ticker", "Name", "Kurs aktuell",
        "Abstand GD200 (%)", "Abstand GD130 (%)",
        "MOM260 (%)", "MOMJT (%)",
        "Relative Stärke (%)", "Volumen-Score"
    ])

    # Alles numerisch machen (Ranking robuster)
    numeric_cols = ["Kurs aktuell","Abstand GD200 (%)","Abstand GD130 (%)","MOM260 (%)","MOMJT (%)","Relative Stärke (%)","Volumen-Score"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ranking
    rank_cols = ["Abstand GD200 (%)","Abstand GD130 (%)","MOM260 (%)","MOMJT (%)","Relative Stärke (%)","Volumen-Score"]
    rank_df = pd.DataFrame({c+" Rank": rank_numeric(df[c], ascending=False) for c in rank_cols})
    df["Momentum-Score"] = rank_df.sum(axis=1, skipna=True)

    # Ampel (terciles; Fallback bei wenig Werten)
    valid = df["Momentum-Score"].dropna()
    if len(valid) >= 3:
        q33, q66 = np.nanpercentile(valid, [33, 66])
    elif len(valid) == 2:
        q33 = q66 = np.nanmedian(valid)
    elif len(valid) == 1:
        q33 = q66 = valid.iloc[0]
    else:
        q33 = q66 = np.nan

    for i, s in df["Momentum-Score"].items():
        if pd.isna(s):
            df.at[i, "Signal"] = "⚪ n/a"
        elif not pd.isna(q33) and s <= q33:
            df.at[i, "Signal"] = "🟢 Stark"
        elif not pd.isna(q66) and s <= q66:
            df.at[i, "Signal"] = "🟡 Neutral"
        else:
            df.at[i, "Signal"] = "🔴 Schwach"

    # Sortierung & Anzeige
    df = df.sort_values("Momentum-Score", na_position="last").reset_index(drop=True)
    st.dataframe(df)

    # Export
    st.download_button(
        "💾 CSV exportieren",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="momentum_ergebnisse.csv",
        mime="text/csv"
            )
