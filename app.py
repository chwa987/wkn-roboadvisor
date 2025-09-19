# app.py
# Momentum-Screener mit Handlungsempfehlungen (Kaufen/Halten/Verkaufen)
# Tabs: Analyse | Handlungsempfehlungen

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Momentum-Screener", page_icon="📈", layout="wide")

# ---------------------------- #
#            Utils             #
# ---------------------------- #

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end):
    """
    Holt OHLCV-Daten für mehrere Ticker, robuster Umgang mit 'Adj Close' vs. 'Close'.
    Gibt einen DataFrame (Schlusskurse) und ein Volumen-DF zurück.
    """
    if isinstance(ticker_list, str):
        tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    else:
        tickers = [t.strip() for t in ticker_list if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order

    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    try:
        # yfinance: Einzelabruf ist oft robuster als Multi, aber Multi spart Calls.
        data = yf.download(
            tickers=" ".join(tickers),
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        st.error(f"Fehler beim Download: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Auf 'Adj Close' fallen backen, falls nicht vorhanden
    close_dict, vol_dict = {}, {}
    for t in tickers:
        try:
            df = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        except Exception:
            # Bei Einzel-Ticker fällt das MultiIndex weg
            df = data.copy()

        if df.empty:
            continue

        if "Adj Close" in df.columns:
            closes = df["Adj Close"].rename(t)
        else:
            closes = df.get("Close", pd.Series(dtype=float)).rename(t)

        vols = df.get("Volume", pd.Series(dtype=float)).rename(t)
        close_dict[t] = closes
        vol_dict[t] = vols

    price = pd.concat(close_dict.values(), axis=1) if close_dict else pd.DataFrame()
    volume = pd.concat(vol_dict.values(), axis=1) if vol_dict else pd.DataFrame()

    # Aufräumen
    price = price.sort_index().dropna(how="all")
    volume = volume.reindex_like(price)

    return price, volume


def pct_change_over_window(series: pd.Series, days: int) -> float:
    if len(series.dropna()) < days + 1:
        return np.nan
    start_val = series.dropna().iloc[-(days+1)]
    end_val = series.dropna().iloc[-1]
    if start_val <= 0 or pd.isna(start_val) or pd.isna(end_val):
        return np.nan
    return (end_val / start_val - 1.0) * 100.0


def safe_sma(series: pd.Series, window: int) -> pd.Series:
    if series is None or series.empty:
        return series
    return series.rolling(window=window, min_periods=max(5, window // 5)).mean()


def zscore_last(value: float, mean: float, std: float) -> float:
    if std is None or std == 0 or np.isnan(std):
        return 0.0
    return (value - mean) / std


def volume_score(vol_series: pd.Series, lookback=60):
    """Volume-Multiplikator: aktuelles Vol / SMA(lookback). Caps (0.5 – 2.0)."""
    if vol_series is None or vol_series.dropna().empty:
        return np.nan
    cur = vol_series.dropna().iloc[-1]
    base = vol_series.rolling(lookback, min_periods=max(5, lookback//5)).mean().iloc[-1]
    if base is None or base == 0 or pd.isna(base) or pd.isna(cur):
        return np.nan
    ratio = cur / base
    return float(np.clip(ratio, 0.5, 2.0))


def compute_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame):
    """
    Berechnet alle Kennzahlen je Ticker:
    - MOM260, MOMJT (90 Tage), Relative Stärke (z-score vs. Universe 90T), Volumen-Score
    - Abstände zu GD20/GD50/GD200, Signal (Über/Unter)
    - Momentum-Score (gewichtete Formel)
    """
    results = []

    # Universe-Renditen für Relative Stärke: 90 Tage % Change
    mom90_universe = {}
    for t in price_df.columns:
        mom90_universe[t] = pct_change_over_window(price_df[t], 90)
    mom90_series = pd.Series(mom90_universe).astype(float)
    mu, sigma = mom90_series.mean(), mom90_series.std(ddof=0)

    for t in price_df.columns:
        s = price_df[t].dropna()
        if s.empty or len(s) < 60:
            continue

        last = s.iloc[-1]
        sma20 = safe_sma(s, 20).iloc[-1]
        sma50 = safe_sma(s, 50).iloc[-1]
        sma130 = safe_sma(s, 130).iloc[-1]
        sma200 = safe_sma(s, 200).iloc[-1]

        mom260 = pct_change_over_window(s, 260)
        momJT = pct_change_over_window(s, 90)
        rs_90 = mom90_series.get(t, np.nan)
        rs_z = zscore_last(rs_90, mu, sigma) if not np.isnan(rs_90) else np.nan

        vol_sc = volume_score(volume_df.get(t, pd.Series(dtype=float)), lookback=60)

        def dist(p, m):
            if pd.isna(p) or pd.isna(m) or m == 0:
                return np.nan
            return (p / m - 1.0) * 100.0

        d20 = dist(last, sma20)
        d50 = dist(last, sma50)
        d130 = dist(last, sma130)
        d200 = dist(last, sma200)

        sig50 = "Über GD50" if (not pd.isna(last) and not pd.isna(sma50) and last >= sma50) else "Unter GD50"
        sig200 = "Über GD200" if (not pd.isna(last) and not pd.isna(sma200) and last >= sma200) else "Unter GD200"
        sig20 = "Über GD20" if (not pd.isna(last) and not pd.isna(sma20) and last >= sma20) else "Unter GD20"

        # --- Momentum-Score (gewichtete Formel) ---
        # Skalen: z-Score ist bereits skaliert; MOMs in % -> sanft log-skaliert
        def logp(x):
            if pd.isna(x):
                return np.nan
            # negative oder kleine Werte abflachen
            return np.sign(x) * np.log1p(abs(x))

        mom_part = 0.40 * logp(mom260) + 0.30 * logp(momJT)
        rs_part = 0.20 * (0 if pd.isna(rs_z) else rs_z)
        vol_part = 0.10 * (0 if pd.isna(vol_sc) else (vol_sc - 1.0))  # >1 => +, <1 => -

        score = mom_part + rs_part + vol_part
        # robust: fehlende Teilkomponenten als 0 behandeln
        if pd.isna(score):
            score = 0.0

        results.append({
            "Ticker": t,
            "Kurs aktuell": last,
            "MOM260 (%)": mom260,
            "MOMJT (%)": momJT,
            "Relative Stärke (%)": rs_90,
            "RS z-Score": rs_z,
            "Volumen-Score": vol_sc,
            "Abstand GD20 (%)": d20,
            "Abstand GD50 (%)": d50,
            "Abstand GD130 (%)": d130,
            "Abstand GD200 (%)": d200,
            "GD20-Signal": sig20,
            "GD50-Signal": sig50,
            "GD200-Signal": sig200,
            "Momentum-Score": float(score),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Sortierung & Rank
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df


def parse_ticker_input(s: str):
    if not s:
        return []
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def dot(color: str) -> str:
    return f"<span style='font-size:18px;color:{color}'>●</span>"


def rec_row(row, in_port, top_n=10, reserve=2):
    """Logik Kaufen/Halten/Verkaufen pro Zeile."""
    t = row["Ticker"]
    rank = row["Rank"]
    over50 = row["GD50-Signal"].startswith("Über")
    over200 = row["GD200-Signal"].startswith("Über")

    # Standard-Entscheidungsbaum
    if t in in_port:
        if not over50:
            return "🔴 Verkaufen (unter GD50)"
        if rank <= top_n:
            return "🟡 Halten"
        if rank <= top_n + reserve and over200:
            return "🟡 Halten (Reserve)"
        return "🔴 Verkaufen (nicht mehr Top)"
    else:
        if rank <= top_n and over50 and over200:
            return "🟢 Kaufen"
        if rank <= top_n + reserve and over50 and over200:
            return "🟡 Beobachten (Reserve)"
        return "—"


# ---------------------------- #
#          Sidebar             #
# ---------------------------- #

st.sidebar.header("⚙️ Einstellungen")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", min_value=3, max_value=50, value=10, step=1)
reserve_m = st.sidebar.number_input("Reserven (Nachrücker)", min_value=0, max_value=20, value=2, step=1)
start_date = st.sidebar.date_input("Startdatum (Datenabruf)", value=datetime.today() - timedelta(days=900))
end_date = st.sidebar.date_input("Enddatum", value=datetime.today())

st.title("📊 Momentum-Analyse mit Handlungsempfehlungen")

# Eingabe: CSV (Ticker & optional Name) oder manuelle Ticker
uploaded = st.file_uploader("CSV mit **Ticker** und optional **Name** hochladen", type=["csv"])
tickers_txt = st.text_input("Oder Ticker (Yahoo Finance) eingeben, komma-getrennt:", "APP, LEU, XMTR, RHM.DE")
portfolio_txt = st.text_input("(Optional) Aktuelle Portfolio-Ticker (für Halten/Verkaufen):", "LEU")

# CSV verarbeiten
name_map = {}
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if "Ticker" not in df_in.columns:
            st.error("In der CSV muss mindestens eine Spalte **Ticker** enthalten sein.")
        else:
            if "Name" in df_in.columns:
                name_map = dict(zip(df_in["Ticker"].astype(str), df_in["Name"].astype(str)))
            tickers_txt = ", ".join(df_in["Ticker"].astype(str).tolist())
            st.success(f"{len(df_in)} Ticker aus CSV geladen.")
    except Exception as e:
        st.error(f"CSV konnte nicht gelesen werden: {e}")

tickers = parse_ticker_input(tickers_txt)
in_port = set(parse_ticker_input(portfolio_txt))

if not tickers:
    st.info("Bitte Ticker eingeben oder eine CSV laden.")
    st.stop()

# ---------------------------- #
#           Daten              #
# ---------------------------- #

with st.spinner("Lade Kursdaten …"):
    prices, volumes = fetch_ohlc(tickers, start_date, end_date)

if prices.empty:
    st.warning("Keine Kursdaten geladen.")
    st.stop()

df = compute_indicators(prices, volumes)
if df.empty:
    st.warning("Kennzahlen konnten nicht berechnet werden.")
    st.stop()

# Namen mappen (falls vorhanden)
if name_map:
    df["Name"] = df["Ticker"].map(name_map).fillna(df["Ticker"])
else:
    df["Name"] = df["Ticker"]

# Ampeln für Signale
df["_GD50_dot"] = df["GD50-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))
df["_GD200_dot"] = df["GD200-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))
df["_GD20_dot"] = df["GD20-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))

# ---------------------------- #
#             Tabs             #
# ---------------------------- #

tab1, tab2 = st.tabs(["🔬 Analyse", "🧭 Handlungsempfehlungen"])

with tab1:
    st.subheader("Analyse – alle Kennzahlen")
    show_cols = [
        "Rank", "Ticker", "Name", "Kurs aktuell",
        "MOM260 (%)", "MOMJT (%)", "Relative Stärke (%)", "RS z-Score", "Volumen-Score",
        "Abstand GD20 (%)", "Abstand GD50 (%)", "Abstand GD130 (%)", "Abstand GD200 (%)",
        "GD20-Signal", "GD50-Signal", "GD200-Signal",
        "Momentum-Score"
    ]
    st.dataframe(df[show_cols], use_container_width=True)

with tab2:
    st.subheader("Handlungsempfehlungen – Kaufen / Halten / Verkaufen")

    rec_df = df.copy()
    rec_df["Handlung"] = rec_df.apply(lambda r: rec_row(r, in_port, top_n=top_n, reserve=reserve_m), axis=1)

    # Kompakte, umsetzbare Ansicht
    compact = rec_df[[
        "Rank", "Ticker", "Name", "Momentum-Score",
        "GD50-Signal", "GD200-Signal", "Handlung", "_GD50_dot", "_GD200_dot"
    ]].copy()

    # hübsche Ampeln statt Text
    compact["GD50"] = compact["_GD50_dot"]
    compact["GD200"] = compact["_GD200_dot"]
    compact = compact.drop(columns=["_GD50_dot", "_GD200_dot", "GD50-Signal", "GD200-Signal"])

    # Reihenfolge & Sortierung
    compact = compact.sort_values(["Handlung", "Rank"]).reset_index(drop=True)

    # HTML in Spalten erlauben (für die Dots)
    st.write(
        compact.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    # Klare Textliste unten drunter
    buy_list = compact[compact["Handlung"].str.startswith("🟢")][["Ticker", "Name"]]
    sell_list = compact[compact["Handlung"].str.startswith("🔴")][["Ticker", "Name"]]

    if not buy_list.empty or not sell_list.empty:
        st.markdown("---")
    if not buy_list.empty:
        st.markdown("**🟢 Kaufen:** " + ", ".join(f"{t} ({n})" for t, n in buy_list.to_records(index=False)))
    if not sell_list.empty:
        st.markdown("**🔴 Verkaufen:** " + ", ".join(f"{t} ({n})" for t, n in sell_list.to_records(index=False)))

# Fußnote
st.caption("Hinweis: Kennzahlen dienen ausschließlich zu Informations- und Ausbildungszwecken.")
