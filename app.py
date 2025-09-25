# app.py
# Momentum-Screener mit Handlungsempfehlungen, Backtest & Exposure-Logik
# Universum: CSV-Upload oder manuelle Eingabe
# GD50/GD200-Filter & Momentum-Score
# Exposure passt sich in 10%-Schritten an die Marktbreite an

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Momentum-RoboAdvisor", page_icon="📈", layout="wide")

# ---------------------------- #
#            Utils             #
# ---------------------------- #

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end, batch_size=150):
    """Holt OHLCV-Daten in Batches, robust gegen viele Ticker."""
    if isinstance(ticker_list, str):
        tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    else:
        tickers = [t.strip() for t in ticker_list if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    close_dict, vol_dict = {}, {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                tickers=" ".join(batch),
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception:
            continue

        for t in batch:
            try:
                df = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
            except Exception:
                df = data.copy()
            if df.empty:
                continue
            closes = (df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")).rename(t)
            vols = df.get("Volume", pd.Series(dtype=float)).rename(t)
            close_dict[t] = closes
            vol_dict[t] = vols

    price = pd.concat(close_dict.values(), axis=1) if close_dict else pd.DataFrame()
    volume = pd.concat(vol_dict.values(), axis=1) if vol_dict else pd.DataFrame()
    price = price.sort_index().dropna(how="all")
    volume = volume.reindex_like(price)
    return price, volume


def pct_change_over_window(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return np.nan
    start_val = s.iloc[-(days+1)]
    end_val = s.iloc[-1]
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
    if vol_series is None or vol_series.dropna().empty:
        return np.nan
    cur = vol_series.dropna().iloc[-1]
    base = vol_series.rolling(lookback, min_periods=max(5, lookback//5)).mean().iloc[-1]
    if base is None or base == 0 or pd.isna(base) or pd.isna(cur):
        return np.nan
    ratio = cur / base
    return float(np.clip(ratio, 0.5, 2.0))


def compute_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame):
    results = []
    # Universe-Renditen (130T) für RS
    mom130_universe = {t: pct_change_over_window(price_df[t], 130) for t in price_df.columns}
    mom130_series = pd.Series(mom130_universe).astype(float)
    mu, sigma = mom130_series.mean(), mom130_series.std(ddof=0)

    for t in price_df.columns:
        s = price_df[t].dropna()
        if s.empty or len(s) < 200:
            continue

        last = s.iloc[-1]
        sma50 = safe_sma(s, 50).iloc[-1]
        sma200 = safe_sma(s, 200).iloc[-1]

        mom260 = pct_change_over_window(s, 260)
        mom130 = pct_change_over_window(s, 130)

        rs_130 = mom130_series.get(t, np.nan)
        rs_z = zscore_last(rs_130, mu, sigma) if not np.isnan(rs_130) else np.nan

        vol_sc = volume_score(volume_df.get(t, pd.Series(dtype=float)), lookback=60)

        d50 = (last / sma50 - 1.0) * 100.0 if last and sma50 else np.nan
        d200 = (last / sma200 - 1.0) * 100.0 if last and sma200 else np.nan

        sig50 = "Über GD50" if last >= sma50 else "Unter GD50"
        sig200 = "Über GD200" if last >= sma200 else "Unter GD200"

        score = (
            0.40 * np.log1p(abs(mom260)) * np.sign(mom260)
            + 0.30 * np.log1p(abs(mom130)) * np.sign(mom130)
            + 0.20 * rs_z
            + 0.10 * (vol_sc - 1.0 if not pd.isna(vol_sc) else 0)
        )

        results.append({
            "Ticker": t,
            "Kurs aktuell": last,
            "MOM260 (%)": mom260,
            "MOM130 (%)": mom130,
            "Relative Stärke (130T) (%)": rs_130,
            "RS z-Score": rs_z,
            "Volumen-Score": vol_sc,
            "Abstand GD50 (%)": d50,
            "Abstand GD200 (%)": d200,
            "GD50-Signal": sig50,
            "GD200-Signal": sig200,
            "Momentum-Score": float(score),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df


def parse_ticker_input(s: str):
    return [t.strip().upper() for t in s.split(",") if t.strip()] if s else []


def rec_row(row, in_port, top_n=10, reserve=2):
    t = row["Ticker"]
    rank = row["Rank"]
    over50 = row["GD50-Signal"].startswith("Über")
    over200 = row["GD200-Signal"].startswith("Über")

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


def dot(color: str) -> str:
    return f"<span style='font-size:18px;color:{color}'>●</span>"


def breadth_to_exposure_tens(breadth: float):
    """Breadth (0..1) in diskrete Exposure (0..1) umwandeln in 10%-Schritten."""
    if pd.isna(breadth):
        return 0.0
    v = float(breadth)
    if v >= 0.95:  # ab 95% als 100%
        return 1.0
    step = int(round(v * 10))  # 0..10
    percent = max(0, min(10, step)) * 10
    return percent / 100.0

# ---------------------------- #
#          Sidebar             #
# ---------------------------- #

st.sidebar.header("⚙️ Einstellungen")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", min_value=3, max_value=50, value=10, step=1)
reserve_m = st.sidebar.number_input("Reserven (Nachrücker)", min_value=0, max_value=20, value=2, step=1)
start_date = st.sidebar.date_input("Startdatum", value=datetime.today() - timedelta(days=1000))
end_date = st.sidebar.date_input("Enddatum", value=datetime.today())

# ---------------------------- #
#             App              #
# ---------------------------- #

st.title("📊 Momentum-RoboAdvisor")

uploaded = st.file_uploader("CSV mit **Ticker** und optional **Name** hochladen", type=["csv"])
tickers_txt = st.text_input("Oder Ticker eingeben (kommagetrennt):", "AAPL, MSFT, TSLA, NVDA")
portfolio_txt = st.text_input("(Optional) Aktuelle Portfolio-Ticker:", "AAPL")

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

with st.spinner("Lade Kursdaten …"):
    prices, volumes = fetch_ohlc(tickers, start_date, end_date)

if prices.empty:
    st.warning("Keine Kursdaten geladen.")
    st.stop()

df = compute_indicators(prices, volumes)
if df.empty:
    st.warning("Kennzahlen konnten nicht berechnet werden.")
    st.stop()

df["Name"] = df["Ticker"].map(name_map).fillna(df["Ticker"])
df["_GD50_dot"] = df["GD50-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))
df["_GD200_dot"] = df["GD200-Signal"].apply(lambda s: dot("#16a34a") if s.startswith("Über") else dot("#dc2626"))

# ---------------------------- #
#             Tabs             #
# ---------------------------- #

tab1, tab2, tab3 = st.tabs(["🔬 Analyse", "🧭 Handlungsempfehlungen", "📈 Backtest"])

with tab1:
    st.subheader("Analyse – alle Kennzahlen")
    df_sorted = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df_sorted["Rank"] = np.arange(1, len(df_sorted) + 1)
    st.dataframe(df_sorted, use_container_width=True)

with tab2:
    st.subheader("Handlungsempfehlungen (GD50/GD200 + Rank)")

    # Übersicht: Breadth & Exposure
    breadth = (df["GD200-Signal"] == "Über GD200").mean()
    exposure_now = breadth_to_exposure_tens(breadth)
    effective_holdings_now = max(0, int(round(top_n * exposure_now)))

    st.markdown(f"**Universe size:** {len(df)} Aktien")
    st.markdown(f"**Anteil über GD200 (Breadth):** {breadth:.0%}")
    st.markdown(f"**Discrete Exposure (10%-steps):** {int(exposure_now*10)} (von Top-{top_n})")
    st.markdown(f"**Aktuelle Anzahl geplanter Holdings (Top-N angepasst):** {effective_holdings_now} (von Top-{top_n})")

    rec_df = df.copy()
    rec_df["Handlung"] = rec_df.apply(lambda r: rec_row(r, in_port, top_n=top_n, reserve=reserve_m), axis=1)
    rec_df = rec_df.sort_values("Rank").reset_index(drop=True)
    rec_df["GD50"] = rec_df["_GD50_dot"]
    rec_df["GD200"] = rec_df["_GD200_dot"]
    cols = ["Rank", "Ticker", "Name", "Momentum-Score", "GD50", "GD200", "Handlung"]
    st.write(rec_df[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

with tab3:
    st.subheader("Backtest")

    eq = (prices.pct_change().mean(axis=1) + 1).cumprod()
    fig, ax = plt.subplots()
    ax.plot(eq.index, eq.values)
    ax.set_title("Equity-Kurve (Dummy)")
    st.pyplot(fig)

st.caption("Hinweis: Alles nur zu Informations- und Ausbildungszwecken.")
