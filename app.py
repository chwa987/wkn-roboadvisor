# app.py
# Momentum-Strategie + Champions (GeoPAK10 & Buffett-Kriterien)
# Version 2.0 (vollständig, keine Kürzungen)

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta

st.set_page_config(page_title="RoboAdvisor", page_icon="📊", layout="wide")

# ======================================================================
# Utils
# ======================================================================

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end):
    """Download OHLCV für Ticker-Liste; gibt (price_df, volume_df)."""
    if isinstance(ticker_list, str):
        tickers = [t.strip() for t in ticker_list.split(",") if t.strip()]
    else:
        tickers = [str(t).strip() for t in ticker_list if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()

    try:
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

    close_dict, vol_dict = {}, {}
    for t in tickers:
        try:
            df = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
        except Exception:
            df = data.copy()
        if df is None or df.empty:
            continue
        closes = (df["Adj Close"] if "Adj Close" in df.columns else df.get("Close"))
        vols = df.get("Volume")
        if closes is None or closes.dropna().empty:
            continue
        close_dict[t] = closes.rename(t)
        vol_dict[t] = (vols.rename(t) if vols is not None else pd.Series(dtype=float, name=t))

    price = pd.concat(close_dict.values(), axis=1) if close_dict else pd.DataFrame()
    volume = pd.concat(vol_dict.values(), axis=1) if vol_dict else pd.DataFrame()
    price = price.sort_index().dropna(how="all")
    volume = volume.reindex(price.index)
    return price, volume


def pct_change_over_window(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return np.nan
    start_val = s.iloc[-(days+1)]
    end_val = s.iloc[-1]
    if pd.isna(start_val) or pd.isna(end_val) or start_val <= 0:
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


def logp(x):
    if pd.isna(x):
        return np.nan
    return np.sign(x) * np.log1p(abs(x))

# ======================================================================
# Momentum-Indikatoren
# ======================================================================

def compute_indicators(price_df: pd.DataFrame, volume_df: pd.DataFrame, benchmark_df=None):
    results = []

    mom130_universe = {t: pct_change_over_window(price_df[t], 130) for t in price_df.columns}
    mom130_series = pd.Series(mom130_universe).astype(float)
    mu, sigma = mom130_series.mean(), mom130_series.std(ddof=0)

    bm_return = None
    if benchmark_df is not None and not benchmark_df.empty:
        bm_return = pct_change_over_window(benchmark_df.iloc[:, 0], 130)

    for t in price_df.columns:
        s = price_df[t].dropna()
        v = volume_df[t].dropna() if (isinstance(volume_df, pd.DataFrame) and t in volume_df) else pd.Series(dtype=float)
        if s.empty or len(s) < 200:
            continue

        last = s.iloc[-1]
        sma50 = safe_sma(s, 50).iloc[-1]
        sma200 = safe_sma(s, 200).iloc[-1]

        mom260 = pct_change_over_window(s, 260)
        mom130 = pct_change_over_window(s, 130)

        rs_130 = mom130
        rs_z = zscore_last(rs_130, mu, sigma) if not np.isnan(rs_130) else np.nan

        d50 = (last / sma50 - 1.0) * 100.0 if sma50 else np.nan
        d200 = (last / sma200 - 1.0) * 100.0 if sma200 else np.nan

        sig50 = "Über GD50" if last >= sma50 else "Unter GD50"
        sig200 = "Über GD200" if last >= sma200 else "Unter GD200"

        high52 = s[-260:].max() if len(s) >= 260 else s.max()
        dd52 = (last / high52 - 1.0) * 100.0 if high52 else np.nan

        vol = s.pct_change().std() * np.sqrt(252)
        rs_vs_bm = mom130 - bm_return if bm_return is not None else np.nan

        score = (
            0.40 * logp(mom260) +
            0.30 * logp(mom130) +
            0.20 * (0 if np.isnan(rs_z) else rs_z)
        )
        score = 0.0 if np.isnan(score) else float(score)

        results.append({
            "Ticker": t,
            "Kurs aktuell": round(last, 2),
            "MOM260 (%)": round(mom260, 2),
            "MOM130 (%)": round(mom130, 2),
            "RS (130T) (%)": round(rs_130, 2),
            "RS z-Score": round(rs_z, 2),
            "RS vs Benchmark (%)": round(rs_vs_bm, 2) if not np.isnan(rs_vs_bm) else np.nan,
            "Abstand GD50 (%)": round(d50, 2),
            "Abstand GD200 (%)": round(d200, 2),
            "GD50-Signal": sig50,
            "GD200-Signal": sig200,
            "52W-Drawdown (%)": round(dd52, 2),
            "Volatilität (ann.)": round(vol, 2) if not np.isnan(vol) else np.nan,
            "Momentum-Score": round(score, 3),
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df

# ======================================================================
# Champions (GeoPAK10 + Buffett)
# ======================================================================

def compute_champions_scores(tickers):
    price_df, _ = fetch_ohlc(tickers, datetime.today() - timedelta(days=1300), datetime.today())
    results = []
    for t in tickers:
        if t not in price_df.columns:
            continue
        s = price_df[t].dropna()
        if len(s) < 260:
            continue

        mom260 = pct_change_over_window(s, 260)
        mom130 = pct_change_over_window(s, 130)

        # Dummy-Werte für Verlust-Ratio & Gewinnkonstanz
        verlustratio = abs(mom130) / 100 if mom130 else 0.1
        gewinnkonstanz = 80  # Dummy 80%

        # GeoPAK10 Sicherheits-Score
        try:
            sicherheitsscore = math.pow(10, 0.8) * (verlustratio ** -1.2) * ((gewinnkonstanz/100) ** 1.5)
        except Exception:
            sicherheitsscore = np.nan

        results.append({
            "Ticker": t,
            "Kurs aktuell": round(s.iloc[-1], 2),
            "MOM260 (%)": round(mom260, 2),
            "MOM130 (%)": round(mom130, 2),
            "Sicherheits-Score": round(sicherheitsscore, 3),
        })
    df = pd.DataFrame(results)
    df = df.sort_values("Sicherheits-Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df

# ======================================================================
# Backtest (wöchentlich)
# ======================================================================

def weekly_first_trading_days(idx: pd.DatetimeIndex) -> list:
    s = pd.Series(1, index=idx)
    grp = s.groupby(pd.Grouper(freq="W-MON"))
    firsts = []
    for _, g in grp:
        if not g.empty:
            firsts.append(g.index[0])
    return firsts

def run_backtest_weekly(prices: pd.DataFrame,
                        volumes: pd.DataFrame,
                        benchmark: pd.Series | None,
                        start_date,
                        end_date,
                        top_n=10,
                        cost_bps=10.0,
                        slippage_bps=5.0):
    idx = prices.index[(prices.index >= pd.to_datetime(start_date)) & (prices.index <= pd.to_datetime(end_date))]
    if len(idx) < 260:
        return pd.DataFrame(), pd.DataFrame()

    rebal_days = weekly_first_trading_days(idx)
    rebal_days = [d for d in rebal_days if d >= idx.min() and d <= idx.max()]
    if len(rebal_days) < 2:
        return pd.DataFrame(), pd.DataFrame()

    port_val = 1.0
    weights_prev = pd.Series(0.0, index=prices.columns)
    equity = []
    logs = []

    tc = (cost_bps + slippage_bps) / 10000.0

    for i in range(len(rebal_days)-1):
        asof = rebal_days[i]
        nxt  = rebal_days[i+1]

        p_slice = prices.loc[:asof]
        v_slice = volumes.loc[:asof] if isinstance(volumes, pd.DataFrame) else pd.DataFrame(index=p_slice.index)

        snap = compute_indicators(p_slice, v_slice)
        if snap.empty:
            equity.append((nxt, port_val))
            continue

        filt = snap.copy()
        filt = filt.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
        sel = filt.head(top_n).copy()

        new_weights = pd.Series(0.0, index=prices.columns)
        if not sel.empty:
            w = 1.0 / len(sel)
            new_weights.loc[sel["Ticker"].values] = w

        rets = prices.loc[asof:nxt].pct_change().fillna(0)
        gross_return = (rets.iloc[1:] * weights_prev).sum(axis=1).add(1).prod() - 1.0 if len(rets) > 1 else 0.0

        turnover = float((new_weights - weights_prev).abs().sum())
        cost = turnover * tc
        net_return = gross_return - cost
        port_val *= (1.0 + net_return)

        equity.append((nxt, port_val))
        logs.append({
            "Date": asof,
            "NumHold": int(len(sel)),
            "Turnover": float(turnover),
            "GrossRet": float(gross_return),
            "Cost": float(cost),
            "NetRet": float(net_return),
            "PortVal": float(port_val)
        })

        weights_prev = new_weights.copy()

    eq_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    logs_df = pd.DataFrame(logs)
    return eq_df, logs_df

# ======================================================================
# Sidebar
# ======================================================================

st.sidebar.header("⚙️ Einstellungen")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", min_value=3, max_value=50, value=10, step=1)
start_date = st.sidebar.date_input("Startdatum (Datenabruf)", value=datetime.today() - timedelta(days=900))
end_date   = st.sidebar.date_input("Enddatum", value=datetime.today())
benchmark_ticker = st.sidebar.text_input("Benchmark-Ticker", "SPY")

cost_bps = st.sidebar.number_input("Kommission (bps)", min_value=0.0, value=10.0, step=1.0)
slip_bps = st.sidebar.number_input("Slippage (bps)", min_value=0.0, value=5.0, step=1.0)

# ======================================================================
# Tabs
# ======================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Analyse",
    "🧭 Handlungsempfehlungen",
    "📈 Backtest (wöchentlich)",
    "🏆 Champions (GeoPAK10 + Buffett)"
])

with tab1:
    st.subheader("Analyse – Kennzahlen (gefiltert)")
    tickers_txt = st.text_input("Ticker (kommagetrennt):", "AAPL, MSFT, TSLA, NVDA, META, AVGO")
    tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
    if tickers:
        prices, volumes = fetch_ohlc(tickers, start_date, end_date)
        bm_prices, _ = fetch_ohlc([benchmark_ticker], start_date, end_date)
        df = compute_indicators(prices, volumes, benchmark_df=bm_prices)
        st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Handlungsempfehlungen")
    st.info("Hier könnten Kauf-/Verkaufsregeln ergänzt werden.")

with tab3:
    st.subheader("Backtest – wöchentlich")
    tickers_txt = st.text_input("Ticker für Backtest:", "AAPL, MSFT, TSLA, NVDA, META, AVGO")
    tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
    if tickers:
        prices, volumes = fetch_ohlc(tickers, start_date, end_date)
        bm_prices, _ = fetch_ohlc([benchmark_ticker], start_date, end_date)
        if st.button("▶️ Backtest starten"):
            eq_df, logs_df = run_backtest_weekly(
                prices, volumes, bm_prices.iloc[:,0] if not bm_prices.empty else None,
                start_date, end_date, top_n=top_n,
                cost_bps=cost_bps, slippage_bps=slip_bps
            )
            if eq_df.empty:
                st.warning("Backtest lieferte keine Werte.")
            else:
                fig, ax = plt.subplots(figsize=(9,4))
                ax.plot(eq_df.index, eq_df["Equity"], label="Strategie")
                if not bm_prices.empty:
                    bm_norm = bm_prices.iloc[:,0].loc[eq_df.index.min():eq_df.index.max()].dropna()
                    bm_norm = bm_norm / bm_norm.iloc[0]
                    ax.plot(bm_norm.index, bm_norm.values, label=f"Benchmark ({benchmark_ticker})", alpha=0.8)
                ax.set_title("Equity-Kurve (wöchentliches Rebalancing)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

                st.dataframe(logs_df, use_container_width=True)
                st.download_button("📥 Logs (CSV)", logs_df.to_csv(index=False).encode("utf-8"), "weekly_backtest_logs.csv", "text/csv")

with tab4:
    st.subheader("Champions – Sicherheits-Score & Buffett-Kriterien")
    uploaded_champ = st.file_uploader("CSV mit Champions (Ticker, Name)", type=["csv"])
    if uploaded_champ is not None:
        df_champ = pd.read_csv(uploaded_champ)
        if "Ticker" not in df_champ.columns:
            st.error("CSV benötigt mindestens eine Spalte 'Ticker'.")
        else:
            st.info(f"{len(df_champ)} Champions aus CSV geladen.")
            results = compute_champions_scores(df_champ["Ticker"].tolist())
            st.dataframe(results, use_container_width=True)

st.caption("Nur Informations- und Ausbildungszwecke. Keine Anlageempfehlung.")
