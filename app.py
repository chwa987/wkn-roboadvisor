# app.py
# Momentum-RoboAdvisor v2.0 mit Champions 
# Mit Wahl: wöchentliches oder monatliches Rebalancing
# Analyse, Handlungsempfehlungen, Exposure-Übersicht, Backtest

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Momentum-RoboAdvisor", page_icon="📈", layout="wide")

# ============================================================
# Utils
# ============================================================

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_ohlc(ticker_list, start, end):
    """Download OHLCV für Ticker-Liste; gibt (price_df, volume_df) zurück."""
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

def volume_score(vol_series: pd.Series, lookback=60):
    if vol_series is None or vol_series.dropna().empty:
        return np.nan
    cur = vol_series.dropna().iloc[-1]
    base = vol_series.rolling(lookback, min_periods=max(5, lookback//5)).mean().iloc[-1]
    if base is None or base == 0 or pd.isna(base) or pd.isna(cur):
        return np.nan
    return float(np.clip(cur / base, 0.5, 2.0))

def logp(x):
    if pd.isna(x):
        return np.nan
    return np.sign(x) * np.log1p(abs(x))

# ============================================================
# Indikatoren + Score
# ============================================================

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

        vol_sc = volume_score(v, 60)
        avg_vol = v.rolling(60).mean().iloc[-1] if not v.empty else np.nan

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
            0.20 * (0 if np.isnan(rs_z) else rs_z) +
            0.10 * (0 if np.isnan(vol_sc) else (vol_sc - 1.0))
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
            "Volumen-Score": round(vol_sc, 2) if not np.isnan(vol_sc) else np.nan,
            "Ø Volumen (60T)": round(avg_vol, 0) if not np.isnan(avg_vol) else np.nan,
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

# ============================================================
# Rebalancing
# ============================================================

def weekly_first_trading_days(idx: pd.DatetimeIndex):
    s = pd.Series(1, index=idx)
    grp = s.groupby(pd.Grouper(freq="W-MON"))
    firsts = []
    for _, g in grp:
        if not g.empty:
            firsts.append(g.index[0])
    return firsts

def monthly_first_trading_days(idx: pd.DatetimeIndex):
    s = pd.Series(1, index=idx)
    grp = s.groupby(pd.Grouper(freq="M"))
    firsts = []
    for _, g in grp:
        if not g.empty:
            firsts.append(g.index[0])
    return firsts

def run_backtest(prices, volumes, benchmark, start_date, end_date,
                 top_n=10, min_volume=5_000_000, max_dd52=-50,
                 max_volatility=2.0, apply_benchmark=True,
                 cost_bps=10.0, slippage_bps=5.0, mode="weekly"):

    idx = prices.index[(prices.index >= pd.to_datetime(start_date)) & (prices.index <= pd.to_datetime(end_date))]
    if len(idx) < 260:
        return pd.DataFrame(), pd.DataFrame()

    if mode == "weekly":
        rebal_days = weekly_first_trading_days(idx)
    else:
        rebal_days = monthly_first_trading_days(idx)

    rebal_days = [d for d in rebal_days if d >= idx.min() and d <= idx.max()]
    if len(rebal_days) < 2:
        return pd.DataFrame(), pd.DataFrame()

    port_val = 1.0
    weights_prev = pd.Series(0.0, index=prices.columns)
    equity, logs = [], []
    tc = (cost_bps + slippage_bps) / 10000.0

    for i in range(len(rebal_days)-1):
        asof, nxt = rebal_days[i], rebal_days[i+1]
        p_slice = prices.loc[:asof]
        v_slice = volumes.loc[:asof]

        bm_slice = None
        if benchmark is not None:
            bm_slice = pd.DataFrame({"BM": benchmark.loc[:asof].dropna()})
            if bm_slice.empty:
                bm_slice = None

        snap = compute_indicators(p_slice, v_slice, benchmark_df=bm_slice)
        if snap.empty:
            continue

        filt = snap.copy()
        filt = filt[filt["Ø Volumen (60T)"] >= min_volume]
        filt = filt[filt["52W-Drawdown (%)"] >= max_dd52]
        filt = filt[filt["Volatilität (ann.)"] <= max_volatility]
        if apply_benchmark and "RS vs Benchmark (%)" in filt.columns:
            filt = filt[filt["RS vs Benchmark (%)"] > 0]
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
        logs.append({"Date": asof, "NumHold": len(sel), "Turnover": turnover,
                     "GrossRet": gross_return, "Cost": cost, "NetRet": net_return, "PortVal": port_val})
        weights_prev = new_weights.copy()

    eq_df = pd.DataFrame(equity, columns=["Date", "Equity"]).set_index("Date")
    logs_df = pd.DataFrame(logs)
    return eq_df, logs_df

# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("⚙️ Einstellungen")
top_n = st.sidebar.number_input("Top-N (Kernpositionen)", min_value=3, max_value=50, value=10, step=1)
start_date = st.sidebar.date_input("Startdatum (Datenabruf)", value=datetime.today() - timedelta(days=900))
end_date   = st.sidebar.date_input("Enddatum", value=datetime.today())

st.sidebar.markdown("### Filter")
min_volume = st.sidebar.number_input("Min. Ø Volumen (60T)", min_value=0, value=4_000_000, step=100_000)
max_dd52 = st.sidebar.slider("Max. Drawdown zum 52W-Hoch (%)", -100, 0, -50, step=5)
max_volatility = st.sidebar.slider("Max. Volatilität (ann.)", 0.0, 3.0, 2.0, step=0.05)
apply_benchmark = st.sidebar.checkbox("Nur Aktien > Benchmark (130T)", value=True)
benchmark_ticker = st.sidebar.text_input("Benchmark-Ticker", "SPY")

st.sidebar.markdown("### Backtest")
mode = st.sidebar.radio("Rebalancing-Modus", ["weekly", "monthly"], index=1)
cost_bps = st.sidebar.number_input("Kommission (bps)", min_value=0.0, value=10.0, step=1.0)
slip_bps = st.sidebar.number_input("Slippage (bps)", min_value=0.0, value=5.0, step=1.0)

# ============================================================
# Daten laden
# ============================================================

st.title("📊 Momentum-Analyse (erweitert) – Version 1.3")

uploaded = st.file_uploader("CSV mit **Ticker** und optional **Name**", type=["csv"])
tickers_txt = st.text_input("Oder Ticker (kommagetrennt):", "AAPL, MSFT, TSLA, NVDA, META, AVGO")
portfolio_txt = st.text_input("(Optional) Aktuelle Portfolio-Ticker:", "")

name_map = {}
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if "Ticker" in df_in.columns:
            if "Name" in df_in.columns:
                name_map = dict(zip(df_in["Ticker"].astype(str), df_in["Name"].astype(str)))
            tickers_txt = ", ".join(df_in["Ticker"].astype(str).tolist())
            st.success(f"{len(df_in)} Ticker aus CSV geladen.")
        else:
            st.error("CSV benötigt mindestens die Spalte 'Ticker'.")
    except Exception as e:
        st.error(f"CSV konnte nicht gelesen werden: {e}")

tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
portfolio = set([t.strip().upper() for t in portfolio_txt.split(",") if t.strip()])

if not tickers:
    st.stop()

with st.spinner("Lade Kursdaten …"):
    prices, volumes = fetch_ohlc(tickers, start_date, end_date)
    bm_prices, _ = fetch_ohlc([benchmark_ticker], start_date, end_date)

if prices.empty:
    st.warning("Keine Kursdaten geladen.")
    st.stop()

# ============================================================
# Analyse
# ============================================================

df = compute_indicators(prices, volumes, benchmark_df=bm_prices)
if df.empty:
    st.stop()

df["Name"] = df["Ticker"].map(name_map).fillna(df["Ticker"])

filtered = df.copy()
filtered = filtered[filtered["Ø Volumen (60T)"] >= min_volume]
filtered = filtered[filtered["52W-Drawdown (%)"] >= max_dd52]
filtered = filtered[filtered["Volatilität (ann.)"] <= max_volatility]
if apply_benchmark and "RS vs Benchmark (%)" in filtered.columns:
    filtered = filtered[filtered["RS vs Benchmark (%)"] > 0]
filtered = filtered.sort_values("Momentum-Score", ascending=False).reset_index(drop=True)
filtered["Rank"] = np.arange(1, len(filtered) + 1)

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3 = st.tabs(["🔬 Analyse", "🧭 Handlungsempfehlungen", "📈 Backtest"])

with tab1:
    st.subheader("Analyse – Kennzahlen (gefiltert)")
    st.dataframe(filtered, use_container_width=True)

with tab2:
    st.subheader("Handlungsempfehlungen")
    rec_df = filtered.copy()
    rec_df["Handlung"] = rec_df.apply(lambda r: "🟢 Kaufen" if r["Rank"] <= top_n else "—", axis=1)
    cols = ["Rank", "Ticker", "Name", "Momentum-Score", "GD50-Signal", "GD200-Signal", "Handlung"]
    st.write(rec_df[cols].to_html(escape=False, index=False), unsafe_allow_html=True)

with tab3:
    st.subheader(f"Backtest – {mode}")
    if st.button("▶️ Backtest starten"):
        with st.spinner("Berechne Backtest …"):
            bm_series = bm_prices.iloc[:, 0].copy() if not bm_prices.empty else None
            eq_df, logs_df = run_backtest(
                prices, volumes, bm_series, start_date, end_date,
                top_n=top_n, min_volume=min_volume,
                max_dd52=max_dd52, max_volatility=max_volatility,
                apply_benchmark=apply_benchmark, cost_bps=cost_bps,
                slippage_bps=slip_bps, mode=mode
            )
        if eq_df.empty:
            st.warning("Backtest lieferte keine Werte (zu wenig Daten oder zu strenge Filter?).")
        else:
            # Benchmark-Kurve
            fig, ax = plt.subplots(figsize=(9,4))
            ax.plot(eq_df.index, eq_df["Equity"], label="Strategie")
            if bm_prices is not None and not bm_prices.empty:
                bm_norm = bm_prices.iloc[:,0].loc[eq_df.index.min():eq_df.index.max()].dropna()
                if not bm_norm.empty:
                    bm_norm = (bm_norm / bm_norm.iloc[0])
                    ax.plot(bm_norm.index, bm_norm.values, label=f"Benchmark ({benchmark_ticker})", alpha=0.8)
            ax.set_title(f"Equity-Kurve ({mode}-Rebalancing)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            st.markdown("**Rebalance-Log (pro Periode):**")
            st.dataframe(logs_df, use_container_width=True)
            st.download_button("📥 Logs (CSV)", logs_df.to_csv(index=False).encode("utf-8"),
                               "backtest_logs.csv", "text/csv")
            st.download_button("📥 Equity (CSV)", eq_df.to_csv().encode("utf-8"),
                               "backtest_equity.csv", "text/csv")

st.caption("Nur Informations- und Ausbildungszwecke. Keine Anlageempfehlung.")

# ============================================================
# Champions-Analyse (Version 2.0)
# ============================================================

import math

def compute_champions_scores(tickers_df: pd.DataFrame, start, end):
    """
    Erwartet DataFrame mit Spalten: 'Ticker', 'Name'
    Holt 10 Jahre Kursdaten von yfinance und berechnet:
    - GeoPAK10
    - Verlust-Ratio
    - Gewinnkonstanz
    - Sicherheitsscore
    - Buffett-Kriterien (vereinfachte Approximation)
    """
    results = []
    total = len(tickers_df)
    success = 0

    for _, row in tickers_df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        name = str(row.get("Name", "")).strip()

        if not ticker:
            continue

        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data is None or data.empty or len(data) < 252*10:
                # Weniger als 10 Jahre Historie
                results.append({
                    "Ticker": ticker, "Name": name,
                    "Geo-PAK10": np.nan, "Verlust-Ratio": np.nan,
                    "Gewinnkonstanz": np.nan, "Sicherheitsscore": np.nan,
                    "Buffett-Signal": "—"
                })
                continue

            # Jahres-Returns
            yearly = data["Close"].resample("Y").last().pct_change().dropna()
            if len(yearly) < 10:
                # Nicht genug Jahre für GEO-PAK10
                results.append({
                    "Ticker": ticker, "Name": name,
                    "Geo-PAK10": np.nan, "Verlust-Ratio": np.nan,
                    "Gewinnkonstanz": np.nan, "Sicherheitsscore": np.nan,
                    "Buffett-Signal": "—"
                })
                continue

            # GeoPAK10
            geo = (np.prod(1 + yearly.tail(10)) ** (1/10) - 1) * 100

            # Verlust-Ratio (Durchschnitt Verluste / Durchschnitt Gewinne)
            gains = yearly[yearly > 0].mean() if not yearly[yearly > 0].empty else np.nan
            losses = yearly[yearly < 0].mean() if not yearly[yearly < 0].empty else np.nan
            verlustratio = abs(losses / gains) if (not math.isnan(gains) and not math.isnan(losses) and gains != 0) else np.nan

            # Gewinnkonstanz (% positive Jahre)
            gk = (len(yearly[yearly > 0]) / len(yearly)) * 100

            # Sicherheitsscore
            if not math.isnan(geo) and not math.isnan(verlustratio) and not math.isnan(gk):
                score = (geo ** 0.8) * (verlustratio ** -1.2) * ((gk / 100) ** 1.5)
            else:
                score = np.nan

            # Buffett-Signal (sehr vereinfacht, hier nur Score-Proxy)
            buffett_signal = "Ja" if (not math.isnan(score) and score > 5) else "Nein"

            results.append({
                "Ticker": ticker,
                "Name": name,
                "Geo-PAK10": round(geo, 2) if not math.isnan(geo) else np.nan,
                "Verlust-Ratio": round(verlustratio, 2) if not math.isnan(verlustratio) else np.nan,
                "Gewinnkonstanz": round(gk, 1) if not math.isnan(gk) else np.nan,
                "Sicherheitsscore": round(score, 3) if not math.isnan(score) else np.nan,
                "Buffett-Signal": buffett_signal
            })
            success += 1

        except Exception:
            results.append({
                "Ticker": ticker, "Name": name,
                "Geo-PAK10": np.nan, "Verlust-Ratio": np.nan,
                "Gewinnkonstanz": np.nan, "Sicherheitsscore": np.nan,
                "Buffett-Signal": "—"
            })

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out = df_out.sort_values("Sicherheitsscore", ascending=False, na_position="last").reset_index(drop=True)
        df_out["Rank"] = np.arange(1, len(df_out) + 1)
    return df_out, success, total


# ---------------------------- #
# Champions-Tab
# ---------------------------- #

tab4 = st.tabs(["🏆 Champions"])[0]

with tab4:
    st.subheader("🏆 Champions – Sicherheitsscore & Buffett-Kriterien")

    uploaded_champ = st.file_uploader("CSV mit Champions (Ticker, Name)", type=["csv"], key="champ_upload")

    if uploaded_champ is not None:
        try:
            df_champ_in = pd.read_csv(uploaded_champ)
            if not {"Ticker", "Name"}.issubset(df_champ_in.columns):
                st.error("CSV muss die Spalten 'Ticker' und 'Name' enthalten.")
            else:
                with st.spinner("Berechne Champions-Scores …"):
                    df_champ, success, total = compute_champions_scores(df_champ_in, start_date, end_date)

                if not df_champ.empty:

# ============================================================
# Champions-Analyse (Version 2.0 – FIX: immer 12y Historie)
# ============================================================

import math

def _read_champions_csv(file) -> pd.DataFrame:
    """Robustes Einlesen der Champions-CSV.
    Erwartet mindestens 'Ticker'. 'Name' ist optional."""
    import io
    # 1) Versuch: Standard
    try:
        df = pd.read_csv(file)
    except Exception:
        # 2) Fallback: Semikolon
        file.seek(0)
        df = pd.read_csv(file, sep=";")

    # Spalten vereinheitlichen
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df.columns = [cols_lower[c] for c in df.columns]

    # Mindestens 'ticker' muss existieren
    if "ticker" not in df.columns:
        # Falls erstes Feld Ticker enthält (Notlösung)
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "ticker"})
    if "name" not in df.columns:
        df["name"] = df["ticker"]

    # Aufräumen
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"])
    return df[["ticker", "name"]]


def compute_champions_scores(tickers_df: pd.DataFrame, end):
    """
    Holt ca. 12 Jahre Historie via yfinance (unabhängig von der Sidebar)
    und berechnet: GeoPAK10, Verlust-Ratio, Gewinnkonstanz, Sicherheitsscore.
    """
    end = pd.to_datetime(end)
    start_12y = end - pd.DateOffset(years=12)

    results = []
    total = len(tickers_df)
    parsed_ok = total  # bereits sauber eingelesen
    success = 0

    for _, row in tickers_df.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        name = str(row["name"]).strip()

        # Defensive Defaults
        res = {
            "Ticker": ticker, "Name": name,
            "Sicherheitsscore": np.nan, "Geo-PAK10": np.nan,
            "Verlust-Ratio": np.nan, "Gewinnkonstanz": np.nan,
            "Buffett-Signal": "—"
        }

        try:
            data = yf.download(
                ticker,
                start=start_12y,
                end=end + pd.Timedelta(days=1),
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if data is None or data.empty:
                results.append(res)
                continue

            # Jahres-Returns
            yearly = data["Close"].resample("Y").last().pct_change().dropna()
            if len(yearly) < 10:
                results.append(res)
                continue

            # ---- Kennzahlen ----
            # GeoPAK10
            geo = (np.prod(1 + yearly.tail(10)) ** (1/10) - 1) * 100

            # Verlust-Ratio = Durchschnitt Verluste / Durchschnitt Gewinne (Betrag)
            pos = yearly[yearly > 0]
            neg = yearly[yearly < 0]
            gains = pos.mean() if not pos.empty else np.nan
            losses = neg.mean() if not neg.empty else np.nan
            verlustratio = abs(losses / gains) if (pd.notna(gains) and pd.notna(losses) and gains != 0) else np.nan

            # Gewinnkonstanz = % positive Jahre
            gk = (len(pos) / len(yearly)) * 100

            # Sicherheitsscore nach deiner Formel
            if all(pd.notna([geo, verlustratio, gk])) and verlustratio > 0:
                score = (geo ** 0.8) * (verlustratio ** -1.2) * ((gk / 100) ** 1.5)
            else:
                score = np.nan

            # Buffett-Signal (Info, kein Ranking-Kriterium)
            buffett = "Ja" if (pd.notna(score) and score >= 6) else "Nein"

            res.update({
                "Sicherheitsscore": round(float(score), 3) if pd.notna(score) else np.nan,
                "Geo-PAK10": round(float(geo), 2) if pd.notna(geo) else np.nan,
                "Verlust-Ratio": round(float(verlustratio), 2) if pd.notna(verlustratio) else np.nan,
                "Gewinnkonstanz": round(float(gk), 1) if pd.notna(gk) else np.nan,
                "Buffett-Signal": buffett
            })
            success += 1

        except Exception:
            pass  # res bleibt mit NaN

        results.append(res)

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        # Score vor GeoPAK10 in der Anzeige
        df_out = df_out.sort_values("Sicherheitsscore", ascending=False, na_position="last").reset_index(drop=True)
        df_out["Rank"] = np.arange(1, len(df_out) + 1)
    return df_out, parsed_ok, success, total


# ---------------------------- #
# Champions-Tab (UI)
# ---------------------------- #

tab4 = st.tabs(["🏆 Champions"])[0]
with tab4:
    st.subheader("🏆 Champions – Sicherheitsscore (live, 12 Jahre Historie)")

    uploaded_champ = st.file_uploader("CSV mit Champions (mind. Spalte 'Ticker'; optional 'Name')",
                                      type=["csv"], key="champ_upload_v2")

    if uploaded_champ is not None:
        try:
            df_in = _read_champions_csv(uploaded_champ)
            parsed = len(df_in)
            st.markdown(f"**{parsed}/{parsed} ausgelesen.**")

            with st.spinner("Berechne Champions-Scores …"):
                df_champ, parsed_ok, success, total = compute_champions_scores(df_in, end=end_date)

            if not df_champ.empty:
                cols = ["Rank", "Ticker", "Name", "Sicherheitsscore", "Geo-PAK10",
                        "Verlust-Ratio", "Gewinnkonstanz", "Buffett-Signal"]
                st.dataframe(df_champ[cols], use_container_width=True)

                st.markdown(f"**Erfolgreich berechnet:** {success}/{total} (≥10 Jahre Historie)")
                st.download_button("📥 Ergebnisse (CSV)",
                                   df_champ[cols].to_csv(index=False).encode("utf-8"),
                                   "champions_scores.csv",
                                   "text/csv")
            else:
                st.warning("Keine Champions-Scores berechnet (zu wenig Historie?).")

        except Exception as e:
            st.error(f"Fehler beim Laden/Berechnen: {e}")
    else:
        st.info("Bitte CSV hochladen. Mindestens 'Ticker' (Name optional).")
