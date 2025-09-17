import streamlit as st
import pandas as pd

st.set_page_config(page_title="WKN RoboAdvisor", layout="wide")
st.title("📈 WKN RoboAdvisor – Mehrere Aktien automatisch bewerten")

st.markdown("Gib unten eine Liste von WKNs ein (eine pro Zeile) und klicke auf **Bewerten**, um eine automatische Analyse zu starten.")

# Eingabe
default_wkns = "A1CX3T\nA2QK1G\n881103\nA3DRA5\n938914\nA0B7FY\nA1W6XZ\nA3C9X6\nA1JVSC\n850663"
wkn_input = st.text_area("📥 WKN-Liste (eine pro Zeile)", default_wkns)
wkn_list = [w.strip() for w in wkn_input.splitlines() if w.strip()]

# Bewertung ausführen
if st.button("🔍 Bewerten"):
    result_data = []

    for wkn in wkn_list:
        # 🔧 Hier echte Daten einbinden (API etc.)
        sentiment_score = 7  # Dummy-Wert
        risk_score = 4       # Dummy-Wert
        trade_score = round((sentiment_score - risk_score) * 1.2, 2)

        if trade_score > 5:
            signal = "🟢 Grün – Halten oder Kaufen"
        elif trade_score >= 0:
            signal = "🟡 Gelb – Beobachten / Vorsicht"
        else:
            signal = "🔴 Rot – Verkauf erwägen"

        result_data.append({
            "WKN": wkn,
            "Sentiment-Score": sentiment_score,
            "Risiko-Score": risk_score,
            "Trade-Score": trade_score,
            "Signal": signal
        })

    df = pd.DataFrame(result_data)
    st.markdown("---")
    st.subheader("📊 Ergebnisübersicht")
    st.dataframe(df, use_container_width=True)
