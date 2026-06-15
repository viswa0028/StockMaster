"""
streamlit_app.py
----------------
StockMaster Live Dashboard — displays real-time ML-ranked BUY signals.

Features:
  • Fetches top-20 BUY signals from FastAPI /signals every 10 minutes (auto-refresh)
  • Shows ranked table with Final_Score, RSI, Close price, and per-model scores
  • Visual score bar chart using st.bar_chart
  • Color-coded RSI indicators (green = strong, amber = borderline)
  • Last updated timestamp and live countdown timer

Run with:
    streamlit run streamlit_app.py
"""

import time
import requests
import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
API_BASE_URL   = "http://localhost:8000"
REFRESH_SECS   = 600          # 10 minutes between auto-refreshes
TOP_N_DISPLAY  = 20           # Show top 20

st.set_page_config(
    page_title="StockMaster — Live Buy Signals",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero-subtitle {
    font-size: 0.95rem;
    opacity: 0.7;
    margin-top: 4px;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 18px 24px;
    flex: 1;
    color: white;
}
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 2rem; font-weight: 700; margin-top: 4px; }

/* Ranked table styling */
.rank-number { font-weight: 700; color: #f5c518; font-size: 1.1rem; }
.score-high  { color: #00d084; font-weight: 600; }
.score-mid   { color: #ffd700; font-weight: 600; }
.score-low   { color: #ff6b6b; font-weight: 600; }
.rsi-hot     { background: #1e3a2f; color: #00d084; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; }
.rsi-warm    { background: #3a2e1e; color: #ffd700; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; }

/* Status badge */
.status-live   { display: inline-block; background: #00d084; color: #000; font-size: 0.7rem;
                  font-weight: 700; padding: 3px 10px; border-radius: 99px; letter-spacing: 1px; }
.status-stale  { display: inline-block; background: #ff6b6b; color: #000; font-size: 0.7rem;
                  font-weight: 700; padding: 3px 10px; border-radius: 99px; letter-spacing: 1px; }
.countdown     { font-size: 0.85rem; color: #888; margin-top: 4px; }

div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helper: Fetch signals from FastAPI
# ─────────────────────────────────────────────────────────────
def fetch_signals():
    try:
        resp = requests.get(f"{API_BASE_URL}/signals", timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.error(f"Could not connect to API ({API_BASE_URL}): {e}")
    return None


def fetch_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# Helper: Score colour
# ─────────────────────────────────────────────────────────────
def score_color(val):
    if val >= 0.65:
        return "background-color: #0d2b1e; color: #00d084"
    elif val >= 0.50:
        return "background-color: #2b2a0d; color: #ffd700"
    else:
        return "background-color: #2b0d0d; color: #ff6b6b"


# ─────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────
if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = 0
if "signals_data" not in st.session_state:
    st.session_state.signals_data = None


# ─────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">📈 StockMaster</p>
    <p class="hero-subtitle">Real-Time ML Buy Signal Dashboard &nbsp;|&nbsp; NSE Nifty 500 &nbsp;|&nbsp; Minervini Trend + Ensemble Ranking</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Top Controls
# ─────────────────────────────────────────────────────────────
col_refresh, col_trigger, col_top = st.columns([2, 2, 1])

with col_refresh:
    do_refresh = st.button("🔄  Refresh Now", use_container_width=True)

with col_trigger:
    do_trigger = st.button("⚡  Trigger Live Sync", use_container_width=True)
    if do_trigger:
        try:
            r = requests.post(f"{API_BASE_URL}/trigger-sync", timeout=5)
            if r.status_code == 200:
                st.success("Live sync triggered! Data will refresh in ~2 minutes.")
            else:
                st.warning(f"API returned {r.status_code}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

with col_top:
    top_n = st.selectbox("Show top", [10, 20], index=1)


# ─────────────────────────────────────────────────────────────
# Auto-Fetch Logic
# ─────────────────────────────────────────────────────────────
now = time.time()
if do_refresh or (now - st.session_state.last_fetch_time > REFRESH_SECS):
    with st.spinner("Fetching latest signals..."):
        result = fetch_signals()
        if result:
            st.session_state.signals_data = result
            st.session_state.last_fetch_time = now


# ─────────────────────────────────────────────────────────────
# Health / Status Bar
# ─────────────────────────────────────────────────────────────
health = fetch_health()
st.markdown("---")

h_col1, h_col2, h_col3, h_col4 = st.columns(4)
with h_col1:
    if health:
        badge = '<span class="status-live">LIVE</span>' if health["redis"] == "connected" else '<span class="status-stale">REDIS DOWN</span>'
        st.markdown(f"**Redis** &nbsp; {badge}", unsafe_allow_html=True)
    else:
        st.markdown("**API** &nbsp; <span class='status-stale'>OFFLINE</span>", unsafe_allow_html=True)

with h_col2:
    if health:
        st.markdown(f"**Signals Cached** &nbsp; `{health.get('signals_cached', 0)}`")

with h_col3:
    if health:
        st.markdown(f"**Last Updated** &nbsp; `{health.get('signals_updated', '—')}`")

with h_col4:
    next_refresh = max(0, int(REFRESH_SECS - (now - st.session_state.last_fetch_time)))
    st.markdown(f"**Auto-refresh in** &nbsp; `{next_refresh}s`")

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# Main Dashboard — Signals Table
# ─────────────────────────────────────────────────────────────
data = st.session_state.signals_data

if data and data.get("signals"):
    signals = data["signals"]
    df = pd.DataFrame(signals).head(top_n)

    # ── Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Total Signals", len(signals))
    m2.metric("🎯 Top Score", f"{df['Final_Score'].max():.4f}")
    m3.metric("📊 Avg Score", f"{df['Final_Score'].mean():.4f}")
    m4.metric("📈 Avg RSI", f"{df['RSI'].mean():.1f}")

    st.markdown(f"### 🏅 Top {top_n} BUY Candidates")
    st.caption(f"Updated: {data.get('updated_at', 'Unknown')} &nbsp;|&nbsp; Minervini trend filter + LGBM/XGB/CatBoost/HGB ensemble")

    # ── Styled table display
    display_df = df[["Rank", "Symbol", "Close", "RSI", "Final_Score",
                      "LGBM_Score", "XGB_Score", "CAT_Score", "HGB_Score"]].copy()

    styled = display_df.style.format({
        "Close":       "{:.2f}",
        "RSI":         "{:.1f}",
        "Final_Score": "{:.4f}",
        "LGBM_Score":  "{:.4f}",
        "XGB_Score":   "{:.4f}",
        "CAT_Score":   "{:.4f}",
        "HGB_Score":   "{:.4f}",
    }).background_gradient(
        subset=["Final_Score"], cmap="RdYlGn", vmin=0.3, vmax=0.9
    ).background_gradient(
        subset=["RSI"], cmap="YlOrRd", vmin=40, vmax=90
    ).set_properties(**{"text-align": "center"})

    st.dataframe(styled, use_container_width=True, height=600)

    # ── Bar chart of Final Scores
    st.markdown("### 📊 Final Score Distribution")
    chart_df = df.set_index("Symbol")[["Final_Score"]].sort_values("Final_Score", ascending=True)
    st.bar_chart(chart_df, height=400)

    # ── Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download CSV",
        data=csv,
        file_name=f"buy_signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

elif data and not data.get("signals"):
    st.info("⏳ No signals yet. Market may be closed or the live sync is pending.")
    st.markdown("Try clicking **⚡ Trigger Live Sync** to compute signals immediately.")
else:
    st.warning("Could not load signals. Make sure the FastAPI server is running:")
    st.code("uvicorn live_trading_api:app --host 0.0.0.0 --port 8000", language="bash")


# ─────────────────────────────────────────────────────────────
# Auto-Rerun after REFRESH_SECS
# ─────────────────────────────────────────────────────────────
time.sleep(1)
st.rerun()
