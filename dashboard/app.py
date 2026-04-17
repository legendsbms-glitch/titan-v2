"""
TITAN v2.0 — Streamlit Dashboard
Real-time gold intelligence display
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="TITAN v2.0 — Gold Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a0a; }
    .verdict-box { 
        padding: 20px; border-radius: 10px;
        text-align: center; font-size: 2em; font-weight: bold;
    }
    .buy  { background: linear-gradient(135deg, #00ff88, #00cc66); color: #000; }
    .sell { background: linear-gradient(135deg, #ff4444, #cc0000); color: #fff; }
    .neutral { background: linear-gradient(135deg, #888, #555); color: #fff; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60/gold/000?text=TITAN+v2.0", use_container_width=True)
st.sidebar.title("⚡ TITAN v2.0")
st.sidebar.markdown("**9-Engine Gold Intelligence**")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5min)", value=False)
if auto_refresh:
    import time
    st.rerun()


# ─── Main Header ───────────────────────────────────────────────────────────────
st.title("⚡ TITAN v2.0 — Gold Intelligence System")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

col1, col2, col3 = st.columns([1, 1, 1])


# ─── Run Analysis Button ───────────────────────────────────────────────────────
if st.button("🚀 Run Full Analysis (All 9 Engines)", type="primary"):
    with st.spinner("Running TITAN analysis..."):
        try:
            from core.titan_runner import titan_analyze
            verdict = titan_analyze(verbose=False)
            st.session_state["last_verdict"] = verdict
        except Exception as e:
            st.error(f"Analysis error: {e}")


# ─── Display Verdict ───────────────────────────────────────────────────────────
verdict = st.session_state.get("last_verdict", None)

if verdict:
    direction  = verdict.get("direction", "NEUTRAL")
    confidence = verdict.get("confidence", 0)
    tradeable  = verdict.get("tradeable", False)

    col1, col2, col3 = st.columns(3)

    with col1:
        css_class = direction.lower() if direction in ["BUY", "SELL"] else "neutral"
        emoji = "📈" if direction == "BUY" else "📉" if direction == "SELL" else "➡️"
        st.markdown(
            f'<div class="verdict-box {css_class}">{emoji} {direction}</div>',
            unsafe_allow_html=True
        )

    with col2:
        conf_color = "green" if confidence > 0.7 else "orange" if confidence > 0.55 else "red"
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)

    with col3:
        if tradeable:
            st.success("✅ TRADE APPROVED")
        else:
            st.error("🚫 TRADE BLOCKED")
            for block in verdict.get("hard_blocks", []):
                st.warning(f"• {block}")

    # SL/TP
    if verdict.get("entry"):
        st.subheader("📍 Trade Levels")
        c1, c2, c3 = st.columns(3)
        c1.metric("Entry", f"${verdict.get('entry', 'N/A'):.2f}" if verdict.get('entry') else "N/A")
        c2.metric("Stop Loss", f"${verdict.get('sl', 'N/A'):.2f}" if verdict.get('sl') else "N/A")
        c3.metric("Take Profit", f"${verdict.get('tp', 'N/A'):.2f}" if verdict.get('tp') else "N/A")

    # Engine votes chart
    st.subheader("📊 Engine Vote Breakdown")
    vb = verdict.get("vote_breakdown", {})
    if vb:
        fig = go.Figure(go.Bar(
            x=list(vb.keys()),
            y=list(vb.values()),
            marker_color=["#00ff88" if k == "BUY" else "#ff4444" if k == "SELL" else "#888"
                          for k in vb.keys()]
        ))
        fig.update_layout(
            template="plotly_dark",
            title="Engine Signal Weights",
            yaxis_title="Weight",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # Engine details table
    st.subheader("🔧 Engine Details")
    engine_details = verdict.get("engine_details", {})
    if engine_details:
        df_engines = pd.DataFrame([
            {
                "Engine": name,
                "Signal": v.get("signal", "?"),
                "Confidence": f"{v.get('confidence', 0):.1%}",
                "Weight": f"{v.get('weight', 0):.3f}",
                "Contribution": v.get("contribution", 0),
            }
            for name, v in engine_details.items()
        ])
        st.dataframe(df_engines, use_container_width=True)

    # Warnings
    if verdict.get("soft_warnings"):
        st.subheader("⚠️ Risk Warnings")
        for w in verdict["soft_warnings"]:
            st.warning(w)

else:
    st.info("👆 Click **Run Full Analysis** to start TITAN")
    st.markdown("""
    ### 🔧 Engines
    | # | Engine | Description |
    |---|--------|-------------|
    | 1 | Price Matrix | Multi-TF structure, FVG, key levels |
    | 2 | Sentiment Fusion | FinBERT + Fed + macro events |
    | 3 | Volume & COT | CFTC COT, volume profile, ETF flows |
    | 4 | Macro Correlation | DXY, yields, VIX, cross-asset |
    | 5 | Liquidity Hunt ★ | Stop mapping, manipulation detection |
    | 6 | Regime Detection | HMM, volatility, Wyckoff cycle |
    | 7 | Adversarial Trap ★ | Institutional playbook matching |
    | 8 | Memory & Learning ★ | Trade journal, mistake classification |
    | 9 | Meta-Learning ★ | Dynamic weight optimization |
    """)


# ─── Trade Journal ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📔 Trade Journal")

try:
    from core.db import get_conn, init_db
    init_db()
    conn = get_conn()
    trades_df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
    conn.close()

    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True)

        # Equity curve
        if "pnl" in trades_df.columns:
            closed = trades_df.dropna(subset=["pnl"])
            if not closed.empty:
                closed["cumulative_pnl"] = closed["pnl"].astype(float)[::-1].cumsum()[::-1]
                fig_eq = px.line(
                    closed, y="cumulative_pnl",
                    title="Equity Curve",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No trades yet. Use POST /api/trades to log trades.")

except Exception as e:
    st.warning(f"Trade data unavailable: {e}")
