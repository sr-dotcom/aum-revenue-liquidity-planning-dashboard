"""
Aurora Utensils Manufacturing (AUM) – Executive Dashboard
Pixel-perfect aligned layout with fixed sidebar.

Run with: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Data loaders
try:
    from data_loader import (
        get_revenue_history_and_forecast, get_liquidity_risk_table,
        get_liquidity_config_summary, get_forecast_csv_bytes,
        get_liquidity_csv_bytes, check_data_availability,
    )
except ImportError:
    from app.data_loader import (
        get_revenue_history_and_forecast, get_liquidity_risk_table,
        get_liquidity_config_summary, get_forecast_csv_bytes,
        get_liquidity_csv_bytes, check_data_availability,
    )

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="AUM Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Design System CSS
# =============================================================================

st.markdown("""
<style>
    /* ==========================================================
       AURORA UTENSILS - PREMIUM DARK FINANCE THEME
       Unified Dark Background, Maximum Readability
       ========================================================== */
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* GLOBAL RESET */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        box-sizing: border-box;
    }
    
    /* MAIN PAGE - DARK SLATE BACKGROUND */
    .stApp {
        background: #0f1419 !important;
        background-image: none !important;
    }
    
    /* All text defaults to light */
    .stApp, .stApp * {
        color: #e7e9ea !important;
    }
    
    /* Hide Streamlit branding - but keep sidebar button */
    #MainMenu, footer, header { visibility: hidden !important; }
    
    /* FORCE SIDEBAR ALWAYS VISIBLE */
    [data-testid="stSidebar"] {
        transform: translateX(0) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Make sidebar collapse button VERY visible */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 9999 !important;
        width: 50px !important;
        height: 50px !important;
        background: #1da1f2 !important;
        border: 2px solid #ffffff !important;
        border-radius: 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="collapsedControl"] svg {
        width: 24px !important;
        height: 24px !important;
        color: #ffffff !important;
    }
    [data-testid="collapsedControl"]:hover {
        background: #0d8ed9 !important;
        transform: scale(1.05);
    }
    
    /* ==========================================================
       SIDEBAR - SLIGHTLY LIGHTER DARK
       ========================================================== */
    [data-testid="stSidebar"] {
        background: #16202a !important;
        border-right: 1px solid #2f3336 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 28px 24px !important;
        background: #16202a !important;
    }
    
    /* Sidebar text - pure white */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Hide radio label */
    [data-testid="stSidebar"] .stRadio > label { 
        display: none !important; 
    }
    
    /* Radio container */
    [data-testid="stSidebar"] .stRadio > div {
        display: flex !important;
        flex-direction: column !important;
        gap: 10px !important;
    }
    
    /* Navigation buttons */
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
        background: #1d2a3a !important;
        border: 1px solid #3b4a5a !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin: 0 !important;
        width: 100% !important;
        min-height: 54px !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
        background: #1da1f2 !important;
        border-color: #1da1f2 !important;
    }
    
    /* Hide radio circle */
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    
    /* Nav button text */
    [data-testid="stSidebar"] .stRadio label span {
        font-size: 15px !important;
        font-weight: 500 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar components */
    .sidebar-header {
        text-align: center;
        padding-bottom: 28px;
        border-bottom: 1px solid #2f3336;
        margin-bottom: 24px;
    }
    .sidebar-logo {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #1da1f2, #7856ff);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 16px auto;
        font-size: 30px;
        box-shadow: 0 8px 24px rgba(29,161,242,0.3);
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .sidebar-subtitle {
        font-size: 13px;
        color: #8899a6 !important;
        margin-top: 8px;
    }
    .sidebar-section-label {
        font-size: 11px;
        font-weight: 600;
        color: #8899a6 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 28px 0 14px 0;
    }
    
    /* Status rows */
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
        font-size: 14px;
        border-bottom: 1px solid #2f3336;
    }
    .status-row:last-child { border-bottom: none; }
    .status-name { color: #e7e9ea !important; }
    .status-indicator { display: flex; align-items: center; gap: 8px; }
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    .status-dot.online {
        background: #00ba7c;
        box-shadow: 0 0 12px rgba(0,186,124,0.6);
    }
    .status-dot.offline {
        background: #f4212e;
        box-shadow: 0 0 12px rgba(244,33,46,0.6);
    }
    .status-text { font-size: 13px; font-weight: 500; }
    .status-text.online { color: #00ba7c !important; }
    .status-text.offline { color: #f4212e !important; }
    
    .sidebar-divider {
        border-top: 1px solid #2f3336;
        margin: 28px 0;
    }
    
    /* ==========================================================
       MAIN CONTENT - DARK CARDS ON DARK BACKGROUND
       ========================================================== */
    
    /* Hero card */
    .hero-card {
        background: #16202a;
        border: 1px solid #2f3336;
        border-left: 5px solid #1da1f2;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 32px;
    }
    .hero-card h1 {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0 0 10px 0;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }
    .hero-card p {
        font-size: 16px;
        color: #8899a6 !important;
        margin: 0;
        font-weight: 400;
    }
    
    /* Force equal columns */
    [data-testid="column"] {
        flex: 1 1 0 !important;
        width: 0 !important;
        min-width: 0 !important;
    }
    
    /* KPI Cards */
    .u-card {
        background: #16202a;
        border: 1px solid #2f3336;
        border-radius: 16px;
        padding: 24px 28px;
        width: 100%;
        height: 140px;
        min-height: 140px;
        max-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.25s ease;
    }
    .u-card:hover {
        border-color: #1da1f2;
        background: #1a2836;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #8899a6 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff !important;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }
    .kpi-sub {
        font-size: 13px;
        color: #71767b !important;
        margin-top: 10px;
        font-weight: 400;
    }
    
    /* Risk badges - Vibrant colors */
    .badge {
        display: inline-block;
        padding: 12px 24px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-success {
        background: rgba(0,186,124,0.15) !important;
        color: #00ba7c !important;
        border: 2px solid #00ba7c;
    }
    .badge-warning {
        background: rgba(255,173,31,0.15) !important;
        color: #ffad1f !important;
        border: 2px solid #ffad1f;
    }
    .badge-danger {
        background: rgba(244,33,46,0.15) !important;
        color: #f4212e !important;
        border: 2px solid #f4212e;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 48px 0 24px 0;
        padding-bottom: 16px;
        border-bottom: 1px solid #2f3336;
    }
    .section-header::before {
        content: '';
        width: 5px;
        height: 28px;
        background: linear-gradient(180deg, #1da1f2, #7856ff);
        border-radius: 3px;
    }
    .section-header span {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff !important;
        letter-spacing: -0.3px;
    }
    
    /* Info cards */
    .info-card {
        background: #16202a;
        border: 1px solid #2f3336;
        border-left: 4px solid #1da1f2;
        border-radius: 12px;
        padding: 22px 28px;
    }
    .info-card-title {
        font-size: 14px;
        font-weight: 600;
        color: #1da1f2 !important;
        margin-bottom: 12px;
    }
    .info-card-text {
        font-size: 14px;
        color: #e7e9ea !important;
        line-height: 1.8;
        margin: 0;
    }
    .info-card-text strong {
        color: #ffffff !important;
        font-weight: 600;
    }
    .info-card-text em {
        color: #8899a6 !important;
        font-style: normal;
    }
    
    /* Insight cards */
    .insight-card {
        background: #16202a;
        border: 1px solid #2f3336;
        border-radius: 16px;
        padding: 26px 30px;
        height: 100%;
    }
    .insight-card h4 {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff !important;
        margin: 0 0 18px 0;
        padding-bottom: 14px;
        border-bottom: 1px solid #2f3336;
    }
    .insight-card ul {
        margin: 0;
        padding-left: 20px;
        font-size: 14px;
        color: #e7e9ea !important;
        line-height: 2.1;
    }
    .insight-card li {
        margin-bottom: 8px;
        color: #e7e9ea !important;
    }
    
    /* Chart container */
    .chart-box {
        background: #16202a;
        border: 1px solid #2f3336;
        border-radius: 16px;
        padding: 24px;
        margin: 24px 0;
    }
    
    /* ==========================================================
       FORM ELEMENTS - ALWAYS VISIBLE
       ========================================================== */
    
    /* ALL labels - light text */
    label, .stSlider label, .stNumberInput label, .stSelectbox label,
    .stRadio > label, .stCheckbox label, .stTextInput label {
        color: #e7e9ea !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #1da1f2 !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        color: #ffffff !important;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input {
        color: #ffffff !important;
        background: #1d2a3a !important;
        border: 1px solid #3b4a5a !important;
        border-radius: 10px !important;
    }
    .stTextInput input::placeholder {
        color: #71767b !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: #1d2a3a !important;
        border: 1px solid #3b4a5a !important;
        color: #ffffff !important;
    }
    
    /* Radio in main content */
    [data-testid="stHorizontalBlock"] .stRadio label {
        color: #e7e9ea !important;
        background: #1d2a3a !important;
        border: 1px solid #3b4a5a !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        margin: 0 8px 0 0 !important;
    }
    [data-testid="stHorizontalBlock"] .stRadio label:hover {
        border-color: #1da1f2 !important;
    }
    
    /* Checkbox */
    .stCheckbox span {
        color: #e7e9ea !important;
    }
    
    /* Caption text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #8899a6 !important;
        font-size: 13px !important;
    }
    
    /* ==========================================================
       DATA TABLES
       ========================================================== */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: #16202a !important;
        border: 1px solid #2f3336 !important;
    }
    .stDataFrame th {
        background: #1d2a3a !important;
        color: #ffffff !important;
    }
    .stDataFrame td {
        color: #e7e9ea !important;
        border-color: #2f3336 !important;
    }
    
    /* ==========================================================
       HORIZONTAL TAB NAVIGATION
       ========================================================== */
    
    /* Hide the sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Style horizontal radio buttons as tabs */
    .stRadio > div {
        display: flex !important;
        gap: 12px !important;
        flex-wrap: wrap !important;
    }
    
    .stRadio label[data-baseweb="radio"] {
        background: #1d2a3a !important;
        border: 2px solid #3b4a5a !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    
    .stRadio label[data-baseweb="radio"]:hover {
        background: #2a3a4a !important;
        border-color: #1da1f2 !important;
    }
    
    .stRadio label[data-baseweb="radio"][data-checked="true"] {
        background: #1da1f2 !important;
        border-color: #1da1f2 !important;
    }
    
    /* Hide the radio circle */
    .stRadio label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    
    .stRadio label span {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    
    /* ==========================================================
       BUTTONS
       ========================================================== */
    .stDownloadButton {
        text-align: center;
    }
    .stDownloadButton button {
        background: #1da1f2 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 14px 40px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.3px;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 16px rgba(29,161,242,0.3) !important;
    }
    .stDownloadButton button:hover {
        background: #1a91da !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 28px rgba(29,161,242,0.45) !important;
    }
    
    /* ==========================================================
       ALERTS
       ========================================================== */
    .stSuccess, [data-testid="stAlert"] {
        background: rgba(0,186,124,0.1) !important;
        border: 1px solid #00ba7c !important;
        border-radius: 12px !important;
    }
    .stSuccess p, .stSuccess span {
        color: #00ba7c !important;
        font-weight: 500 !important;
    }
    
    .stError {
        background: rgba(244,33,46,0.1) !important;
        border: 1px solid #f4212e !important;
        border-radius: 12px !important;
    }
    .stError p, .stError span {
        color: #f4212e !important;
        font-weight: 500 !important;
    }
    
    .stWarning {
        background: rgba(255,173,31,0.1) !important;
        border: 1px solid #ffad1f !important;
        border-radius: 12px !important;
    }
    .stWarning p, .stWarning span {
        color: #ffad1f !important;
        font-weight: 500 !important;
    }
    
    /* ==========================================================
       FOOTER
       ========================================================== */
    .footer {
        text-align: center;
        font-size: 13px;
        color: #71767b !important;
        padding: 48px 0 32px 0;
        margin-top: 64px;
        border-top: 1px solid #2f3336;
    }
    
    /* ==========================================================
       PLOTLY CHART DARK MODE
       ========================================================== */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }
    .js-plotly-plot .plotly .modebar-btn {
        color: #8899a6 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Component Functions
# =============================================================================

def hero(title: str, subtitle: str):
    st.markdown(f'''<div class="hero-card"><h1>{title}</h1><p>{subtitle}</p></div>''', unsafe_allow_html=True)

def kpi(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f'''<div class="u-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{sub_html}</div>'''

def kpi_badge(label: str, text: str, badge_type: str = "warning"):
    return f'''<div class="u-card" style="text-align:center;"><div class="kpi-label">{label}</div><div style="margin-top:6px;"><span class="badge badge-{badge_type}">{text}</span></div></div>'''

def section(title: str):
    st.markdown(f'<div class="section-header"><span>{title}</span></div>', unsafe_allow_html=True)

def info_card(title: str, text: str):
    return f'''<div class="info-card"><div class="info-card-title">{title}</div><p class="info-card-text">{text}</p></div>'''

def insight(title: str, items: list):
    li = "".join([f"<li>{i}</li>" for i in items])
    return f'''<div class="insight-card"><h4>{title}</h4><ul>{li}</ul></div>'''

def download_btn(label: str, data: bytes, fname: str, key: str = None):
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.download_button(label, data=data, file_name=fname, mime="text/csv", key=key)

def chart_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#e7e9ea', size=12),
        xaxis=dict(gridcolor='#2f3336', linecolor='#3b4a5a', tickfont=dict(color='#8899a6')),
        yaxis=dict(gridcolor='#2f3336', linecolor='#3b4a5a', tickfont=dict(color='#8899a6')),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=11, color='#e7e9ea')),
        margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(bgcolor="#1d2a3a", font_size=12, font_color="#ffffff", bordercolor="#3b4a5a")
    )

def fmt(val: float) -> str:
    if val >= 1e7: return f"₹{val/1e7:.2f} Cr"
    if val >= 1e5: return f"₹{val/1e5:.2f} L"
    return f"₹{val:,.0f}"

def badge_type(risk: str) -> str:
    return {"Safe": "success", "At Risk": "warning", "Critical": "danger"}.get(risk, "warning")

# =============================================================================
# NAVIGATION - TOP TABS (No sidebar dependency)
# =============================================================================

# Top navigation bar
st.markdown('''
<div style="background: #16202a; border: 1px solid #2f3336; border-radius: 16px; padding: 20px 32px; margin-bottom: 24px; display: flex; align-items: center; justify-content: space-between;">
    <div style="display: flex; align-items: center; gap: 16px;">
        <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #1da1f2, #7856ff); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px;">📊</div>
        <div>
            <div style="font-size: 18px; font-weight: 700; color: #ffffff;">AUM Executive Dashboard</div>
            <div style="font-size: 13px; color: #8899a6;">Aurora Utensils Manufacturing</div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# Tab navigation
page = st.radio(
    "Navigate",
    ["Overview", "Revenue Forecast", "Liquidity Risk"],
    horizontal=True,
    label_visibility="collapsed"
)

# =============================================================================
# Overview Page
# =============================================================================

def page_overview():
    hero("Aurora Utensils Manufacturing", "Executive Dashboard – Liquidity & Revenue Analytics")
    
    if not all(check_data_availability().values()):
        st.warning("Some data files are missing. Run Phase 5 and 6 notebooks first.")
        return
    
    try:
        df_hist, df_fc = get_revenue_history_and_forecast()
        summary = get_liquidity_config_summary()
    except Exception as e:
        st.error(str(e)); return
    
    last = df_hist.sort_values('year_month').iloc[-1]
    avg_fc = df_fc['revenue'].mean()
    
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.markdown(kpi("Last Actual Revenue", fmt(last['revenue']), last['year_month'].strftime('%b %Y')), unsafe_allow_html=True)
    c2.markdown(kpi("Avg Forecast Revenue", fmt(avg_fc), "Next 12 months"), unsafe_allow_html=True)
    c3.markdown(kpi("Cash Conversion Cycle", f"{summary['ccc_days']:.0f} days", "DSO + DIO − DPO"), unsafe_allow_html=True)
    c4.markdown(kpi_badge("Liquidity Status", summary['risk_band'], badge_type(summary['risk_band'])), unsafe_allow_html=True)
    
    st.caption("💡 **CCC**: Days cash is tied up in receivables and inventory before payables.")
    
    section("Revenue Performance")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist['year_month'], y=df_hist['revenue'], mode='lines', name='Historical',
                             line=dict(color='#6366f1', width=2), fill='tozeroy', fillcolor='rgba(99,102,241,0.08)'))
    fig.add_trace(go.Scatter(x=df_fc['year_month'], y=df_fc['revenue'], mode='lines+markers', name='Forecast',
                             line=dict(color='#8b5cf6', width=2, dash='dash'), marker=dict(size=5)))
    fig.update_layout(**chart_layout(), height=360, hovermode="x unified")
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    section("Model & Structure Summary")
    c1, c2 = st.columns(2, gap="medium")
    c1.markdown(info_card("Forecasting Model", "<strong>Model:</strong> Gradient Boosting (Production)<br><strong>Validation MAPE:</strong> 8.67%<br><strong>Horizon:</strong> 12 months ahead"), unsafe_allow_html=True)
    c2.markdown(info_card("Liquidity Structure", f"<strong>Score:</strong> {summary['score']:.4f}<br><strong>Safe Threshold:</strong> ≥ {summary['safe_threshold']}<br><strong>Status:</strong> {summary['risk_band']}"), unsafe_allow_html=True)

# =============================================================================
# Forecast Page with Revenue Scenarios
# =============================================================================

def page_forecast():
    hero("Revenue Forecast", "12-Month Ahead Predictions with Machine Learning")
    
    try:
        df_hist, df_fc = get_revenue_history_and_forecast()
    except Exception as e:
        st.error(str(e)); return
    
    last = df_hist.sort_values('year_month').iloc[-1]
    
    # KPI Row
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.markdown(kpi("Model", "Gradient Boosting"), unsafe_allow_html=True)
    c2.markdown(kpi("Last Actual", last['year_month'].strftime('%B %Y')), unsafe_allow_html=True)
    c3.markdown(kpi("Last Revenue", fmt(last['revenue'])), unsafe_allow_html=True)
    c4.markdown(kpi("Avg Forecast", fmt(df_fc['revenue'].mean())), unsafe_allow_html=True)
    
    # =========================================================================
    # PHASE 8: Revenue Scenarios
    # =========================================================================
    section("Revenue Scenario Analysis")
    
    st.markdown(info_card("Scenario Planning", 
        "Select a scenario to see how demand shocks affect the forecast. "
        "Base uses the ML model output; Optimistic and Pessimistic apply ±10% adjustments to <em>forecast months only</em>."), 
        unsafe_allow_html=True)
    
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    
    scenario = st.radio(
        "Revenue Scenario", 
        ["Base", "Optimistic (+10%)", "Pessimistic (−10%)"],
        horizontal=True,
        key="scenario_selector"
    )
    st.caption("📊 Scenario adjusts forecast months only; historical data remains unchanged.")    # Apply scenario factor
    if scenario == "Optimistic (+10%)":
        factor = 1.10
    elif scenario == "Pessimistic (−10%)":
        factor = 0.90
    else:
        factor = 1.0
    
    df_scenario = df_fc.copy()
    df_scenario['scenario_revenue'] = df_fc['revenue'] * factor
    
    section("Historical vs Forecast")
    
    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scatter(
        x=df_hist['year_month'], y=df_hist['revenue'], 
        mode='lines', name='Historical',
        line=dict(color='#6366f1', width=2), 
        fill='tozeroy', fillcolor='rgba(99,102,241,0.06)'
    ))
    # Base Forecast
    fig.add_trace(go.Scatter(
        x=df_fc['year_month'], y=df_fc['revenue'], 
        mode='lines+markers', name='Base Forecast',
        line=dict(color='#8b5cf6', width=2.5, dash='dash'), 
        marker=dict(size=7, symbol='diamond')
    ))
    # Scenario Forecast (if not Base)
    if factor != 1.0:
        scenario_name = "Optimistic" if factor > 1 else "Pessimistic"
        fig.add_trace(go.Scatter(
            x=df_scenario['year_month'], y=df_scenario['scenario_revenue'], 
            mode='lines+markers', name=f'{scenario_name} Forecast',
            line=dict(color='#10b981' if factor > 1 else '#ef4444', width=2.5), 
            marker=dict(size=6)
        ))
    
    fig.update_layout(**chart_layout(), height=360, hovermode="x unified")
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    section("Forecast Details")
    
    tbl = df_fc.copy()
    tbl['Month'] = tbl['year_month'].dt.strftime('%B %Y')
    tbl['Base Revenue'] = tbl['revenue'].apply(lambda x: f"₹{x:,.0f}")
    tbl['Scenario Revenue'] = (tbl['revenue'] * factor).apply(lambda x: f"₹{x:,.0f}")
    
    hist_idx = df_hist.set_index('year_month')
    yoy = []
    for _, r in df_fc.iterrows():
        prev = r['year_month'] - pd.DateOffset(years=1)
        if prev in hist_idx.index:
            chg = ((r['revenue'] / hist_idx.loc[prev, 'revenue']) - 1) * 100
            yoy.append(f"{chg:+.1f}%")
        else:
            yoy.append("—")
    tbl['YoY (Base)'] = yoy
    
    if factor != 1.0:
        st.dataframe(tbl[['Month', 'Base Revenue', 'Scenario Revenue', 'YoY (Base)']], use_container_width=True, hide_index=True)
    else:
        st.dataframe(tbl[['Month', 'Base Revenue', 'YoY (Base)']], use_container_width=True, hide_index=True)
    
    csv = get_forecast_csv_bytes()
    if csv:
        download_btn("Download Base Forecast CSV", csv, "forecast.csv", key="dl_forecast")

# =============================================================================
# Liquidity Page with What-If Planner & Cash Buffer Simulation
# =============================================================================

def page_liquidity():
    hero("Liquidity Risk Analysis", "Structural Assessment – What-If Planning – Cash Simulation")
    
    try:
        df_risk = get_liquidity_risk_table()
        summary = get_liquidity_config_summary()
    except Exception as e:
        st.error(str(e)); return
    
    # =========================================================================
    # KPI Cards (Current Structural Values)
    # =========================================================================
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.markdown(kpi("Cash Conversion Cycle", f"{summary['ccc_days']:.0f} days", "DSO + DIO − DPO"), unsafe_allow_html=True)
    c2.markdown(kpi("Operating Cash Margin", f"{summary['operating_cash_margin']*100:.1f}%", "Gross Margin − Fixed Costs"), unsafe_allow_html=True)
    c3.markdown(kpi("Structural Liq. Score", f"{summary['score']:.4f}", f"Safe ≥ {summary['safe_threshold']}"), unsafe_allow_html=True)
    c4.markdown(kpi_badge("Risk Level", summary['risk_band'], badge_type(summary['risk_band'])), unsafe_allow_html=True)
    
    st.caption("💡 **Operating Cash Margin**: % of revenue left after fixed costs. **Liquidity Score**: Margin minus CCC penalty.")
    
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown(info_card("Score Interpretation", 
        f"The structural score measures working capital efficiency; it is constant across months in this rule-based model. "
        f"Current: <strong>{summary['score']:.4f}</strong> (Safe ≥ {summary['safe_threshold']}, At Risk ≥ {summary['at_risk_threshold']}, Critical < {summary['at_risk_threshold']}). "
        f"Status: <strong>{summary['risk_band']}</strong>."), 
        unsafe_allow_html=True)
    
    # =========================================================================
    # PHASE 8: What-If Liquidity Planner
    # =========================================================================
    section("What-If Liquidity Planner")
    
    st.markdown(info_card("Planning Tool", 
        "Structural score is based only on margin, fixed costs, and CCC. "
        "Use the sliders to test whether planned improvements would move AUM into the Safe zone."), 
        unsafe_allow_html=True)
    
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    
    # Sliders for what-if parameters - 5 sliders in 2 rows
    slider_row1 = st.columns(3, gap="medium")
    
    with slider_row1[0]:
        target_dso = st.slider(
            "DSO (days)", 
            min_value=20, max_value=90, 
            value=int(summary['dso_days']),
            key="whatif_dso"
        )
    
    with slider_row1[1]:
        target_dio = st.slider(
            "DIO (days)", 
            min_value=20, max_value=120, 
            value=int(summary['dio_days']),
            key="whatif_dio"
        )
    
    with slider_row1[2]:
        target_dpo = st.slider(
            "DPO (days)", 
            min_value=0, max_value=120, 
            value=int(summary['dpo_days']),
            key="whatif_dpo"
        )
    
    slider_row2 = st.columns(2, gap="medium")
    
    with slider_row2[0]:
        target_gm = st.slider(
            "Gross Margin (%)", 
            min_value=10, max_value=60, 
            value=int(summary['gross_margin_pct'] * 100),
            key="whatif_gm"
        )
    
    with slider_row2[1]:
        target_fc = st.slider(
            "Fixed Cost Ratio (%)", 
            min_value=5, max_value=40, 
            value=int(summary['fixed_cost_ratio'] * 100),
            key="whatif_fc"
        )
    
    # Compute new CCC and score
    wc_penalty = summary['wc_penalty']
    target_ccc = target_dso + target_dio - target_dpo
    gm_new = target_gm / 100.0
    fc_new = target_fc / 100.0
    op_margin_new = gm_new - fc_new
    score_new = op_margin_new - (target_ccc / 365.0) * wc_penalty
    
    # Determine new risk band using existing thresholds from config
    safe_threshold = summary['safe_threshold']
    at_risk_threshold = summary['at_risk_threshold']
    if score_new >= safe_threshold:
        risk_band_new = "Safe"
    elif score_new <= at_risk_threshold:
        risk_band_new = "Critical"
    else:
        risk_band_new = "At Risk"
    
    # Display computed metrics
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    
    comp_cols = st.columns(4, gap="medium")
    
    delta_score = score_new - summary['score']
    delta_display = f"{delta_score:+.4f}"
    
    comp_cols[0].markdown(kpi("CCC", f"{target_ccc} days", "DSO + DIO − DPO"), unsafe_allow_html=True)
    comp_cols[1].markdown(kpi("Op. Cash Margin", f"{op_margin_new*100:.1f}%"), unsafe_allow_html=True)
    comp_cols[2].markdown(kpi("Adj. Liq. Score", f"{score_new:.4f}", f"Δ {delta_display}"), unsafe_allow_html=True)
    comp_cols[3].markdown(kpi_badge("Risk Band", risk_band_new, badge_type(risk_band_new)), unsafe_allow_html=True)
    
    # Explanatory text
    st.markdown(f"""
    <div class="info-card" style="margin-top: 12px;">
        <div class="info-card-title">Projected Outcome</div>
        <p class="info-card-text">
            With DSO = <strong>{target_dso}d</strong>, DIO = <strong>{target_dio}d</strong>, DPO = <strong>{target_dpo}d</strong> 
            (CCC = <strong>{target_ccc}d</strong>), Gross Margin = <strong>{target_gm}%</strong>, Fixed Cost Ratio = <strong>{target_fc}%</strong>, 
            the structural liquidity score would be <strong>{score_new:.4f}</strong>, which falls in the <strong>{risk_band_new}</strong> band.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # PHASE 8: Cash Buffer & Covenant Simulation
    # =========================================================================
    section("Cash Buffer & Covenant Simulation")
    
    st.markdown(info_card("Simulation", 
        "Simulate monthly cash balance trajectory based on operating cash flows. Compare against a minimum cash covenant to identify potential breaches."), 
        unsafe_allow_html=True)
    
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    
    # Inputs for simulation - 3 columns
    input_cols = st.columns(3, gap="medium")
    
    with input_cols[0]:
        opening_cash_cr = st.number_input(
            "Starting Cash Balance (₹ Cr)", 
            min_value=0.0, max_value=500.0, value=50.0, step=5.0,
            key="opening_cash"
        )
    
    with input_cols[1]:
        covenant_cr = st.number_input(
            "Minimum Cash Covenant (₹ Cr)", 
            min_value=0.0, max_value=100.0, value=25.0, step=2.5,
            key="covenant"
        )
    
    with input_cols[2]:
        cash_scenario = st.selectbox(
            "Cash Flow Scenario",
            ["Base", "Optimistic (+10%)", "Pessimistic (−10%)"],
            key="cash_scenario"
        )
    
    # Scenario multiplier
    if cash_scenario == "Optimistic (+10%)":
        cash_factor = 1.10
    elif cash_scenario == "Pessimistic (−10%)":
        cash_factor = 0.90
    else:
        cash_factor = 1.0
    
    st.caption("💰 **Cash Covenant**: Minimum cash balance required to avoid a breach.")
    
    # Compute cash balance trajectory
    df_sim = df_risk.copy()
    df_sim = df_sim.sort_values('year_month').reset_index(drop=True)
    
    # Convert operating cash flow to Crores (assuming values are in INR) and apply scenario
    df_sim['cash_flow_cr'] = (df_sim['operating_cash_flow'] / 1e7) * cash_factor
    
    # Cumulative cash balance
    cash_balance = []
    current_balance = opening_cash_cr
    for cf in df_sim['cash_flow_cr']:
        current_balance += cf
        cash_balance.append(current_balance)
    
    df_sim['cash_balance_cr'] = cash_balance
    df_sim['below_covenant'] = df_sim['cash_balance_cr'] < covenant_cr
    
    # Plot cash balance vs covenant
    fig_cash = go.Figure()
    
    # Split by historical vs forecast for colouring
    df_hist_sim = df_sim[~df_sim['is_forecast']]
    df_fc_sim = df_sim[df_sim['is_forecast']]
    
    # Historical cash balance line
    if len(df_hist_sim) > 0:
        fig_cash.add_trace(go.Scatter(
            x=df_hist_sim['year_month'], y=df_hist_sim['cash_balance_cr'],
            mode='lines+markers', name='Historical',
            line=dict(color='#6366f1', width=2),
            marker=dict(size=5)
        ))
    
    # Forecast cash balance line
    if len(df_fc_sim) > 0:
        fig_cash.add_trace(go.Scatter(
            x=df_fc_sim['year_month'], y=df_fc_sim['cash_balance_cr'],
            mode='lines+markers', name='Forecast',
            line=dict(color='#8b5cf6', width=2, dash='dash'),
            marker=dict(size=5)
        ))
    
    # Covenant line
    fig_cash.add_hline(
        y=covenant_cr, 
        line_dash="dash", 
        line_color="#ef4444", 
        annotation_text="Covenant", 
        annotation_position="right"
    )
    
    # Highlight breaches
    breaches = df_sim[df_sim['below_covenant']]
    if len(breaches) > 0:
        fig_cash.add_trace(go.Scatter(
            x=breaches['year_month'], y=breaches['cash_balance_cr'],
            mode='markers', name='Breach',
            marker=dict(size=12, color='#ef4444', symbol='x')
        ))
    
    fig_cash.update_layout(
        **chart_layout(), 
        height=320, 
        hovermode="x unified",
        yaxis_title="Cash Balance (₹ Cr)"
    )
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig_cash, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Breach summary
    breach_count = df_sim['below_covenant'].sum()
    total_months = len(df_sim)
    
    if breach_count > 0:
        first_breach = df_sim[df_sim['below_covenant']].iloc[0]['year_month'].strftime('%B %Y')
        st.error(f"⚠️ Covenant breached in **{breach_count}** of {total_months} month(s). First breach: **{first_breach}**.")
    else:
        st.success(f"✅ Cash balance never falls below the covenant level across all {total_months} months.")
    
    # =========================================================================
    # Cash Flow Timeline (existing)
    # =========================================================================
    section("Cash Flow Timeline")
    
    fig = go.Figure()
    hist = df_risk[~df_risk['is_forecast']]
    fc = df_risk[df_risk['is_forecast']]
    fig.add_trace(go.Bar(x=hist['year_month'], y=hist['operating_cash_flow'], name='Historical', marker_color='#6366f1'))
    fig.add_trace(go.Bar(x=fc['year_month'], y=fc['operating_cash_flow'], name='Forecast', marker_color='#8b5cf6'))
    fig.update_layout(**chart_layout(), height=320, barmode='relative')
    
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================================================================
    # Classification Table (existing)
    # =========================================================================
    section("Classification Table")
    
    col_main, col_filter = st.columns([3, 1], gap="medium")
    
    with col_filter:
        fc_only = st.checkbox("Forecast only", value=False, key="liq_fc_only")
        yrs = sorted(df_risk['year_month'].dt.year.unique().tolist())
        yr = st.selectbox("Year", ['All'] + yrs, key="liq_year")
    
    view = df_risk.copy()
    if yr != 'All':
        view = view[view['year_month'].dt.year == int(yr)]
    if fc_only:
        view = view[view['is_forecast']]
    
    with col_main:
        tbl = view[['year_month', 'revenue', 'operating_cash_flow', 'adjusted_liquidity_score', 'liquidity_risk_label', 'is_forecast']].copy()
        tbl['Month'] = tbl['year_month'].dt.strftime('%B %Y')
        tbl['Revenue'] = tbl['revenue'].apply(lambda x: f"₹{x:,.0f}")
        tbl['Cash Flow'] = tbl['operating_cash_flow'].apply(lambda x: f"₹{x:,.0f}")
        tbl['Score'] = tbl['adjusted_liquidity_score'].apply(lambda x: f"{x:.4f}")
        tbl['Risk'] = tbl['liquidity_risk_label']
        tbl['Type'] = tbl['is_forecast'].apply(lambda x: 'Forecast' if x else 'Historical')
        st.dataframe(tbl[['Month', 'Revenue', 'Cash Flow', 'Score', 'Risk', 'Type']], use_container_width=True, hide_index=True, height=300)
    
    csv = get_liquidity_csv_bytes()
    if csv:
        download_btn("Download Risk Table CSV", csv, "liquidity_risk.csv", key="dl_liquidity")
    
    # =========================================================================
    # Insights & Recommendations
    # =========================================================================
    section("Insights & Recommendations")
    
    total = len(df_risk)
    hist_n = (~df_risk['is_forecast']).sum()
    fc_n = df_risk['is_forecast'].sum()
    
    c1, c2 = st.columns(2, gap="medium")
    c1.markdown(insight("Analysis Summary", [
        f"Total months analysed: {total}",
        f"Historical: {hist_n} | Forecast: {fc_n}",
        f"Structural Score: {summary['score']:.4f}",
        f"Risk Band: {summary['risk_band']}"
    ]), unsafe_allow_html=True)
    c2.markdown(insight("Recommendations", [
        f"CCC of {summary['ccc_days']:.0f}d exceeds optimal; aim for ≤37d",
        "Reduce DSO by 10–15 days to speed collections",
        "Alternative: Increase gross margin to ~35%",
        "Use What-If Planner to test improvements"
    ]), unsafe_allow_html=True)

# =============================================================================
# Router
# =============================================================================

if page == "Overview":
    page_overview()
elif page == "Revenue Forecast":
    page_forecast()
elif page == "Liquidity Risk":
    page_liquidity()

st.markdown(f'<div class="footer">Aurora Utensils Manufacturing · Executive Dashboard · Phase 8 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
