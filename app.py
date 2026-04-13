"""
Credit Card Trap - Financial Awareness & Decision-Support System
================================================================
Streamlit-based UI with 3 intelligence layers:
  1. Rule-Based Engine - detects specific traps
  2. ML Risk Model - classifies overall danger
  3. Explanation Layer - makes results understandable
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from rules_engine import run_all_checks
from ml_engine import load_model, predict_risk, get_feature_importance
from explanation_engine import (
    generate_trap_summary,
    generate_risk_explanation,
    generate_actionable_advice,
    compute_financial_metrics,
)

# ── Design System ─────────────────────────────────────────────────────────────
BG       = "#0A0E27"
SURFACE  = "#141824"
SURFACE2 = "#1A1F37"
BORDER   = "#1E2540"
PRIMARY  = "#EAB308"   # yellow-400
PRIMARY2 = "#D97706"   # amber-600
ACCENT   = "#10B981"   # emerald
WARNING  = "#F59E0B"
DANGER   = "#EF4444"
TEXT     = "#F1F5F9"
MUTED    = "#64748B"
FAINT    = "#1E2540"

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinAdvisor — Credit Card Trap Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ─── Global ─────────────────────────────────────────── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{
    background: {BG} !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    color: {TEXT} !important;
}}
.block-container {{
    padding: 0 2rem 2rem !important;
    max-width: 100% !important;
}}

/* ─── Remove Streamlit chrome ───────────────────────── */
#MainMenu, footer, header,
[data-testid="stToolbar"] {{
    display: none !important;
}}
[data-testid="stHeader"] {{
    background: transparent !important;
    border: none !important;
}}

/* ─── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
    min-width: 360px !important;
    width: 360px !important;
}}
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] > div:first-child {{
    background: {SURFACE} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
[data-testid="stSidebar"] .stCaption {{ color: {MUTED} !important; }}
/* Sidebar collapse/expand button */
[data-testid="stSidebar"] button[kind="header"],
[data-testid="collapsedControl"] {{
    color: {TEXT} !important;
    background: {SURFACE} !important;
}}

/* ─── Inputs ─────────────────────────────────────────── */
input[type="number"], .stSelectbox > div > div,
[data-baseweb="select"] > div {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-family: 'DM Sans', system-ui !important;
}}
input[type="number"]:focus {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 3px {PRIMARY}33 !important;
}}
[data-baseweb="select"] > div:hover {{ border-color: {PRIMARY}88 !important; }}
[data-testid="stNumberInput"] label p,
[data-testid="stSelectbox"] label p,
[data-testid="stCheckbox"] label p {{
    color: {TEXT} !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}}

/* ─── Tabs ───────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {SURFACE} !important;
    border-bottom: 1px solid {BORDER} !important;
    padding: 0 8px !important;
    gap: 4px !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent !important;
    color: {MUTED} !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 12px 22px !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: color 0.15s !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {PRIMARY} !important;
    border-bottom: 2px solid {PRIMARY} !important;
    background: transparent !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{
    background: {BG} !important;
    padding-top: 24px !important;
}}

/* ─── Button ─────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY2} 100%) !important;
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    letter-spacing: 0.025em !important;
    box-shadow: 0 4px 20px {PRIMARY}40 !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px {PRIMARY}60 !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; }}

/* ─── Metrics ────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 18px 22px !important;
}}
[data-testid="stMetricLabel"] p {{
    color: {MUTED} !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
}}

/* ─── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {PRIMARY}; }}

/* ─── Header ─────────────────────────────────────────── */
.fin-header {{
    background: linear-gradient(135deg, {SURFACE} 0%, {SURFACE2} 100%);
    border: 1px solid {PRIMARY}30;
    padding: 2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px {PRIMARY}15, 0 0 0 1px {PRIMARY}10;
    position: relative;
    overflow: hidden;
}}
.fin-header::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, {PRIMARY}15 0%, transparent 70%);
    border-radius: 50%;
}}
.fin-header::after {{
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 140px; height: 140px;
    background: radial-gradient(circle, {PRIMARY}10 0%, transparent 70%);
    border-radius: 50%;
}}
.fin-header h1 {{
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 800;
    color: {TEXT};
    margin: 0;
    position: relative;
    z-index: 1;
}}
.fin-header h1 span {{
    background: linear-gradient(90deg, {PRIMARY}, {PRIMARY2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.fin-header p {{
    color: {MUTED};
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
}}

/* ─── Logo pill ──────────────────────────────────────── */
.logo-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: {PRIMARY}15;
    border: 1px solid {PRIMARY}30;
    border-radius: 50px;
    padding: 6px 16px;
    margin-bottom: 1rem;
    font-size: 0.8rem;
    font-weight: 700;
    color: {PRIMARY};
    letter-spacing: 0.05em;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
}}

/* ─── Summary Banner ─────────────────────────────────── */
.summary-banner {{
    border-radius: 16px;
    padding: 1.2rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    font-size: 1rem;
    font-weight: 600;
    border: 1px solid;
}}
.banner-critical {{
    background: {DANGER}18;
    border-color: {DANGER}40;
    color: #fca5a5;
}}
.banner-danger {{
    background: {DANGER}12;
    border-color: {DANGER}30;
    color: #fca5a5;
}}
.banner-warning {{
    background: {WARNING}15;
    border-color: {WARNING}35;
    color: #fde68a;
}}
.banner-caution {{
    background: {WARNING}10;
    border-color: {WARNING}25;
    color: #fde68a;
}}
.banner-safe {{
    background: {ACCENT}12;
    border-color: {ACCENT}30;
    color: #6ee7b7;
}}

/* ─── Trap Cards ─────────────────────────────────────── */
.trap-card {{
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 0.9rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: 1px solid;
}}
.trap-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.3);
}}
.trap-safe {{
    background: {ACCENT}0D;
    border-color: {ACCENT}30;
}}
.trap-medium {{
    background: {WARNING}12;
    border-color: {WARNING}40;
}}
.trap-high {{
    background: {DANGER}12;
    border-color: {DANGER}40;
}}
.trap-card h4 {{
    margin: 0 0 0.5rem 0;
    font-size: 1.05rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
}}
.trap-card p {{
    margin: 0.3rem 0;
    font-size: 0.9rem;
    line-height: 1.55;
    color: {MUTED};
}}

/* ─── Badges ─────────────────────────────────────────── */
.badge {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    border: 1px solid;
}}
.badge-safe   {{ background: {ACCENT}15; color: {ACCENT}; border-color: {ACCENT}30; }}
.badge-medium {{ background: {WARNING}15; color: {WARNING}; border-color: {WARNING}30; }}
.badge-high   {{ background: {DANGER}15; color: {DANGER}; border-color: {DANGER}30; }}

/* ─── Metric Card ────────────────────────────────────── */
.metric-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s, transform 0.2s;
}}
.metric-card:hover {{
    border-color: {PRIMARY}40;
    transform: translateY(-2px);
}}
.metric-card .metric-label {{
    font-size: 0.78rem;
    color: {MUTED};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.metric-card .metric-value {{
    font-size: 1.5rem;
    font-weight: 800;
    color: {TEXT};
    margin: 0.3rem 0;
    font-family: 'Syne', sans-serif;
}}
.metric-card .metric-desc {{
    font-size: 0.78rem;
    color: {MUTED};
}}

/* ─── Advice Card ────────────────────────────────────── */
.advice-card {{
    background: {SURFACE};
    border-left: 3px solid {PRIMARY};
    border-radius: 0 12px 12px 0;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.9rem;
    border-top: 1px solid {BORDER};
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
}}
.advice-card h5 {{
    margin: 0 0 0.5rem 0;
    color: {TEXT};
    font-size: 0.97rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
}}
.advice-card p {{
    margin: 0.2rem 0;
    color: {MUTED};
    font-size: 0.88rem;
    line-height: 1.5;
}}

/* ─── Flow Diagram ───────────────────────────────────── */
.flow-diagram {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    font-size: 0.9rem;
    color: {MUTED};
    line-height: 2.2;
}}
.flow-step {{
    display: inline-block;
    background: {PRIMARY}20;
    border: 1px solid {PRIMARY}40;
    color: {PRIMARY};
    padding: 0.4rem 1.1rem;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.82rem;
    margin: 0.3rem;
    font-family: 'Syne', sans-serif;
}}
.flow-arrow {{
    color: {PRIMARY}60;
    font-size: 1.3rem;
    margin: 0 0.3rem;
}}

/* ─── Welcome cards ──────────────────────────────────── */
.welcome-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}}
.welcome-card:hover {{
    border-color: {PRIMARY}40;
    transform: translateY(-3px);
    box-shadow: 0 8px 32px {PRIMARY}12;
}}
.welcome-icon {{
    font-size: 2.2rem;
    margin-bottom: 0.8rem;
}}
.welcome-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 0.6rem;
}}
.welcome-desc {{
    font-size: 0.85rem;
    color: {MUTED};
    line-height: 1.6;
}}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fin-header">
    <div class="logo-pill">⚡ FinAdvisor</div>
    <h1>💳 Credit Card <span>Trap Detector</span></h1>
    <p>AI-Powered Financial Awareness & Decision Support System</p>
</div>
""", unsafe_allow_html=True)


# ── Load ML Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

model = get_model()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📝 Credit Profile")
    st.markdown("---")

    st.markdown("### 💰 Income & Score")
    monthly_income = st.number_input(
        "Monthly Income (₹)", min_value=0, max_value=10000000,
        value=50000, step=5000,
    )
    credit_score = st.slider(
        "Credit Score (CIBIL)", min_value=300, max_value=900, value=700, step=1,
    )

    st.markdown("### 💳 Credit Details")
    credit_limit = st.number_input(
        "Total Credit Limit (₹)", min_value=0, max_value=50000000,
        value=200000, step=10000,
    )
    outstanding_balance = st.number_input(
        "Outstanding Balance (₹)", min_value=0, max_value=50000000,
        value=50000, step=5000,
    )
    number_of_cards = st.number_input(
        "Number of Credit Cards", min_value=0, max_value=20, value=2, step=1,
    )

    st.markdown("### 📅 EMI & Payments")
    number_of_emis = st.number_input(
        "Number of Active EMIs", min_value=0, max_value=20, value=1, step=1,
    )
    monthly_emi_amount = st.number_input(
        "Total Monthly EMI Amount (₹)", min_value=0, max_value=10000000,
        value=10000, step=1000,
    )
    payment_behavior = st.selectbox(
        "Payment Behavior",
        options=["Full", "Partial", "Minimum", "Missed"], index=0,
    )
    late_payments = st.number_input(
        "Late Payments (Last Year)", min_value=0, max_value=24, value=0, step=1,
    )

    st.markdown("### 🏧 Cash Advances")
    cash_withdrawal_amount = st.number_input(
        "Cash Advance Amount (₹)", min_value=0, max_value=10000000,
        value=0, step=1000,
    )

    st.markdown("---")
    analyze_button = st.button("🔍 Analyze My Credit", use_container_width=True, type="primary")


# ── Main Content ───────────────────────────────────────────────────────────────
if analyze_button:
    user_data = {
        "monthly_income": monthly_income,
        "credit_score": credit_score,
        "credit_limit": credit_limit,
        "outstanding_balance": outstanding_balance,
        "number_of_cards": number_of_cards,
        "number_of_emis": number_of_emis,
        "monthly_emi_amount": monthly_emi_amount,
        "payment_behavior": payment_behavior,
        "late_payments": late_payments,
        "cash_withdrawal_amount": cash_withdrawal_amount,
    }

    with st.spinner("Running analysis across all 3 intelligence layers..."):
        trap_results   = run_all_checks(user_data)
        trap_summary   = generate_trap_summary(trap_results)
        risk_result    = predict_risk(model, user_data)
        feat_importance = get_feature_importance(model)
        risk_explanation = generate_risk_explanation(risk_result, feat_importance)
        advice_list    = generate_actionable_advice(user_data, trap_results, risk_result)
        financial_metrics = compute_financial_metrics(user_data)

    # Summary Banner
    banner_class = f"banner-{trap_summary['overall_status']}"
    st.markdown(f"""
    <div class="summary-banner {banner_class}">
        {trap_summary['status_text']} &nbsp;·&nbsp;
        ML Risk: <strong>{risk_result['risk_label']}</strong> ({risk_result['risk_score']}/100) &nbsp;·&nbsp;
        Traps Found: <strong>{trap_summary['detected_count']}/{trap_summary['total_traps']}</strong>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🔍 Trap Detection",
        "📊 Risk Assessment",
        "💡 Insights & Advice",
    ])

    # ── TAB 1: Trap Detection ──────────────────────────────────────────────────
    with tab1:
        st.markdown("### 🔍 Rule-Based Trap Detection")
        st.markdown("*Our rule engine scans for 6 common credit card traps based on your financial data.*")
        st.markdown("")

        for trap in trap_results:
            if trap.detected:
                css = "trap-high" if trap.severity == "high" else "trap-medium"
                badge = "badge-high" if trap.severity == "high" else "badge-medium"
                badge_text = "🚨 DANGER" if trap.severity == "high" else "⚠️ WARNING"
            else:
                css, badge, badge_text = "trap-safe", "badge-safe", "✅ SAFE"

            st.markdown(f"""
            <div class="trap-card {css}">
                <h4>{trap.icon} {trap.trap_name} &nbsp;
                    <span class="badge {badge}">{badge_text}</span>
                </h4>
                <p>{trap.description}</p>
            </div>
            """, unsafe_allow_html=True)

            if trap.detected:
                with st.expander(f"📖 Details: {trap.trap_name}", expanded=False):
                    st.markdown(f"**⚡ Consequences:**  \n{trap.consequences}")
                    st.markdown(f"**💡 Suggestion:**  \n{trap.suggestion}")

    # ── TAB 2: Risk Assessment ─────────────────────────────────────────────────
    with tab2:
        st.markdown("### 📊 ML-Powered Risk Assessment")
        st.markdown("*A trained Decision Tree model classifies your overall credit risk level.*")
        st.markdown("")

        col_gauge, col_probs = st.columns([1, 1])

        with col_gauge:
            risk_score = risk_result["risk_score"]
            gauge_color = DANGER if risk_score >= 70 else (WARNING if risk_score >= 40 else ACCENT)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={
                    "text": f"Risk Score<br><span style='font-size:0.9em;color:{gauge_color}'>{risk_result['risk_label']}</span>",
                    "font": {"size": 16, "color": TEXT},
                },
                number={"suffix": "/100", "font": {"size": 34, "color": TEXT}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": SURFACE2,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35],  "color": "rgba(16, 185, 129, 0.09)"},
                        {"range": [35, 65], "color": "rgba(245, 158, 11, 0.09)"},
                        {"range": [65, 100],"color": "rgba(239, 68, 68, 0.09)"},
                    ],
                    "threshold": {
                        "line": {"color": TEXT, "width": 2},
                        "thickness": 0.8,
                        "value": risk_score,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=300,
                margin=dict(t=80, b=30, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": TEXT},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_probs:
            probs = risk_result["probabilities"]
            prob_df = pd.DataFrame({
                "Risk Level": list(probs.keys()),
                "Probability": [v * 100 for v in probs.values()],
            })
            colors = {"Low Risk": ACCENT, "Medium Risk": WARNING, "High Risk": DANGER}
            fig_probs = px.bar(
                prob_df, x="Risk Level", y="Probability",
                color="Risk Level", color_discrete_map=colors,
                text=[f"{v:.1f}%" for v in prob_df["Probability"]],
            )
            fig_probs.update_traces(textposition="outside", textfont=dict(color=TEXT, size=13))
            fig_probs.update_layout(
                title={"text": "Risk Probability Distribution", "font": {"color": TEXT, "size": 15}},
                height=300,
                margin=dict(t=60, b=30, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color=MUTED),
                yaxis=dict(color=MUTED, title="Probability (%)"),
                showlegend=False,
            )
            st.plotly_chart(fig_probs, use_container_width=True)

        # Feature Importance
        st.markdown("#### 🎯 What Drives Your Risk?")
        importance_df = pd.DataFrame(feat_importance, columns=["Feature", "Importance"])
        importance_df = importance_df[importance_df["Importance"] > 0.01]
        importance_df["Importance"] = importance_df["Importance"] * 100

        fig_feat = px.bar(
            importance_df.iloc[::-1], x="Importance", y="Feature",
            orientation="h",
            text=[f"{v:.1f}%" for v in importance_df.iloc[::-1]["Importance"]],
            color="Importance",
            color_continuous_scale=[SURFACE2, PRIMARY, PRIMARY2],
        )
        fig_feat.update_traces(textposition="outside", textfont=dict(color=TEXT, size=11))
        fig_feat.update_layout(
            title={"text": "Feature Importance in Risk Prediction", "font": {"color": TEXT, "size": 15}},
            height=max(280, len(importance_df) * 45),
            margin=dict(t=60, b=30, l=10, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color=MUTED, title="Importance (%)"),
            yaxis=dict(color=MUTED, title=""),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        st.markdown(f"> {risk_explanation['risk_context']}")

    # ── TAB 3: Insights & Advice ───────────────────────────────────────────────
    with tab3:
        st.markdown("### 💡 Personalized Insights & Advice")
        st.markdown("*Actionable recommendations to improve your financial health.*")
        st.markdown("")

        st.markdown("#### 📝 Risk Analysis")
        st.markdown(risk_explanation["risk_narrative"])
        st.markdown("")

        if risk_explanation["top_factors"]:
            st.markdown("#### 🎯 Top Risk Factors")
            for factor in risk_explanation["top_factors"]:
                st.markdown(
                    f"- **{factor['description']}** — contributes {factor['importance']}% to the risk model"
                )
            st.markdown("")

        st.markdown("#### 📊 Financial Health Metrics")
        metric_cols = st.columns(4)
        for i, metric in enumerate(financial_metrics):
            with metric_cols[i % 4]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric['name']}</div>
                    <div class="metric-value">{metric['status']} {metric['value']}</div>
                    <div class="metric-desc">{metric['description']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### 🗺️ Action Plan")
        for adv in advice_list:
            st.markdown(f"""
            <div class="advice-card">
                <h5>{adv['priority']} — {adv['title']}</h5>
                <p><strong>Category:</strong> {adv['category']}</p>
                <p>{adv['detail']}</p>
                <p style="font-size:0.8rem; color: rgba(255,255,255,0.35);">
                    Impact: {adv['impact']}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### ⚙️ How This System Works")
        st.markdown("""
        <div class="flow-diagram">
            <span class="flow-step">📝 User Input</span>
            <span class="flow-arrow">→</span>
            <span class="flow-step">✅ Validation</span>
            <span class="flow-arrow">→</span>
            <span class="flow-step">🔧 Feature Engineering</span>
            <br>
            <span class="flow-arrow">↓</span>
            <br>
            <span class="flow-step">📜 Rules Engine</span>
            &nbsp;&nbsp;
            <span class="flow-step">🤖 ML Engine</span>
            &nbsp;&nbsp;
            <span class="flow-step">🧮 Calculators</span>
            <br>
            <span class="flow-arrow">↓</span>
            <br>
            <span class="flow-step">🔍 Trap Detection</span>
            &nbsp;&nbsp;
            <span class="flow-step">📊 Risk Prediction</span>
            <br>
            <span class="flow-arrow">↓</span>
            <br>
            <span class="flow-step">💡 Explanation Engine</span>
            <br>
            <span class="flow-arrow">↓</span>
            <br>
            <span class="flow-step">📈 Visualization + Advice</span>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Welcome State ──────────────────────────────────────────────────────────
    st.markdown("")
    col1, col2, col3 = st.columns(3)

    for col, icon, title, desc in [
        (col1, "🔍", "Rule-Based Engine",
         "Detects 6 specific credit card traps — minimum payment, credit overuse, cash withdrawal, EMI overload, late payments, and unsafe practices."),
        (col2, "🤖", "ML Risk Model",
         "A trained Decision Tree classifier predicts your overall credit risk level (Low / Medium / High) with probability distribution."),
        (col3, "💡", "Explanation Layer",
         "Makes results understandable with personalized explanations, financial health metrics, and actionable correction suggestions."),
    ]:
        with col:
            st.markdown(f"""
            <div class="welcome-card">
                <div class="welcome-icon">{icon}</div>
                <div class="welcome-title">{title}</div>
                <div class="welcome-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        f"<div style='text-align:center; color:{MUTED}; padding:1.5rem; font-size:0.95rem;'>"
        f"👈 Fill in your credit details in the sidebar and click <strong style='color:{PRIMARY}'>Analyze My Credit</strong> to get started."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div class="trap-card trap-high">
            <h4>❌ Common Mistakes</h4>
            <p>• Pay only minimum due every month</p>
            <p>• Overuse credit limit (>30%)</p>
            <p>• Withdraw cash from credit card</p>
            <p>• Take unnecessary EMIs</p>
            <p>• Miss payment due dates</p>
            <p>• Ignore CIBIL score impact</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="trap-card trap-safe">
            <h4>✔️ How This System Helps</h4>
            <p>• Identifies your specific mistakes</p>
            <p>• Explains real ₹ cost of each trap</p>
            <p>• Suggests personalised corrections</p>
            <p>• Predicts your overall risk level</p>
            <p>• Tracks key financial health metrics</p>
            <p>• Provides an actionable improvement plan</p>
        </div>
        """, unsafe_allow_html=True)
