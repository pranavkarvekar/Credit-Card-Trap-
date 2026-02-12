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

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from rules_engine import run_all_checks
from ml_engine import load_model, predict_risk, get_feature_importance
from explanation_engine import (
    generate_trap_summary,
    generate_risk_explanation,
    generate_actionable_advice,
    compute_financial_metrics,
)


# ── Page Configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Card Trap Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ─── Global Dark Theme ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #141937 40%, #1a1f3d 100%);
    font-family: 'Inter', sans-serif;
}

/* ─── Header ────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 50%);
    animation: shimmer 6s infinite;
}
@keyframes shimmer {
    0%, 100% { transform: rotate(0deg); }
    50% { transform: rotate(180deg); }
}
.main-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(to right, #ffffff, #e0e7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    position: relative;
    z-index: 1;
}
.main-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1.1rem;
    margin-top: 0.5rem;
    font-weight: 300;
    position: relative;
    z-index: 1;
}

/* ─── Metric Cards ──────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, rgba(30, 35, 70, 0.9), rgba(20, 25, 55, 0.95));
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    border-color: rgba(102, 126, 234, 0.5);
    box-shadow: 0 6px 24px rgba(102, 126, 234, 0.15);
    transform: translateY(-2px);
}
.metric-card .metric-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0.3rem 0;
}
.metric-card .metric-desc {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.4);
}

/* ─── Trap Cards ────────────────────────────────────────── */
.trap-card {
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.trap-card:hover {
    transform: translateY(-2px);
}
.trap-safe {
    background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.trap-medium {
    background: linear-gradient(145deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05));
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.trap-high {
    background: linear-gradient(145deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.trap-card h4 {
    margin: 0 0 0.5rem 0;
    font-size: 1.15rem;
    font-weight: 600;
}
.trap-card p {
    margin: 0.3rem 0;
    font-size: 0.93rem;
    line-height: 1.5;
    color: rgba(255,255,255,0.8);
}

/* ─── Status Badges ─────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-safe { background: rgba(16, 185, 129, 0.2); color: #10b981; }
.badge-medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.badge-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

/* ─── Advice Cards ──────────────────────────────────────── */
.advice-card {
    background: linear-gradient(145deg, rgba(30, 35, 70, 0.9), rgba(20, 25, 55, 0.95));
    border-left: 4px solid #667eea;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.advice-card h5 {
    margin: 0 0 0.5rem 0;
    color: #e0e7ff;
    font-size: 1.05rem;
}
.advice-card p {
    margin: 0.2rem 0;
    color: rgba(255,255,255,0.75);
    font-size: 0.92rem;
    line-height: 1.5;
}

/* ─── Summary Banner ────────────────────────────────────── */
.summary-banner {
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    font-size: 1.15rem;
    font-weight: 600;
}
.banner-critical {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.15));
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
}
.banner-danger {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.1));
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #fca5a5;
}
.banner-warning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.1));
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fde68a;
}
.banner-caution {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(217, 119, 6, 0.05));
    border: 1px solid rgba(245, 158, 11, 0.2);
    color: #fde68a;
}
.banner-safe {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.1));
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
}

/* ─── Sidebar ───────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1330 0%, #171d42 100%) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #667eea !important;
}

/* ─── Tabs ──────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(20, 25, 55, 0.5);
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 10px 20px;
    color: rgba(255,255,255,0.6);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #ffffff !important;
}

/* ─── Flow Diagram ──────────────────────────────────────── */
.flow-diagram {
    background: linear-gradient(145deg, rgba(30, 35, 70, 0.8), rgba(20, 25, 55, 0.9));
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    font-size: 0.95rem;
    color: rgba(255,255,255,0.8);
    line-height: 2;
}
.flow-step {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    margin: 0.3rem;
}
.flow-arrow {
    color: #667eea;
    font-size: 1.5rem;
    margin: 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🛡️ Credit Card Trap Detector</h1>
    <p>AI-Powered Financial Awareness & Decision Support System</p>
</div>
""", unsafe_allow_html=True)


# ── Load ML Model ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_model():
    """Load the ML model (cached)."""
    return load_model()


model = get_model()


# ── Sidebar: User Input Module ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📝 Your Financial Profile")
    st.markdown("---")

    st.markdown("### 💰 Income & Score")
    monthly_income = st.number_input(
        "Monthly Income (₹)", min_value=0, max_value=10000000,
        value=50000, step=5000, help="Your total monthly income"
    )
    credit_score = st.slider(
        "Credit Score", min_value=300, max_value=900,
        value=700, step=1, help="Your current credit score (CIBIL/Experian)"
    )

    st.markdown("### 💳 Credit Details")
    credit_limit = st.number_input(
        "Total Credit Limit (₹)", min_value=0, max_value=50000000,
        value=200000, step=10000, help="Total credit limit across all cards"
    )
    outstanding_balance = st.number_input(
        "Outstanding Balance (₹)", min_value=0, max_value=50000000,
        value=50000, step=5000, help="Current outstanding balance"
    )
    number_of_cards = st.number_input(
        "Number of Credit Cards", min_value=0, max_value=20,
        value=2, step=1
    )

    st.markdown("### 📅 EMI & Payments")
    number_of_emis = st.number_input(
        "Number of Active EMIs", min_value=0, max_value=20,
        value=1, step=1
    )
    monthly_emi_amount = st.number_input(
        "Total Monthly EMI Amount (₹)", min_value=0, max_value=10000000,
        value=10000, step=1000
    )
    payment_behavior = st.selectbox(
        "Payment Behavior",
        options=["Full", "Partial", "Minimum", "Missed"],
        index=0,
        help="How do you typically pay your credit card bill?"
    )
    late_payments = st.number_input(
        "Number of Late Payments (Last Year)", min_value=0, max_value=24,
        value=0, step=1
    )

    st.markdown("### 🏧 Cash & Spending")
    cash_withdrawal_amount = st.number_input(
        "Cash Withdrawal on CC (₹)", min_value=0, max_value=10000000,
        value=0, step=1000, help="Cash advances from credit card"
    )
    total_spend_last_year = st.number_input(
        "Total Spend Last Year (₹)", min_value=0, max_value=100000000,
        value=300000, step=10000
    )
    avg_transaction_amount = st.number_input(
        "Avg Transaction Amount (₹)", min_value=0, max_value=10000000,
        value=2000, step=500
    )
    max_transaction_amount = st.number_input(
        "Max Transaction Amount (₹)", min_value=0, max_value=50000000,
        value=25000, step=1000
    )

    st.markdown("---")
    analyze_button = st.button("🔍 **Analyze My Credit**", use_container_width=True, type="primary")


# ── Main Content ───────────────────────────────────────────────────────────────

if analyze_button:
    # Build user data dict
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
        "total_spend_last_year": total_spend_last_year,
        "avg_transaction_amount": avg_transaction_amount,
        "max_transaction_amount": max_transaction_amount,
    }

    # ── Run all 3 engines ──
    with st.spinner("🔄 Running analysis across all 3 intelligence layers..."):
        # Layer 1: Rule-Based
        trap_results = run_all_checks(user_data)
        trap_summary = generate_trap_summary(trap_results)

        # Layer 2: ML Risk
        risk_result = predict_risk(model, user_data)
        feat_importance = get_feature_importance(model)

        # Layer 3: Explanation
        risk_explanation = generate_risk_explanation(risk_result, feat_importance)
        advice_list = generate_actionable_advice(user_data, trap_results, risk_result)
        financial_metrics = compute_financial_metrics(user_data)

    # ── Summary Banner ──
    banner_class = f"banner-{trap_summary['overall_status']}"
    st.markdown(f"""
    <div class="summary-banner {banner_class}">
        {trap_summary['status_text']} &nbsp;|&nbsp;
        ML Risk: <strong>{risk_result['risk_label']}</strong> ({risk_result['risk_score']}/100) &nbsp;|&nbsp;
        Traps Found: <strong>{trap_summary['detected_count']}/{trap_summary['total_traps']}</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── 3 Tabs ──
    tab1, tab2, tab3 = st.tabs([
        "🔍 Trap Detection",
        "📊 Risk Assessment",
        "💡 Insights & Advice",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: Trap Detection (Rule-Based Engine)
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 🔍 Rule-Based Trap Detection")
        st.markdown("*Our rule engine scans for 6 common credit card traps based on your financial data.*")
        st.markdown("")

        for trap in trap_results:
            if trap.detected:
                css_class = "trap-high" if trap.severity == "high" else "trap-medium"
                badge_class = "badge-high" if trap.severity == "high" else "badge-medium"
                badge_text = "🚨 DANGER" if trap.severity == "high" else "⚠️ WARNING"
            else:
                css_class = "trap-safe"
                badge_class = "badge-safe"
                badge_text = "✅ SAFE"

            with st.container():
                st.markdown(f"""
                <div class="trap-card {css_class}">
                    <h4>{trap.icon} {trap.trap_name} &nbsp;
                        <span class="badge {badge_class}">{badge_text}</span>
                    </h4>
                    <p>{trap.description}</p>
                </div>
                """, unsafe_allow_html=True)

                if trap.detected:
                    with st.expander(f"📖 Details: {trap.trap_name}", expanded=False):
                        st.markdown(f"**⚡ Consequences:**  \n{trap.consequences}")
                        st.markdown(f"**💡 Suggestion:**  \n{trap.suggestion}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: Risk Assessment (ML Engine)
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📊 ML-Powered Risk Assessment")
        st.markdown("*A trained Decision Tree model classifies your overall credit risk level.*")
        st.markdown("")

        col_gauge, col_probs = st.columns([1, 1])

        with col_gauge:
            # Risk Gauge
            risk_score = risk_result["risk_score"]
            if risk_score >= 70:
                gauge_color = "#ef4444"
            elif risk_score >= 40:
                gauge_color = "#f59e0b"
            else:
                gauge_color = "#10b981"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                title={
                    "text": f"Risk Score<br><span style='font-size:0.9em;color:{gauge_color}'>"
                            f"{risk_result['risk_label']}</span>",
                    "font": {"size": 18, "color": "white"},
                },
                number={"suffix": "/100", "font": {"size": 36, "color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.3)"},
                    "bar": {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "rgba(20, 25, 55, 0.8)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                        {"range": [35, 65], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [65, 100], "color": "rgba(239, 68, 68, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": risk_score,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=320,
                margin=dict(t=80, b=30, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_probs:
            # Probability Distribution
            probs = risk_result["probabilities"]
            prob_df = pd.DataFrame({
                "Risk Level": list(probs.keys()),
                "Probability": [v * 100 for v in probs.values()],
            })
            colors = {"Low Risk": "#10b981", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}
            fig_probs = px.bar(
                prob_df, x="Risk Level", y="Probability",
                color="Risk Level",
                color_discrete_map=colors,
                text=[f"{v:.1f}%" for v in prob_df["Probability"]],
            )
            fig_probs.update_traces(textposition="outside", textfont=dict(color="white", size=14))
            fig_probs.update_layout(
                title={"text": "Risk Probability Distribution", "font": {"color": "white", "size": 16}},
                height=320,
                margin=dict(t=60, b=30, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="rgba(255,255,255,0.6)"),
                yaxis=dict(color="rgba(255,255,255,0.6)", title="Probability (%)"),
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
            color_continuous_scale=["#667eea", "#764ba2", "#ef4444"],
        )
        fig_feat.update_traces(textposition="outside", textfont=dict(color="white", size=12))
        fig_feat.update_layout(
            title={"text": "Feature Importance in Risk Prediction", "font": {"color": "white", "size": 16}},
            height=max(280, len(importance_df) * 45),
            margin=dict(t=60, b=30, l=10, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="rgba(255,255,255,0.6)", title="Importance (%)"),
            yaxis=dict(color="rgba(255,255,255,0.6)", title=""),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        # Risk Context
        st.markdown(f"> {risk_explanation['risk_context']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: Insights & Advice (Explanation Layer)
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 💡 Personalized Insights & Advice")
        st.markdown("*Actionable recommendations to improve your financial health.*")
        st.markdown("")

        # Risk Narrative
        st.markdown("#### 📝 Risk Analysis")
        st.markdown(risk_explanation["risk_narrative"])
        st.markdown("")

        # Top Risk Factors
        if risk_explanation["top_factors"]:
            st.markdown("#### 🎯 Top Risk Factors")
            for factor in risk_explanation["top_factors"]:
                st.markdown(
                    f"- **{factor['description']}** — contributes {factor['importance']}% to the risk model"
                )
            st.markdown("")

        # Financial Health Metrics
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

        # Actionable Advice
        st.markdown("#### 🗺️ Action Plan")
        for adv in advice_list:
            st.markdown(f"""
            <div class="advice-card">
                <h5>{adv['priority']} — {adv['title']}</h5>
                <p><strong>Category:</strong> {adv['category']}</p>
                <p>{adv['detail']}</p>
                <p style="font-size:0.82rem; color: rgba(255,255,255,0.5);">
                    Impact: {adv['impact']}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # System Architecture Flow
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
            <span class="flow-arrow">&nbsp;</span>
            <span class="flow-step">🤖 ML Engine</span>
            <span class="flow-arrow">&nbsp;</span>
            <span class="flow-step">🧮 Calculators</span>
            <br>
            <span class="flow-arrow">↓</span>
            <br>
            <span class="flow-step">🔍 Trap Detection</span>
            <span class="flow-arrow">&nbsp;</span>
            <span class="flow-step">📊 Risk Prediction</span>
            <br>
            <span class="flow-arrow">↓</span>
            <br>
            <span class="flow-step">💡 Explanation Engine</span>
            <br>
            <span class="flow-arrow">↓</span>
            <br>
            <span class="flow-step">📈 Visualization + Advice</span>
            <br>
            <span class="flow-arrow">↓</span>
            <br>
            <span class="flow-step">👤 User</span>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Welcome State ──
    st.markdown("")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align:center; padding:2rem;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">🔍</div>
            <div class="metric-value" style="font-size:1.2rem;">Rule-Based Engine</div>
            <div class="metric-desc" style="margin-top:0.5rem;">
                Detects 6 specific credit card traps including minimum payment, credit overuse,
                cash withdrawal, EMI overload, late payments, and unsafe practices.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align:center; padding:2rem;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">🤖</div>
            <div class="metric-value" style="font-size:1.2rem;">ML Risk Model</div>
            <div class="metric-desc" style="margin-top:0.5rem;">
                A trained Decision Tree classifier predicts your overall credit risk level
                (Low / Medium / High) with probability distribution.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align:center; padding:2rem;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">💡</div>
            <div class="metric-value" style="font-size:1.2rem;">Explanation Layer</div>
            <div class="metric-desc" style="margin-top:0.5rem;">
                Makes results understandable with personalized explanations,
                financial health metrics, and actionable correction suggestions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
    <div style="text-align:center; color: rgba(255,255,255,0.5); padding: 2rem;">
        <p style="font-size:1.1rem;">👈 Fill in your financial details in the sidebar and click
        <strong>Analyze My Credit</strong> to get started.</p>
    </div>
    """, unsafe_allow_html=True)

    # Problems we solve
    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="trap-card trap-high">
            <h4>❌ Common Mistakes Users Make</h4>
            <p>• Pay only minimum due</p>
            <p>• Overuse credit limit</p>
            <p>• Withdraw cash from credit card</p>
            <p>• Take unnecessary EMIs</p>
            <p>• Miss due dates</p>
            <p>• Use unsafe credit practices</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="trap-card trap-safe">
            <h4>✔️ How This System Helps</h4>
            <p>• Identifies your specific mistakes</p>
            <p>• Explains consequences clearly</p>
            <p>• Suggests personalized corrections</p>
            <p>• Predicts your overall risk level</p>
            <p>• Tracks key financial health metrics</p>
            <p>• Provides an actionable improvement plan</p>
        </div>
        """, unsafe_allow_html=True)
