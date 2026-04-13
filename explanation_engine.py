"""
Credit Card Trap - Explanation & Advice Engine
=================================================
Makes rule-based and ML results understandable by generating
human-readable explanations, financial calculations, and advice.
"""


def generate_trap_summary(trap_results):
    """
    Generate a high-level summary of all trap detection results.

    Returns:
        dict with: total_traps, high_count, medium_count, safe_count,
                   overall_status, summary_text
    """
    total = len(trap_results)
    high = sum(1 for t in trap_results if t.detected and t.severity == "high")
    medium = sum(1 for t in trap_results if t.detected and t.severity == "medium")
    detected = sum(1 for t in trap_results if t.detected)
    safe = total - detected

    if high >= 2:
        overall = "critical"
        status_text = "🚨 CRITICAL: Multiple high-severity traps detected!"
    elif high >= 1:
        overall = "danger"
        status_text = "🔴 DANGER: High-severity credit trap detected!"
    elif medium >= 2:
        overall = "warning"
        status_text = "⚠️ WARNING: Multiple credit traps detected."
    elif medium >= 1:
        overall = "caution"
        status_text = "🟡 CAUTION: A credit trap has been detected."
    else:
        overall = "safe"
        status_text = "✅ SAFE: No significant credit traps detected."

    return {
        "total_traps": total,
        "detected_count": detected,
        "high_count": high,
        "medium_count": medium,
        "safe_count": safe,
        "overall_status": overall,
        "status_text": status_text,
    }


def generate_risk_explanation(risk_result, feature_importance):
    """
    Generate a human-readable explanation of the ML risk prediction.

    Args:
        risk_result: dict from ml_engine.predict_risk()
        feature_importance: list from ml_engine.get_feature_importance()

    Returns:
        dict with: risk_narrative, top_factors, risk_context
    """
    risk_label = risk_result["risk_label"]
    risk_score = risk_result["risk_score"]
    probabilities = risk_result["probabilities"]

    # Build narrative
    if risk_score >= 70:
        narrative = (
            f"Your financial profile indicates **{risk_label}** with a risk score of **{risk_score}/100**. "
            f"This is a concerning level that requires immediate attention. Multiple factors in your credit "
            f"behavior are contributing to elevated risk. Without corrective action, you may face increasing "
            f"financial strain and deteriorating credit health."
        )
    elif risk_score >= 40:
        narrative = (
            f"Your financial profile indicates **{risk_label}** with a risk score of **{risk_score}/100**. "
            f"While not critical, there are areas of concern that should be addressed. Some of your credit "
            f"habits may be pushing you toward higher risk territory."
        )
    else:
        narrative = (
            f"Your financial profile indicates **{risk_label}** with a risk score of **{risk_score}/100**. "
            f"Your credit behavior appears healthy. Continue maintaining good habits to preserve "
            f"your strong financial standing."
        )

    # Top contributing factors
    top_factors = []
    factor_descriptions = {
        "Annual_Income": "Your annual income level",
        "Credit_Score": "Your current credit score",
        "Number_of_Credit_Lines": "The number of active credit lines",
        "Credit_Utilization_Ratio": "How much of your available credit you're using",
        "Debt_To_Income_Ratio": "Your total debt relative to income",
        "Number_of_Late_Payments": "Your history of late payments",
        "Total_Spend_Last_Year": "Your total spending in the last year",
        "Avg_Transaction_Amount": "Your average transaction size",
        "Max_Transaction_Amount": "Your largest transaction amount",
    }

    for feat, imp in feature_importance[:5]:
        if imp > 0.01:
            top_factors.append({
                "feature": feat,
                "importance": round(imp * 100, 1),
                "description": factor_descriptions.get(feat, feat),
            })

    # Risk context
    probs = probabilities
    risk_context = (
        f"The model estimates a **{probs.get('Low Risk', 0)*100:.1f}%** chance of low risk, "
        f"**{probs.get('Medium Risk', 0)*100:.1f}%** chance of medium risk, and "
        f"**{probs.get('High Risk', 0)*100:.1f}%** chance of high risk based on your profile."
    )

    return {
        "risk_narrative": narrative,
        "top_factors": top_factors,
        "risk_context": risk_context,
    }


def generate_actionable_advice(user_data, trap_results, risk_result):
    """
    Generate personalized, actionable advice based on all analysis results.

    Returns:
        list of advice dicts with: priority, category, title, detail, impact
    """
    advice_list = []
    risk_score = risk_result["risk_score"]

    # Advice from traps
    for trap in trap_results:
        if trap.detected:
            priority = "🔴 Urgent" if trap.severity == "high" else "🟡 Important"
            advice_list.append({
                "priority": priority,
                "category": trap.trap_name,
                "title": f"Address: {trap.trap_name}",
                "detail": trap.suggestion,
                "impact": "High impact on financial health" if trap.severity == "high" else "Moderate impact on financial health",
            })

    # General financial health advice based on overall risk
    if risk_score >= 70:
        advice_list.append({
            "priority": "🔴 Urgent",
            "category": "Overall Strategy",
            "title": "Create an Emergency Debt Repayment Plan",
            "detail": "Your overall risk is high. Create a structured debt repayment plan using either the "
                      "avalanche method (paying highest-interest debts first) or the snowball method (paying "
                      "smallest debts first for psychological wins). Consider seeking financial counseling.",
            "impact": "Critical for long-term financial stability",
        })

    monthly_income = user_data.get("monthly_income", 0)
    if monthly_income > 0:
        credit_score = user_data.get("credit_score", 700)
        if credit_score < 650:
            advice_list.append({
                "priority": "🟡 Important",
                "category": "Credit Score",
                "title": "Rebuild Your Credit Score",
                "detail": f"Your credit score ({credit_score}) is below the good threshold (650+). "
                          f"Pay all bills on time, reduce utilization below 30%, avoid new credit "
                          f"applications, and check your credit report for errors.",
                "impact": "Improves loan eligibility and reduces interest rates",
            })

        outstanding = user_data.get("outstanding_balance", 0)
        monthly_emi = user_data.get("monthly_emi_amount", 0)
        savings_ratio = 1.0 - (monthly_emi + outstanding * 0.035) / max(monthly_income, 1)
        if savings_ratio < 0.20:
            advice_list.append({
                "priority": "🟡 Important",
                "category": "Savings",
                "title": "Build an Emergency Fund",
                "detail": f"After debt obligations, you have very little margin for savings. Aim to build "
                          f"an emergency fund covering 3-6 months of expenses (₹{monthly_income*3:,.0f} - "
                          f"₹{monthly_income*6:,.0f}). Start small with even ₹{max(monthly_income*0.05, 1000):,.0f}/month.",
                "impact": "Prevents future debt spirals during emergencies",
            })

    if not advice_list:
        advice_list.append({
            "priority": "✅ Maintain",
            "category": "General",
            "title": "Keep Up the Good Work!",
            "detail": "Your credit health looks good. Continue paying bills on time, keeping utilization "
                      "low, and monitoring your credit score regularly.",
            "impact": "Sustained financial wellness",
        })

    return advice_list


def compute_financial_metrics(user_data):
    """
    Compute key financial health metrics for the user.

    Returns:
        list of metric dicts with: name, value, status, description
    """
    monthly_income = max(user_data.get("monthly_income", 1), 1)
    credit_limit = max(user_data.get("credit_limit", 1), 1)
    outstanding = user_data.get("outstanding_balance", 0)
    monthly_emi = user_data.get("monthly_emi_amount", 0)
    late_payments = user_data.get("late_payments", 0)
    credit_score = user_data.get("credit_score", 700)
    num_cards = user_data.get("number_of_cards", 1)

    annual_income = monthly_income * 12
    utilization = outstanding / credit_limit
    total_debt = outstanding + (monthly_emi * 12)
    dti = total_debt / annual_income
    emi_ratio = monthly_emi / monthly_income

    # Monthly interest estimate
    monthly_interest = outstanding * 0.035

    # Free cash after obligations
    free_cash = monthly_income - monthly_emi - monthly_interest

    metrics = [
        {
            "name": "Credit Utilization",
            "value": f"{utilization*100:.1f}%",
            "status": "🟢" if utilization < 0.30 else ("🟡" if utilization < 0.50 else "🔴"),
            "description": "Ideal: Below 30%",
        },
        {
            "name": "Debt-to-Income Ratio",
            "value": f"{dti*100:.1f}%",
            "status": "🟢" if dti < 0.36 else ("🟡" if dti < 0.50 else "🔴"),
            "description": "Ideal: Below 36%",
        },
        {
            "name": "EMI-to-Income Ratio",
            "value": f"{emi_ratio*100:.1f}%",
            "status": "🟢" if emi_ratio < 0.30 else ("🟡" if emi_ratio < 0.40 else "🔴"),
            "description": "Ideal: Below 30%",
        },
        {
            "name": "Credit Score",
            "value": str(credit_score),
            "status": "🟢" if credit_score >= 750 else ("🟡" if credit_score >= 650 else "🔴"),
            "description": "Excellent: 750+, Good: 650-749",
        },
        {
            "name": "Late Payments",
            "value": str(late_payments),
            "status": "🟢" if late_payments == 0 else ("🟡" if late_payments <= 2 else "🔴"),
            "description": "Ideal: 0",
        },
        {
            "name": "Monthly Interest Burden",
            "value": f"₹{monthly_interest:,.0f}",
            "status": "🟢" if monthly_interest < monthly_income * 0.05 else (
                "🟡" if monthly_interest < monthly_income * 0.15 else "🔴"
            ),
            "description": "Estimated at 3.5% monthly rate",
        },
        {
            "name": "Free Cash After Obligations",
            "value": f"₹{max(free_cash, 0):,.0f}",
            "status": "🟢" if free_cash > monthly_income * 0.30 else (
                "🟡" if free_cash > 0 else "🔴"
            ),
            "description": "Income minus EMI and interest",
        },
        {
            "name": "Active Credit Cards",
            "value": str(num_cards),
            "status": "🟢" if num_cards <= 3 else ("🟡" if num_cards <= 5 else "🔴"),
            "description": "Recommended: 1-3 cards",
        },
    ]

    return metrics














