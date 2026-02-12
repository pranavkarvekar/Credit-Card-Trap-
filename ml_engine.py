"""
Credit Card Trap - ML Risk Prediction Engine
================================================
Loads the pre-trained model and provides risk predictions
with probability distributions and feature importance.
"""

import os
import numpy as np
import pandas as pd
import joblib

from model import FEATURES, RISK_LABELS, MODEL_PATH, train_model


def load_model():
    """Load the pre-trained model from disk. Train if not found."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not found. Training now...")
        train_model()
    return joblib.load(MODEL_PATH)


def prepare_user_features(user_data):
    """
    Convert user inputs into the feature vector expected by the model.
    Maps user-friendly inputs to model features.
    """
    monthly_income = user_data.get("monthly_income", 50000)
    annual_income = monthly_income * 12

    credit_score = user_data.get("credit_score", 700)
    num_cards = user_data.get("number_of_cards", 1)
    credit_limit = user_data.get("credit_limit", 100000)
    outstanding = user_data.get("outstanding_balance", 0)
    late_payments = user_data.get("late_payments", 0)
    total_spend = user_data.get("total_spend_last_year", 0)
    avg_txn = user_data.get("avg_transaction_amount", 0)
    max_txn = user_data.get("max_transaction_amount", 0)
    monthly_emi = user_data.get("monthly_emi_amount", 0)

    # Calculate derived features
    utilization = outstanding / max(credit_limit, 1)
    total_debt = outstanding + (monthly_emi * 12)
    debt_to_income = total_debt / max(annual_income, 1)

    feature_dict = {
        "Annual_Income": annual_income,
        "Credit_Score": credit_score,
        "Number_of_Credit_Lines": num_cards,
        "Credit_Utilization_Ratio": round(utilization, 4),
        "Debt_To_Income_Ratio": round(debt_to_income, 4),
        "Number_of_Late_Payments": late_payments,
        "Total_Spend_Last_Year": total_spend,
        "Avg_Transaction_Amount": avg_txn,
        "Max_Transaction_Amount": max_txn,
    }

    return pd.DataFrame([feature_dict])[FEATURES]


def predict_risk(model, user_data):
    """
    Predict risk level for a user.

    Returns:
        dict with keys: risk_level, risk_label, probabilities, risk_score
    """
    X = prepare_user_features(user_data)
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Build probability dict for all classes the model knows
    prob_dict = {}
    for i, cls in enumerate(model.classes_):
        prob_dict[RISK_LABELS.get(cls, f"Class {cls}")] = float(probabilities[i])

    # Ensure all risk labels are present
    for label in RISK_LABELS.values():
        if label not in prob_dict:
            prob_dict[label] = 0.0

    # Calculate a 0-100 risk score
    risk_score = (
        prob_dict.get("Low Risk", 0) * 10 +
        prob_dict.get("Medium Risk", 0) * 50 +
        prob_dict.get("High Risk", 0) * 95
    )

    return {
        "risk_level": int(prediction),
        "risk_label": RISK_LABELS.get(prediction, "Unknown"),
        "probabilities": prob_dict,
        "risk_score": round(risk_score, 1),
    }


def get_feature_importance(model):
    """
    Get ranked feature importances from the trained model.

    Returns:
        list of (feature_name, importance) tuples, sorted descending
    """
    importances = model.feature_importances_
    feature_importance = list(zip(FEATURES, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    return feature_importance
