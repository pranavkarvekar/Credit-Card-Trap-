"""
Credit Card Trap - ML Model Training & Persistence
=====================================================
Trains a DecisionTreeClassifier on Credit_Card_Dataset.csv to predict
risk levels (Low=0, Medium=1, High=2). Based on CreditCardTrap.ipynb.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "Credit_Card_Dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

FEATURES = [
    "Annual_Income",
    "Credit_Score",
    "Number_of_Credit_Lines",
    "Credit_Utilization_Ratio",
    "Debt_To_Income_Ratio",
    "Number_of_Late_Payments",
    "Total_Spend_Last_Year",
    "Avg_Transaction_Amount",
    "Max_Transaction_Amount",
]

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


# ── Risk Assignment (from notebook) ────────────────────────────────────────────

def assign_risk(row):
    """Assign a risk level based on rule-based thresholds."""
    if row["Defaulted"] == 1 or row["Credit_Score"] < 560:
        return 2  # High Risk
    elif row["Credit_Score"] < 650 or row["Number_of_Late_Payments"] >= 2:
        return 1  # Medium Risk
    else:
        return 0  # Low Risk


# ── Training ───────────────────────────────────────────────────────────────────

def train_model():
    """Train the risk model and save to disk."""
    print("📂 Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("🏷️  Assigning risk labels...")
    df["Risk_Level"] = df.apply(assign_risk, axis=1)

    X = df[FEATURES]
    y = df["Risk_Level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🤖 Training DecisionTreeClassifier...")
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=list(RISK_LABELS.values())))

    print(f"💾 Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("✅ Model saved successfully!\n")

    return model, acc


if __name__ == "__main__":
    train_model()
