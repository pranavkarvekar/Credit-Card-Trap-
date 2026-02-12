"""
Credit Card Trap - Rule-Based Trap Detection Engine
======================================================
Detects 6 specific credit card traps based on user-provided inputs.
"""


class TrapResult:
    """Result of a single trap detection rule."""

    def __init__(self, trap_name, detected, severity, icon, description, consequences, suggestion):
        self.trap_name = trap_name
        self.detected = detected
        self.severity = severity          # "high", "medium", "low", "safe"
        self.icon = icon
        self.description = description
        self.consequences = consequences
        self.suggestion = suggestion

    def to_dict(self):
        return {
            "trap_name": self.trap_name,
            "detected": self.detected,
            "severity": self.severity,
            "icon": self.icon,
            "description": self.description,
            "consequences": self.consequences,
            "suggestion": self.suggestion,
        }


def check_minimum_payment_trap(user_data):
    """Trap 1: User pays only the minimum due."""
    payment_behavior = user_data.get("payment_behavior", "full").lower()
    outstanding = user_data.get("outstanding_balance", 0)

    detected = payment_behavior == "minimum" and outstanding > 0

    if detected:
        # Estimate interest burden
        monthly_interest_rate = 0.035  # ~3.5% per month (typical CC rate)
        annual_interest = outstanding * monthly_interest_rate * 12
        return TrapResult(
            trap_name="Minimum Payment Trap",
            detected=True,
            severity="high",
            icon="💳",
            description=f"You are paying only the minimum due on an outstanding balance of ₹{outstanding:,.0f}. "
                        f"This means most of your payment goes toward interest, not principal.",
            consequences=f"At ~3.5% monthly interest, you could pay approximately ₹{annual_interest:,.0f}/year "
                         f"in interest alone. Your debt may never be fully repaid, trapping you in an endless cycle.",
            suggestion="Pay at least 2-3x the minimum due each month. Prioritize clearing high-interest credit card "
                       "debt before any discretionary spending. Consider balance transfer to a lower-rate option.",
        )
    else:
        return TrapResult(
            trap_name="Minimum Payment Trap",
            detected=False,
            severity="safe",
            icon="💳",
            description="You are not stuck in the minimum payment trap.",
            consequences="",
            suggestion="Continue paying your full balance or more than the minimum each month.",
        )


def check_credit_overuse_trap(user_data):
    """Trap 2: Credit utilization ratio is too high."""
    credit_limit = user_data.get("credit_limit", 1)
    outstanding = user_data.get("outstanding_balance", 0)

    if credit_limit <= 0:
        credit_limit = 1  # Avoid division by zero

    utilization = outstanding / credit_limit

    if utilization > 0.75:
        severity = "high"
        detected = True
    elif utilization > 0.50:
        severity = "medium"
        detected = True
    else:
        severity = "safe"
        detected = False

    if detected:
        return TrapResult(
            trap_name="Credit Overuse Trap",
            detected=True,
            severity=severity,
            icon="📊",
            description=f"Your credit utilization is {utilization*100:.1f}% (₹{outstanding:,.0f} of ₹{credit_limit:,.0f} limit). "
                        f"Experts recommend keeping utilization below 30%.",
            consequences="High utilization damages your credit score, increases interest burden, "
                         "and makes you a higher-risk borrower. It reduces available credit for emergencies.",
            suggestion=f"Reduce your outstanding balance to below ₹{credit_limit*0.30:,.0f} (30% of your limit). "
                       f"Consider requesting a credit limit increase or spreading across multiple cards.",
        )
    else:
        return TrapResult(
            trap_name="Credit Overuse Trap",
            detected=False,
            severity="safe",
            icon="📊",
            description=f"Your credit utilization is {utilization*100:.1f}%, which is within safe limits.",
            consequences="",
            suggestion="Keep maintaining your utilization below 30% for optimal credit score health.",
        )


def check_cash_withdrawal_trap(user_data):
    """Trap 3: Cash advances on credit card."""
    cash_withdrawal = user_data.get("cash_withdrawal_amount", 0)

    detected = cash_withdrawal > 0

    if detected:
        # Cash advance fee + interest from day 1
        fee = max(cash_withdrawal * 0.025, 500)  # Typical 2.5% or ₹500 min
        monthly_interest = cash_withdrawal * 0.035
        return TrapResult(
            trap_name="Cash Withdrawal Trap",
            detected=True,
            severity="high" if cash_withdrawal > 10000 else "medium",
            icon="🏧",
            description=f"You withdrew ₹{cash_withdrawal:,.0f} as a cash advance from your credit card. "
                        f"Cash advances attract immediate charges with no interest-free period.",
            consequences=f"You'll pay ~₹{fee:,.0f} in transaction fees plus ~₹{monthly_interest:,.0f}/month in interest "
                         f"from day one. There is no grace period for cash advances.",
            suggestion="Never use credit cards for cash advances. Use debit cards, emergency funds, or short-term "
                       "personal loans which have much lower interest rates.",
        )
    else:
        return TrapResult(
            trap_name="Cash Withdrawal Trap",
            detected=False,
            severity="safe",
            icon="🏧",
            description="You have not made any cash withdrawals on your credit card.",
            consequences="",
            suggestion="Avoid cash advances on credit cards as they carry the highest charges.",
        )


def check_emi_trap(user_data):
    """Trap 4: Excessive EMI burden relative to income."""
    monthly_income = user_data.get("monthly_income", 1)
    monthly_emi = user_data.get("monthly_emi_amount", 0)
    num_emis = user_data.get("number_of_emis", 0)

    if monthly_income <= 0:
        monthly_income = 1

    emi_ratio = monthly_emi / monthly_income

    if emi_ratio > 0.50:
        severity = "high"
        detected = True
    elif emi_ratio > 0.40:
        severity = "medium"
        detected = True
    else:
        severity = "safe"
        detected = False

    if detected:
        return TrapResult(
            trap_name="EMI Overload Trap",
            detected=True,
            severity=severity,
            icon="🔄",
            description=f"Your total EMI burden is {emi_ratio*100:.1f}% of monthly income "
                        f"(₹{monthly_emi:,.0f} EMIs across {num_emis} loans on ₹{monthly_income:,.0f} income). "
                        f"This exceeds the safe threshold of 40%.",
            consequences="Excessive EMI burden leaves insufficient income for essentials and emergencies. "
                         "Missing EMI payments triggers penalties and severely damages credit score.",
            suggestion=f"Reduce EMI obligations to below 40% of income (₹{monthly_income*0.40:,.0f}). "
                       f"Consider prepaying smaller loans, consolidating debt, or avoiding new EMI purchases.",
        )
    else:
        return TrapResult(
            trap_name="EMI Overload Trap",
            detected=False,
            severity="safe",
            icon="🔄",
            description=f"Your EMI-to-income ratio is {emi_ratio*100:.1f}%, within comfortable limits.",
            consequences="",
            suggestion="Keep your total EMI obligations below 40% of monthly income.",
        )


def check_late_payment_trap(user_data):
    """Trap 5: Missed due dates / late payments."""
    late_payments = user_data.get("late_payments", 0)

    if late_payments >= 3:
        severity = "high"
        detected = True
    elif late_payments >= 1:
        severity = "medium"
        detected = True
    else:
        severity = "safe"
        detected = False

    if detected:
        penalty_estimate = late_payments * 1200  # Approx ₹1000-1500 per late payment
        return TrapResult(
            trap_name="Late Payment Trap",
            detected=True,
            severity=severity,
            icon="⏰",
            description=f"You have {late_payments} late payment(s) on record. "
                        f"Each late payment incurs penalties and negatively impacts your credit score.",
            consequences=f"Estimated penalties: ~₹{penalty_estimate:,.0f}. Late payments remain on credit reports "
                         f"for up to 7 years and can reduce your credit score by 50-100 points per incident.",
            suggestion="Set up auto-pay for at least the minimum due. Use calendar reminders 3-5 days before "
                       "due dates. Negotiate with your bank to waive late fees if you have a good track record.",
        )
    else:
        return TrapResult(
            trap_name="Late Payment Trap",
            detected=False,
            severity="safe",
            icon="⏰",
            description="You have no late payments on record. Great discipline!",
            consequences="",
            suggestion="Maintain your perfect payment record by using auto-pay features.",
        )


def check_unsafe_practices_trap(user_data):
    """Trap 6: Too many credit lines combined with high debt-to-income."""
    num_cards = user_data.get("number_of_cards", 1)
    monthly_income = user_data.get("monthly_income", 1)
    outstanding = user_data.get("outstanding_balance", 0)
    monthly_emi = user_data.get("monthly_emi_amount", 0)

    if monthly_income <= 0:
        monthly_income = 1

    annual_income = monthly_income * 12
    total_debt = outstanding + (monthly_emi * 12)
    debt_to_income = total_debt / annual_income

    issues = []
    if num_cards > 4:
        issues.append(f"You hold {num_cards} credit cards (recommended: ≤3)")
    if debt_to_income > 0.50:
        issues.append(f"Your debt-to-income ratio is {debt_to_income*100:.1f}% (should be <50%)")

    detected = len(issues) > 0

    if num_cards > 5 and debt_to_income > 0.60:
        severity = "high"
    elif detected:
        severity = "medium"
    else:
        severity = "safe"

    if detected:
        return TrapResult(
            trap_name="Unsafe Practices Trap",
            detected=True,
            severity=severity,
            icon="⚠️",
            description="Unsafe credit practices detected: " + "; ".join(issues) + ".",
            consequences="Multiple cards increase temptation to overspend and make tracking payments difficult. "
                         "High debt-to-income ratio makes you a risky borrower and can lead to a debt spiral.",
            suggestion="Close unused credit cards starting with ones that have annual fees. Create a debt repayment "
                       "plan (avalanche or snowball method). Avoid opening new credit lines until debt is manageable.",
        )
    else:
        return TrapResult(
            trap_name="Unsafe Practices Trap",
            detected=False,
            severity="safe",
            icon="⚠️",
            description="No unsafe credit practices detected.",
            consequences="",
            suggestion="Continue managing a reasonable number of credit lines and keeping debt-to-income low.",
        )


def run_all_checks(user_data):
    """Run all trap detection rules and return results."""
    checks = [
        check_minimum_payment_trap,
        check_credit_overuse_trap,
        check_cash_withdrawal_trap,
        check_emi_trap,
        check_late_payment_trap,
        check_unsafe_practices_trap,
    ]

    results = []
    for check in checks:
        results.append(check(user_data))

    return results
