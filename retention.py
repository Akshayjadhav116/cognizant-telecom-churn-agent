def retention_recommendation_tool(
    risk_level,
    churn_drivers,
    customer_data
):
    """
    Generate personalized retention actions based on
    customer risk, churn drivers, and customer profile.
    """

    actions = []

    # Normalize values
    risk = str(risk_level).strip().lower()

    tenure = float(customer_data.get("tenure", 0))
    contract = str(customer_data.get("Contract", "")).strip().lower()
    internet = str(customer_data.get("InternetService", "")).strip().lower()
    tech_support = str(customer_data.get("TechSupport", "")).strip().lower()
    online_security = str(
        customer_data.get("OnlineSecurity", "")
    ).strip().lower()
    monthly_charges = float(
        customer_data.get("MonthlyCharges", 0)
    )
    senior_citizen = customer_data.get("SeniorCitizen", 0)

    # ---------------------------------------------------------
    # 1. Risk-based immediate action
    # ---------------------------------------------------------

    if risk == "high":
        actions.append(
            "Contact the customer proactively through the "
            "retention team because of the high churn risk."
        )

    elif risk == "medium":
        actions.append(
            "Engage the customer with a personalized retention "
            "offer before churn risk increases."
        )

    else:
        actions.append(
            "Maintain customer engagement through loyalty "
            "benefits and proactive service communication."
        )

    # ---------------------------------------------------------
    # 2. Short-tenure customer
    # ---------------------------------------------------------

    if tenure <= 6:
        actions.append(
            "Provide early-tenure onboarding support and check "
            "for installation, service, or usability issues."
        )

    # ---------------------------------------------------------
    # 3. Month-to-month contract
    # ---------------------------------------------------------

    if contract == "month-to-month":
        actions.append(
            "Offer an attractive one-year or two-year contract "
            "with a suitable incentive to improve customer retention."
        )

    # ---------------------------------------------------------
    # 4. High monthly charges
    # ---------------------------------------------------------

    if monthly_charges >= 80:
        actions.append(
            "Review the customer's monthly bill and offer a "
            "more suitable plan or targeted pricing incentive."
        )

    # ---------------------------------------------------------
    # 5. Missing technical support
    # ---------------------------------------------------------

    if tech_support == "no":
        actions.append(
            "Offer technical support assistance and proactively "
            "check whether unresolved service issues are affecting "
            "customer satisfaction."
        )

    # ---------------------------------------------------------
    # 6. Missing online security
    # ---------------------------------------------------------

    if online_security == "no":
        actions.append(
            "Offer online security as an optional value-added "
            "service to improve the customer's perceived service value."
        )

    # ---------------------------------------------------------
    # 7. Fiber optic customer
    # ---------------------------------------------------------

    if internet == "fiber optic":
        actions.append(
            "Review the customer's fiber-optic service experience "
            "and check for connectivity or service-quality concerns."
        )

    # ---------------------------------------------------------
    # 8. Senior citizen support
    # ---------------------------------------------------------

    if senior_citizen in [1, "1", True, "Yes", "yes"]:
        actions.append(
            "Provide assisted customer support and simple service "
            "guidance to improve the overall customer experience."
        )

    # ---------------------------------------------------------
    # 9. Remove duplicate actions
    # ---------------------------------------------------------

    unique_actions = []

    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)

    # ---------------------------------------------------------
    # 10. Return maximum 5 focused actions
    # ---------------------------------------------------------

    return unique_actions[:5]