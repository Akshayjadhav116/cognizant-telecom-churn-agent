def retention_recommendation_tool(
    risk_level,
    churn_drivers,
    customer_data
):
    """
    Recommend retention actions based on churn risk
    and the customer's churn drivers.
    """

    recommendations = []

    # High-risk customer
    if risk_level == "High":
        recommendations.append(
            "Contact the customer proactively through the retention team."
        )

        recommendations.append(
            "Offer a personalized retention incentive based on the customer's needs."
        )

    # Medium-risk customer
    elif risk_level == "Medium":
        recommendations.append(
            "Send a personalized retention offer before the customer decides to leave."
        )

        recommendations.append(
            "Review the customer's service experience and identify possible improvements."
        )

    # Low-risk customer
    else:
        recommendations.append(
            "Maintain engagement through loyalty benefits and personalized offers."
        )

    # Check churn drivers
    driver_names = [
        driver["feature"]
        for driver in churn_drivers
    ]

    if "tenure" in driver_names:
        recommendations.append(
            "Provide an onboarding or loyalty program to strengthen customer engagement."
        )

    if "Contract_Month-to-month" in driver_names:
        recommendations.append(
            "Consider encouraging the customer to move to a longer-term contract."
        )

    if "MonthlyCharges" in driver_names:
        recommendations.append(
            "Review the customer's monthly charges and consider a suitable plan."
        )

    if "PaymentMethod_Electronic check" in driver_names:
        recommendations.append(
            "Review the customer's payment experience and offer convenient payment options."
        )

    if "InternetService_Fiber optic" in driver_names:
        recommendations.append(
            "Check whether the customer's internet service experience requires support."
        )

    return {
        "risk_level": risk_level,
        "recommended_actions": recommendations
    }