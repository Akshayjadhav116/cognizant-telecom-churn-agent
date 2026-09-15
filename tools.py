from predictor import predict_churn
from explainer import explain_customer


def churn_prediction_tool(customer_data):
    """
    Tool 1:
    Predict the customer's probability of churn.
    """

    result = predict_churn(customer_data)

    return result


def churn_explanation_tool(customer_data):
    """
    Tool 2:
    Explain why the customer may churn.
    """

    result = explain_customer(customer_data, top_n=5)

    return {
        "churn_drivers": result["churn_drivers"],
        "protective_factors": result["protective_factors"]
    }