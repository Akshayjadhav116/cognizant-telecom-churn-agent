import joblib
import pandas as pd
import shap

from preprocessing import preprocess_customer


# Load trained model
model = joblib.load("churn_final_model.pkl")

# Load the exact model features
features = joblib.load("churn_model_features.pkl")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)


def explain_customer(customer_data, top_n=5):
    """
    Explain why the ML model predicts churn
    for a customer.
    """

    # Step 1: Preprocess customer
    customer_df = preprocess_customer(customer_data)

    # Step 2: Add missing features
    for feature in features:
        if feature not in customer_df.columns:
            customer_df[feature] = 0

    # Step 3: Keep exact model feature order
    customer_df = customer_df[features]

    # Step 4: Calculate SHAP values
    shap_values = explainer.shap_values(customer_df)

    # Get values for this customer
    values = shap_values[0]

    # Create feature explanation table
    explanation = pd.DataFrame({
        "feature": features,
        "shap_value": values,
        "customer_value": customer_df.iloc[0].values
    })

    # Sort by absolute SHAP importance
    explanation["absolute_shap"] = explanation["shap_value"].abs()

    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False
    )

    # Top factors
    top_features = explanation.head(top_n)

    # Factors increasing churn
    churn_drivers = top_features[
        top_features["shap_value"] > 0
    ][["feature", "shap_value", "customer_value"]]

    # Factors reducing churn
    protective_factors = top_features[
        top_features["shap_value"] < 0
    ][["feature", "shap_value", "customer_value"]]

    return {
        "churn_drivers": churn_drivers.to_dict("records"),
        "protective_factors": protective_factors.to_dict("records"),
        "all_features": explanation.to_dict("records")
    }