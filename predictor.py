import joblib
import pandas as pd

from preprocessing import preprocess_customer


# Load the trained ML model
model = joblib.load("churn_final_model.pkl")

# Load the exact features expected by the model
features = joblib.load("churn_model_features.pkl")


def predict_churn(customer_data):
    """
    Predict customer churn from raw customer information.
    """

    # Step 1: Preprocess raw customer information
    customer_df = preprocess_customer(customer_data)

    # Step 2: Add any missing model features
    for feature in features:
        if feature not in customer_df.columns:
            customer_df[feature] = 0

    # Step 3: Keep exactly the same feature order
    customer_df = customer_df[features]

    # Step 4: Get churn probability
    probability = model.predict_proba(customer_df)[0][1]

    # Step 5: Prediction
    prediction = "Churn" if probability >= 0.5 else "No Churn"

    # Step 6: Risk level
    if probability >= 0.65:
        risk_level = "High"
    elif probability >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": prediction,
        "risk_level": risk_level
    }