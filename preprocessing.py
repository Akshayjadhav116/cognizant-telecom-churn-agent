import pandas as pd


# Binary columns used by the ML team's preprocessing
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]


# Categorical columns one-hot encoded by the ML team
MULTI_CAT_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]


def preprocess_customer(customer_data):
    """
    Convert raw customer information into the format
    expected by the trained churn model.
    """

    # Convert dictionary to DataFrame
    df = pd.DataFrame([customer_data])

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        ).fillna(0)

    # Convert binary columns
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({
                "Yes": 1,
                "No": 0,
                "Male": 1,
                "Female": 0
            })

    # One-hot encode categorical columns
    existing_cat_cols = [
        col for col in MULTI_CAT_COLS
        if col in df.columns
    ]

    if existing_cat_cols:
        df = pd.get_dummies(
            df,
            columns=existing_cat_cols,
            drop_first=True
        )

    return df