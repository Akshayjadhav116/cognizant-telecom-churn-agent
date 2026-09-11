import streamlit as st
import requests
from agent import run_agent_structured


class join:
    """Format an iterable as a newline-separated bullet list.

    The helper keeps list rendering consistent and safely handles missing or
    non-string values returned by the AI backend.
    """

    def __init__(self, values, separator="\n", prefix="- "):
        self.values = values or []
        self.separator = separator
        self.prefix = prefix

    def __str__(self):
        return self.separator.join(
            f"{self.prefix}{value}" for value in self.values
        )

    def __call__(self):
        return str(self)


# -----------------------------
# Dashboard Styling
# -----------------------------

st.set_page_config(
    page_title="Telecom Customer Churn AI",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
        background-color: #1f2937;
        text-align: center;
    }

    .metric-title {
        font-size: 16px;
        color: #9ca3af;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Telecom Customer Churn AI",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📊 Telecom Customer Churn AI")
st.write(
    "AI-powered telecom customer churn prediction, "
    "explanation and retention strategy."
)

st.divider()


# ---------------------------------------------------------
# CUSTOMER INFORMATION
# ---------------------------------------------------------

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=5
    )


with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# ---------------------------------------------------------
# BILLING INFORMATION
# ---------------------------------------------------------

st.divider()

st.header("💳 Billing Information")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

with col2:
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=250.0,
        step=10.0
    )


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

st.divider()

analyze = st.button(
    "🔍 Analyze Customer",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# SEND DATA TO FASTAPI
# ---------------------------------------------------------

if analyze:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:

        with st.spinner("🤖 AI is analyzing the customer..."):

            result = run_agent_structured(customer)

        if not isinstance(result, dict):
            st.error("❌ The AI backend returned an invalid response.")
            st.code(str(result))
        else:
            st.success("Customer analysis completed successfully!")

            # -------------------------------------------------
            # AI RESULT
            # -------------------------------------------------

            st.divider()
            st.header("📈 AI Churn Analysis")

            analysis = f"""
Churn Probability: {result.get("churn_probability", 0) * 100:.2f}%
Churn Prediction: {result.get("churn_prediction", "N/A")}
Risk Level: {result.get("risk_level", "N/A")}

Churn Drivers:
{join(result.get("churn_drivers", []))}

Protective Factors:
{join(result.get("protective_factors", []))}

Retention Actions:
{join(result.get("retention_actions", []))}
"""
            if not isinstance(analysis, str):
                analysis = str(analysis)

            # ------------------------------
            # KEY METRICS
            # ------------------------------
            import re

            probability_match = re.search(
                r"Churn Probability:\*?\*?\s*([0-9]+(?:\.[0-9]+)?)%",
                analysis,
                re.IGNORECASE,
            )

            prediction_match = re.search(
                r"Churn Prediction:\*?\*?\s*(Churn|No Churn)",
                analysis,
                re.IGNORECASE,
            )

            risk_match = re.search(
                r"Risk Level:\*?\*?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                analysis,
                re.IGNORECASE,
            )

            probability = probability_match.group(1) if probability_match else "N/A"
            prediction = prediction_match.group(1).strip() if prediction_match else "N/A"
            risk = risk_match.group(1).strip() if risk_match else "N/A"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Churn Probability", f"{probability}%")

            with col2:
                st.metric("Prediction", prediction)

            with col3:
                st.metric("Risk Level", risk)

            st.divider()
            st.markdown(analysis)

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to FastAPI backend.\n\n"
            "Make sure the backend is running on "
            "http://127.0.0.1:8000"
        )

    except requests.exceptions.Timeout:

        st.error(
            "⏳ The AI request took too long. "
            "Please try again."
        )

    except Exception as e:

        st.error(f"Unexpected error: {e}")