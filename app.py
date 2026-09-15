import streamlit as st
import plotly.graph_objects as go

from agent import run_agent_structured
from predictor import predict_churn
from explainer import explain_customer
from retention import retention_recommendation_tool


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telecom Customer Churn AI",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# HELPER: FORMAT ML FEATURE NAMES
# =========================================================

def format_feature_name(feature, customer_data, direction):
    """
    Convert raw ML/SHAP feature names into
    accurate business-friendly explanations
    using the customer's actual values.
    """

    tenure = float(customer_data.get("tenure", 0))
    monthly_charges = float(customer_data.get("MonthlyCharges", 0))
    total_charges = float(customer_data.get("TotalCharges", 0))

    contract = str(
        customer_data.get("Contract", "")
    ).strip()

    internet = str(
        customer_data.get("InternetService", "")
    ).strip()

    payment = str(
        customer_data.get("PaymentMethod", "")
    ).strip()

    tech_support = str(
        customer_data.get("TechSupport", "")
    ).strip()

    online_security = str(
        customer_data.get("OnlineSecurity", "")
    ).strip()

    online_backup = str(
        customer_data.get("OnlineBackup", "")
    ).strip()

    device_protection = str(
        customer_data.get("DeviceProtection", "")
    ).strip()

    paperless_billing = str(
        customer_data.get("PaperlessBilling", "")
    ).strip()

    senior_citizen = customer_data.get(
        "SeniorCitizen", 0
    )


    # -----------------------------------------------------
    # TENURE
    # -----------------------------------------------------

    if feature == "tenure":

        if tenure <= 6:
            return f"Short Customer Tenure ({int(tenure)} months)"

        elif tenure >= 36:
            return f"Long Customer Tenure ({int(tenure)} months)"

        else:
            return f"Customer Tenure ({int(tenure)} months)"


    # -----------------------------------------------------
    # CONTRACT
    # -----------------------------------------------------

    if feature == "Contract_Month-to-month":

        if contract == "Month-to-month":
            return "Month-to-Month Contract"

        return "Not on Month-to-Month Contract"


    if feature == "Contract_One year":

        if contract == "One year":
            return "One-Year Contract"

        return "Not on One-Year Contract"


    if feature == "Contract_Two year":

        if contract == "Two year":
            return "Two-Year Contract"

        return "Not on Two-Year Contract"


    # -----------------------------------------------------
    # INTERNET SERVICE
    # -----------------------------------------------------

    if feature == "InternetService_Fiber optic":

        if internet == "Fiber optic":
            return "Fiber Optic Internet Service"

        elif internet == "DSL":
            return "DSL Internet Service"

        return "No Internet Service"


    # -----------------------------------------------------
    # PAYMENT METHOD
    # -----------------------------------------------------

    if feature == "PaymentMethod_Electronic check":

        if payment == "Electronic check":
            return "Electronic Check Payment"

        return f"{payment} Payment"


    # -----------------------------------------------------
    # TECH SUPPORT
    # -----------------------------------------------------

    if feature == "TechSupport_No":

        if tech_support == "No":
            return "No Technical Support"

        return "Technical Support Available"


    # -----------------------------------------------------
    # ONLINE SECURITY
    # -----------------------------------------------------

    if feature == "OnlineSecurity_No":

        if online_security == "No":
            return "No Online Security"

        return "Online Security Enabled"


    # -----------------------------------------------------
    # ONLINE BACKUP
    # -----------------------------------------------------

    if feature == "OnlineBackup_No":

        if online_backup == "No":
            return "No Online Backup"

        return "Online Backup Enabled"


    # -----------------------------------------------------
    # DEVICE PROTECTION
    # -----------------------------------------------------

    if feature == "DeviceProtection_No":

        if device_protection == "No":
            return "No Device Protection"

        return "Device Protection Enabled"


    # -----------------------------------------------------
    # PAPERLESS BILLING
    # -----------------------------------------------------

    if feature == "PaperlessBilling":

        if paperless_billing == "Yes":
            return "Paperless Billing Enabled"

        return "Paperless Billing Disabled"


    # -----------------------------------------------------
    # SENIOR CITIZEN
    # -----------------------------------------------------

    if feature == "SeniorCitizen":

        if senior_citizen in [1, "1", True, "Yes", "yes"]:
            return "Senior Citizen"

        return "Non-Senior Customer"


    # -----------------------------------------------------
    # BILLING
    # -----------------------------------------------------

    if feature == "MonthlyCharges":

        return f"Monthly Charges (${monthly_charges:.2f})"


    if feature == "TotalCharges":

        return f"Total Charges (${total_charges:.2f})"


    # -----------------------------------------------------
    # DEFAULT
    # -----------------------------------------------------

    return str(feature).replace(
        "_",
        " "
    ).title()


# =========================================================
# HELPER: FORMAT LIST
# =========================================================

def format_feature_list(features, customer_data, direction):
    """
    Format a list of SHAP/AI features for display.
    """

    formatted = []

    for feature in features:

        # Gemini may sometimes return dictionaries
        if isinstance(feature, dict):
            feature = feature.get("feature", str(feature))

        formatted.append(
            format_feature_name(
                feature,
                customer_data,
                direction
            )
        )

    return formatted


# =========================================================
# LOCAL FALLBACK
# =========================================================

def run_local_fallback(customer):
    """
    Run local ML + SHAP + retention analysis
    when Gemini is unavailable.
    """

    # -------------------------
    # 1. ML Prediction
    # -------------------------

    prediction_result = predict_churn(customer)

    # -------------------------
    # 2. SHAP Explanation
    # -------------------------

    explanation_result = explain_customer(
        customer,
        top_n=5
    )

    # -------------------------
    # 3. Retention Strategy
    # -------------------------

    retention_actions = retention_recommendation_tool(
        prediction_result["risk_level"],
        explanation_result["churn_drivers"],
        customer
    )

    # -------------------------
    # 4. Format Features
    # -------------------------

    churn_drivers = [
        format_feature_name(
            item["feature"],
            customer,
            "risk"
        )
        for item in explanation_result["churn_drivers"]
    ]

    protective_factors = [
        format_feature_name(
            item["feature"],
            customer,
            "protective"
        )
        for item in explanation_result["protective_factors"]
    ]

    # -------------------------
    # 5. Final Result
    # -------------------------

    return {
        "churn_prediction":
            prediction_result["prediction"],

        "churn_probability":
            prediction_result["churn_probability"],

        "risk_level":
            prediction_result["risk_level"],

        "churn_drivers":
            churn_drivers,

        "protective_factors":
            protective_factors,

        "retention_actions":
            retention_actions
    }


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
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

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Telecom Customer Churn AI")

st.write(
    "AI-powered telecom customer churn prediction, "
    "explanation and retention strategy."
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
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


# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# COLUMN 3
# ---------------------------------------------------------

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


# =========================================================
# BILLING INFORMATION
# =========================================================

st.divider()

st.header("💳 Billing Information")

col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# CONTRACT
# ---------------------------------------------------------

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


# ---------------------------------------------------------
# PAPERLESS BILLING
# ---------------------------------------------------------

with col2:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


# ---------------------------------------------------------
# PAYMENT METHOD
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# CHARGES
# ---------------------------------------------------------

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


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.divider()

analyze = st.button(
    "🔍 Analyze Customer",
    type="primary",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    # -----------------------------------------------------
    # CUSTOMER DATA
    # -----------------------------------------------------

    customer = {

        "gender": gender,

        "SeniorCitizen":
            senior_citizen,

        "Partner":
            partner,

        "Dependents":
            dependents,

        "tenure":
            tenure,

        "PhoneService":
            phone_service,

        "MultipleLines":
            multiple_lines,

        "InternetService":
            internet_service,

        "OnlineSecurity":
            online_security,

        "OnlineBackup":
            online_backup,

        "DeviceProtection":
            device_protection,

        "TechSupport":
            tech_support,

        "StreamingTV":
            streaming_tv,

        "StreamingMovies":
            streaming_movies,

        "Contract":
            contract,

        "PaperlessBilling":
            paperless_billing,

        "PaymentMethod":
            payment_method,

        "MonthlyCharges":
            monthly_charges,

        "TotalCharges":
            total_charges
    }


    # -----------------------------------------------------
    # RUN GEMINI
    # -----------------------------------------------------

    with st.spinner(
        "🤖 AI is analyzing the customer..."
    ):

        try:

            result = run_agent_structured(
                customer
            )

            st.info(
                "🤖 Analysis powered by Gemini Agent"
            )

        except Exception:

            # -------------------------------------------------
            # GEMINI UNAVAILABLE → LOCAL FALLBACK
            # -------------------------------------------------

            result = run_local_fallback(
                customer
            )

            st.warning(
                "⚡ Gemini is temporarily unavailable. "
                "Using local ML + SHAP + retention analysis."
            )


    # =====================================================
    # VALIDATE RESULT
    # =====================================================

    if not isinstance(result, dict):

        st.error(
            "❌ The AI backend returned an invalid response."
        )

        st.code(str(result))

    else:

        st.success(
            "Customer analysis completed successfully!"
        )


        # =================================================
        # AI CHURN ANALYSIS
        # =================================================

        st.divider()

        st.header(
            "📈 AI Churn Analysis"
        )


        # -------------------------------------------------
        # GET VALUES
        # -------------------------------------------------

        churn_probability = float(
            result.get(
                "churn_probability",
                0
            )
        )

        churn_prediction = str(
            result.get(
                "churn_prediction",
                "N/A"
            )
        )

        risk_level_display = str(
            result.get(
                "risk_level",
                "N/A"
            )
        )


        # -------------------------------------------------
        # FORMAT FEATURES
        # -------------------------------------------------

        formatted_churn_drivers = (
            format_feature_list(
                result.get(
                    "churn_drivers",
                    []
                ),
                customer,
                "risk"
            )
        )


        formatted_protective_factors = (
            format_feature_list(
                result.get(
                    "protective_factors",
                    []
                ),
                customer,
                "protective"
            )
        )


        retention_actions = result.get(
            "retention_actions",
            []
        )


        # =================================================
        # KEY METRICS
        # =================================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Prediction",
                churn_prediction
            )


        with col3:

            st.metric(
                "Risk Level",
                risk_level_display
            )


        # =================================================
        # PROBABILITY GAUGE
        # =================================================

        st.markdown(
            "### 📊 Churn Probability Dashboard"
        )


        probability_value = max(
            0.0,
            min(
                100.0,
                churn_probability * 100
            )
        )


        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability_value,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 32
                    }
                },

                title={
                    "text": "Churn Risk Score"
                },

                gauge={

                    "axis": {
                        "range": [0, 100],
                        "ticksuffix": "%"
                    },

                    "bar": {
                        "color": "#ff4b4b"
                    },

                    "steps": [

                        {
                            "range": [0, 35],
                            "color": "#1f7a4d"
                        },

                        {
                            "range": [35, 65],
                            "color": "#c98b00"
                        },

                        {
                            "range": [65, 100],
                            "color": "#b52b2b"
                        }
                    ],

                    "threshold": {

                        "line": {
                            "color": "white",
                            "width": 4
                        },

                        "thickness": 0.75,

                        "value":
                            probability_value
                    }
                }
            )
        )


        fig.update_layout(

            height=350,

            margin=dict(
                l=30,
                r=30,
                t=60,
                b=20
            ),

            paper_bgcolor=
                "rgba(0,0,0,0)",

            font={
                "color": "white"
            }
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


        # =================================================
        # CHURN DRIVERS
        # =================================================

        st.divider()

        st.subheader(
            "🔴 Churn Drivers"
        )


        if formatted_churn_drivers:

            for driver in formatted_churn_drivers:

                st.markdown(
                    f"- **{driver}**"
                )

        else:

            st.write(
                "No major churn drivers identified."
            )


        # =================================================
        # PROTECTIVE FACTORS
        # =================================================

        st.subheader(
            "🟢 Protective Factors"
        )


        if formatted_protective_factors:

            for factor in formatted_protective_factors:

                st.markdown(
                    f"- **{factor}**"
                )

        else:

            st.write(
                "No significant protective factors identified."
            )


        # =================================================
        # RETENTION ACTIONS
        # =================================================

        st.subheader(
            "🎯 Retention Actions"
        )


        if retention_actions:

            for action in retention_actions:

                st.markdown(
                    f"- {action}"
                )

        else:

            st.write(
                "No specific retention action generated."
            )


        # =================================================
        # EXPLAINABILITY NOTE
        # =================================================

        st.divider()

        st.caption(
            "ℹ️ Protective factors and churn drivers are "
            "model-attributed factors based on SHAP "
            "explanations for this prediction. They do not "
            "necessarily represent causal relationships."
        )