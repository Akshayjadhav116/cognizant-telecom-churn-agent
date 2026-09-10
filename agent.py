import os
import json

from dotenv import load_dotenv
from google import genai

from tools import churn_prediction_tool
from tools import churn_explanation_tool
from retention import retention_recommendation_tool


# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def prediction_agent_tool(customer_json: str) -> dict:
    """
    Predict the churn probability, prediction, and risk level
    for a telecom customer.

    Args:
        customer_json: Customer information as a JSON string.

    Returns:
        Churn prediction result.
    """

    customer_data = json.loads(customer_json)

    return churn_prediction_tool(customer_data)


def explanation_agent_tool(customer_json: str) -> dict:
    """
    Explain the main reasons why a telecom customer may churn.

    Args:
        customer_json: Customer information as a JSON string.

    Returns:
        Churn drivers and protective factors.
    """

    customer_data = json.loads(customer_json)

    return churn_explanation_tool(customer_data)

def retention_agent_tool(
    risk_level: str,
    churn_drivers_json: str,
    customer_json: str
) -> dict:
    """
    Agent-friendly wrapper for the retention recommendation tool.
    """

    churn_drivers = json.loads(churn_drivers_json)
    customer_data = json.loads(customer_json)

    return retention_recommendation_tool(
        risk_level,
        churn_drivers,
        customer_data
    )
def run_agent_structured(customer_data):
    """
    Run the Agentic AI system and return structured JSON.
    """

    customer_json = json.dumps(customer_data)

    prompt = f"""
You are a Telecom Customer Churn AI Agent.

Analyze the following telecom customer:

{customer_json}

Use the available tools to:
1. Predict churn probability, prediction, and risk level.
2. Identify the main churn drivers and protective factors.
3. Generate retention recommendations.

Return ONLY valid JSON.

The JSON must follow exactly this structure:

{{
    "churn_prediction": "Churn or No Churn",
    "churn_probability": 0.0,
    "risk_level": "Low, Medium, or High",
    "churn_drivers": [],
    "protective_factors": [],
    "retention_actions": []
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "tools": [
                prediction_agent_tool,
                explanation_agent_tool,
                retention_agent_tool
            ]
        }
    )

    result = response.text.strip()

    # Remove markdown code fences if Gemini adds them
    if result.startswith("```"):
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

    return json.loads(result)


# Test customer
if __name__ == "__main__":

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 250.0
    }

    result = run_agent_structured(customer)

    print("\n===== AGENT RESPONSE =====")
    print(result)