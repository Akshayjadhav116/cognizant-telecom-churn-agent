from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from agent import run_agent_structured


app = FastAPI(
    title="Telecom Customer Churn AI",
    description="Agentic AI system for telecom customer churn prediction",
    version="1.0.0"
)


class CustomerRequest(BaseModel):
    customer: Dict[str, Any]


@app.get("/")
def home():
    return {
        "message": "Telecom Customer Churn AI is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/analyze")
def analyze_customer(request: CustomerRequest):

    result = run_agent_structured(request.customer)

    return {
        "status": "success",
        "analysis": result
    }