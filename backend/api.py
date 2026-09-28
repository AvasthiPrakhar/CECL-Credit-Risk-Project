"""
FastAPI Backend Service for CECL Credit Risk Assessment & AI Governance.
Handles ML model inference for Probability of Default (PD) and Loss Given Default (LGD),
as well as LLM orchestration for automated Model Risk Management (MRM) reporting.
"""
import os
import json
import logging
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Configure application logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(title="CECL Credit Risk API", version="1.0.0")

# Configure CORS for decoupled frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-load ML artifacts at startup to reduce inference latency
try:
    pd_model = joblib.load('pd_model.joblib')
    lgd_model = joblib.load('lgd_model.joblib')
    scaler = joblib.load('scaler.joblib')
    logger.info("Machine Learning artifacts loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ML artifacts: {e}")
    raise RuntimeError("ML artifacts unavailable. Ensure .joblib files reside in the backend directory.")

class LoanData(BaseModel):
    """Pydantic schema for incoming loan data validation."""
    loan_amnt: float = Field(..., description="Total loan amount requested")
    int_rate: float = Field(..., description="Interest rate on the loan")
    installment: float = Field(..., description="Monthly payment owed by the borrower")
    annual_inc: float = Field(..., description="Self-reported annual income")
    dti: float = Field(..., description="Debt-to-income ratio")
    delinq_2yrs: float = Field(..., description="Number of 30+ days past-due incidences in the past 2 years")
    open_acc: float = Field(..., description="Number of open credit lines")
    pub_rec: float = Field(..., description="Number of derogatory public records")
    revol_util: float = Field(..., description="Revolving line utilization rate")
    total_acc: float = Field(..., description="Total number of credit lines currently in the borrower's credit file")

class ReportRequest(BaseModel):
    """Pydantic schema for LLM provider selection."""
    provider: str = Field(..., description="Selected LLM Provider (e.g., Groq, Google)")

@app.get("/")
def health_check():
    """Endpoint to verify API operational status."""
    return {"status": "healthy", "service": "CECL Risk API"}

@app.post("/predict_cecl")
def predict_cecl(loan: LoanData):
    """
    Calculates Expected Credit Loss (ECL) adhering to ASC 326 / CECL framework.
    Computes PD and LGD, applies macroeconomic scenario weightings, and returns risk metrics.
    """
    try:
        features = np.array([[
            loan.loan_amnt, loan.int_rate, loan.installment, loan.annual_inc, 
            loan.dti, loan.delinq_2yrs, loan.open_acc, loan.pub_rec, 
            loan.revol_util, loan.total_acc
        ]])
        scaled_features = scaler.transform(features)
        
        pd_val = float(pd_model.predict_proba(scaled_features)[0][1])
        lgd_val = float(lgd_model.predict(scaled_features)[0])
        ead_val = float(loan.loan_amnt) 
        
        ecl_base = pd_val * lgd_val * ead_val
        ecl_pessimistic = (pd_val * 1.2) * (lgd_val * 1.1) * ead_val
        ecl_optimistic = (pd_val * 0.8) * (lgd_val * 0.9) * ead_val
        
        final_cecl = (0.50 * ecl_base) + (0.30 * ecl_pessimistic) + (0.20 * ecl_optimistic)
        
        return {
            "PD": round(pd_val, 4),
            "LGD": round(lgd_val, 4),
            "Expected_Credit_Loss": round(final_cecl, 2)
        }
    except Exception as e:
        logger.error(f"Inference error during CECL prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during risk prediction.")

@app.post("/generate_mrm_report")
def generate_report(req: ReportRequest):
    """
    Automates Model Risk Management (MRM) documentation via orchestrated LLMs.
    Ingests portfolio metrics and methodology to dynamically generate audit compliance reports.
    """
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        logger.error("metrics.json file not found.")
        raise HTTPException(status_code=500, detail="Portfolio metrics data unavailable on server.")
        
    prompt = f"""
    You are a Senior Model Risk Auditor. Generate a 3-paragraph MRM report for a CECL model.
    Context Data: The model processed {metrics.get('Total_Rows_Processed', 'N/A')} loans. Total exposure is {metrics.get('Total_Portfolio_Exposure', 'N/A')}.
    Class imbalance was handled using SMOTE. Model uses XGBoost for PD and Random Forest for LGD.
    Explain the methodology and add a note on governance limitations. Make it highly professional.
    """

    try:
        # Orchestrate LLM based on client-side provider selection
        if req.provider == "Groq (LLaMA 3)":
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise HTTPException(status_code=500, detail="Groq API credential not configured.")
            llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_key, temperature=0.2)
            
        elif req.provider == "Groq (Mixtral)":
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise HTTPException(status_code=500, detail="Groq API credential not configured.")
            llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=groq_key, temperature=0.2)
            
        elif req.provider == "Google (Gemini)":
            google_key = os.getenv("GOOGLE_API_KEY")
            if not google_key:
                raise HTTPException(status_code=500, detail="Google API credential not configured.")
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_key, temperature=0.2)
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported LLM Provider selected.")
        
        response = llm.invoke(prompt)
        return {"report": response.content}
        
    except Exception as e:
        logger.error(f"LLM orchestration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
