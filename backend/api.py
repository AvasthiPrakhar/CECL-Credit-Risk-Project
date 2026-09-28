"""
FastAPI Backend Service for CECL Credit Risk Assessment & AI Governance.
"""
import os
import json
import logging
import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CECL Credit Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    pd_model = joblib.load('pd_model.joblib')
    lgd_model = joblib.load('lgd_model.joblib')
    scaler = joblib.load('scaler.joblib')
    logger.info("ML artifacts loaded.")
except Exception as e:
    logger.error(f"Failed to load ML artifacts: {e}")

class LoanData(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    delinq_2yrs: float
    open_acc: float
    pub_rec: float
    revol_util: float
    total_acc: float

class ReportRequest(BaseModel):
    model_id: str
    loan_params: dict = Field(default={}, description="The loan parameters used for inference")
    prediction_results: dict = Field(default={}, description="The inference results (PD, LGD, CECL)")

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
def get_model_metrics():
    try:
        with open("metrics.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Metrics file not found."}

@app.get("/models")
def get_live_models():
    models_list = []
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            url = "https://api.groq.com/openai/v1/models"
            headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for m in data:
                    m_id = m.get("id")
                    if "whisper" not in m_id.lower():
                        models_list.append({"id": m_id, "display_name": f"Groq ({m_id})"})
        except Exception as e:
            logger.error(f"Failed to fetch Groq models: {e}")
            
    models_list.append({"id": "gemini-3.8-flash", "display_name": "Google (gemini-3.8-flash)"})
    return {"models": models_list}

@app.post("/predict_cecl")
def predict_cecl(loan: LoanData):
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
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/generate_mrm_report")
def generate_report(req: ReportRequest):
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
        
    prompt = f"""
    You are a Senior Risk Auditor. Generate a Model Risk and Loan Decision Report.
    
    Model Training Context: Processed {metrics.get('Total_Rows_Processed', '1.3M+')} loans. Used SMOTE for class imbalance, XGBoost for PD, and Random Forest for LGD.
    
    Current Loan Application Parameters:
    {json.dumps(req.loan_params, indent=2)}
    
    Inference Results for this specific loan:
    {json.dumps(req.prediction_results, indent=2)}
    
    Write a highly professional 3-paragraph report explaining the methodology, interpreting the specific inference results for this borrower's parameters, and adding a note on governance limitations.
    """

    try:
        model_id = req.model_id
        if "gemini" in model_id.lower():
            google_key = os.getenv("GOOGLE_API_KEY")
            if not google_key:
                raise HTTPException(status_code=500, detail="Google key missing.")
            llm = ChatGoogleGenerativeAI(model=model_id, google_api_key=google_key, temperature=0.2)
        else:
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise HTTPException(status_code=500, detail="Groq key missing.")
            llm = ChatGroq(model_name=model_id, groq_api_key=groq_key, temperature=0.2)
        
        response = llm.invoke(prompt)
        return {"report": response.content}
        
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
