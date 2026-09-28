from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load Models
try:
    pd_model = joblib.load('pd_model.joblib')
    lgd_model = joblib.load('lgd_model.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    print(f"Error loading models: {e}. Ensure .joblib files are in the backend folder.")

app = FastAPI(title="CECL Credit Risk API", version="1.0")

# Enable CORS (Allows your Streamlit frontend to communicate with this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/predict_cecl")
def predict_cecl(loan: LoanData):
    # Convert input to 2D numpy array
    features = np.array([[loan.loan_amnt, loan.int_rate, loan.installment, loan.annual_inc, 
                          loan.dti, loan.delinq_2yrs, loan.open_acc, loan.pub_rec, 
                          loan.revol_util, loan.total_acc]])
    
    scaled_features = scaler.transform(features)
    
    # Predict PD and LGD (Casting to standard Python float for JSON compatibility)
    pd_val = float(pd_model.predict_proba(scaled_features)[0][1])
    lgd_val = float(lgd_model.predict(scaled_features)[0])
    ead_val = float(loan.loan_amnt) 
    
    # Calculate Scenarios
    ecl_base = pd_val * lgd_val * ead_val
    ecl_pessimistic = (pd_val * 1.2) * (lgd_val * 1.1) * ead_val
    ecl_optimistic = (pd_val * 0.8) * (lgd_val * 0.9) * ead_val
    
    # Final Probability-Weighted CECL (ASC 326)
    final_cecl = (0.50 * ecl_base) + (0.30 * ecl_pessimistic) + (0.20 * ecl_optimistic)
    
    return {
        "PD": round(pd_val, 4),
        "LGD": round(lgd_val, 4),
        "Expected_Credit_Loss": round(final_cecl, 2)
    }

@app.post("/generate_mrm_report")
def generate_report():
    # Securely fetch the API key from environment variables
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise HTTPException(status_code=500, detail="Google API Key is not configured on the server.")
    
    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=google_api_key, temperature=0.2)
        
        prompt = f"""
        You are a Senior Model Risk Auditor. Generate a 3-paragraph MRM report for a CECL model.
        Context Data: The model processed {metrics['Total_Rows_Processed']} loans. Total exposure is {metrics['Total_Portfolio_Exposure']}.
        Class imbalance was handled using SMOTE. Model uses XGBoost for PD and Random Forest for LGD.
        Explain the methodology and add a note on governance limitations. Make it highly professional.
        """
        
        response = llm.invoke(prompt)
        return {"report": response.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
